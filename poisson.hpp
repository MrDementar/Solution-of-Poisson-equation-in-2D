#ifndef POISSON_HPP
#define POISSON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace poisson {

using Field = std::function<double(double, double)>;

// nx and ny count mesh intervals, so the solution has (nx+1)*(ny+1) nodes.
struct Rectangle {
    double x0 = 0.0, x1 = 1.0, y0 = 0.0, y1 = 1.0;
    int nx = 64, ny = 64;

    double x(int i) const { return i == nx ? x1 : x0 + (x1 - x0) * (double(i) / nx); }
    double y(int j) const { return j == ny ? y1 : y0 + (y1 - y0) * (double(j) / ny); }
};

struct Options {
    double omega = 0.0;  // Zero selects the rectangular-grid SOR estimate.
    double atol = 1e-8, rtol = 0.0;
    int max_iterations = 20000;
    int check_every = 10;
    int tile_size = 16;
    int threads = 0;     // Zero uses the OpenMP runtime setting.
};

struct Solution {
    Rectangle grid;
    std::vector<double> values;  // values[j*(nx+1) + i]
    int iterations = 0;
    double omega = 0.0;
    double initial_residual_inf = 0.0, residual_inf = 0.0;
    bool converged = false;

    double at(int i, int j) const {
        if (i < 0 || i > grid.nx || j < 0 || j > grid.ny)
            throw std::out_of_range("Grid index outside rectangle");
        return values.at(std::size_t(j) * (std::size_t(grid.nx) + 1) + i);
    }

    // Bilinear interpolant, including all four edges and corners.
    double operator()(double x, double y) const {
        if (!(x >= grid.x0 && x <= grid.x1 && y >= grid.y0 && y <= grid.y1))
            throw std::out_of_range("Point outside rectangle");
        const double sx = (x - grid.x0) / (grid.x1 - grid.x0) * grid.nx;
        const double sy = (y - grid.y0) / (grid.y1 - grid.y0) * grid.ny;
        const int i = std::min(static_cast<int>(sx), grid.nx - 1);
        const int j = std::min(static_cast<int>(sy), grid.ny - 1);
        const double a = sx - i, b = sy - j;
        const double lower = (1.0 - a) * at(i, j) + a * at(i + 1, j);
        const double upper = (1.0 - a) * at(i, j + 1) + a * at(i + 1, j + 1);
        return (1.0 - b) * lower + b * upper;
    }
};

// Solves -Delta u = rhs with u = boundary on the edge of grid.
// Callbacks are sampled once on the calling thread before the parallel sweeps.
inline Solution solve(const Rectangle& grid, const Field& rhs, const Field& boundary,
                      const Options& opt = {}, const Field& initial = {}) {
    const int nx = grid.nx, ny = grid.ny;
    if (!rhs || !boundary || nx < 2 || ny < 2 ||
        nx == std::numeric_limits<int>::max() || ny == std::numeric_limits<int>::max())
        throw std::invalid_argument("Supply rhs, boundary and at least two intervals per axis");
    const double hx = (grid.x1 - grid.x0) / nx, hy = (grid.y1 - grid.y0) / ny;
    if (!(hx > 0.0 && hy > 0.0 && std::isfinite(hx) && std::isfinite(hy)))
        throw std::invalid_argument("Rectangle must have finite positive mesh spacings");
    if (!(opt.omega >= 0.0 && opt.omega < 2.0) ||
        !std::isfinite(opt.atol) || !std::isfinite(opt.rtol) ||
        opt.atol < 0.0 || opt.rtol < 0.0 || opt.max_iterations < 0 ||
        opt.check_every < 1 || opt.tile_size < 1 || opt.threads < 0)
        throw std::invalid_argument("Invalid relaxation, tolerance, iteration or thread setting");

    const double ax = (1.0 / hx) / hx, ay = (1.0 / hy) / hy;
    const double diagonal = 2.0 * (ax + ay);
    if (!(ax > 0.0 && ay > 0.0 && std::isfinite(diagonal)))
        throw std::invalid_argument("Mesh coefficients outside floating-point range");
    Solution sol;
    sol.grid = grid;
    sol.omega = opt.omega;
    if (sol.omega == 0.0) {
        const double half_pi = std::acos(-1.0) / 2.0;
        const double sx = std::sin(half_pi / nx), sy = std::sin(half_pi / ny);
        // delta = 1 - spectral radius of the Jacobi iteration matrix.
        const double delta = 2.0 * ((ax / (ax + ay)) * sx * sx + (ay / (ax + ay)) * sy * sy);
        sol.omega = 2.0 / (1.0 + std::sqrt(delta * (2.0 - delta)));
    }

    const std::size_t stride = std::size_t(nx) + 1, rows = std::size_t(ny) + 1;
    if (rows > sol.values.max_size() / stride)
        throw std::length_error("Mesh allocation exceeds addressable memory");
    sol.values.resize(stride * rows);
    std::vector<double> f(sol.values.size());
    auto& u = sol.values;
    for (int j = 0; j <= ny; ++j) {
        const double y = grid.y(j);
        for (int i = 0; i <= nx; ++i) {
            const double x = grid.x(i);
            const std::size_t k = std::size_t(j) * stride + i;
            if (i == 0 || j == 0 || i == nx || j == ny) {
                u[k] = boundary(x, y);
            } else {
                f[k] = rhs(x, y);
                if (initial) u[k] = initial(x, y);
            }
            if (!std::isfinite(u[k]) || !std::isfinite(f[k]))
                throw std::domain_error("A problem function returned a non-finite value");
        }
    }

    const auto residual = [&](std::size_t k) {
        return f[k] + ax * ((u[k - 1] - u[k]) + (u[k + 1] - u[k]))
                    + ay * ((u[k - stride] - u[k]) + (u[k + stride] - u[k]));
    };
    const auto row_residual = [&](int j) {
        double norm = 0.0;
        for (int i = 1; i < nx; ++i) {
            const double value = std::abs(residual(std::size_t(j) * stride + i));
            if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
            norm = std::max(norm, value);
        }
        return norm;
    };
    for (int j = 1; j < ny; ++j)
        sol.residual_inf = std::max(sol.residual_inf, row_residual(j));
    sol.initial_residual_inf = sol.residual_inf;
    const double tolerance = opt.atol + opt.rtol * sol.initial_residual_inf;
    if (!std::isfinite(sol.residual_inf) || !std::isfinite(tolerance))
        throw std::overflow_error("Initial residual or stopping threshold overflows");
    sol.converged = sol.residual_inf <= tolerance;
    if (sol.converged || opt.max_iterations == 0) return sol;

    const int bx = 1 + (nx - 2) / opt.tile_size, by = 1 + (ny - 2) / opt.tile_size;
    const int requested = opt.threads == 0 ? omp_get_max_threads() : opt.threads;
    const int team = std::min(requested, std::min(bx, by));
    std::vector<double> thread_norm(team, 0.0);

    #pragma omp parallel num_threads(team) if(team > 1)
    {
        for (int step = 1; ; ++step) {
            // A tile needs fresh values from the tiles to its west and south.
            // Tiles with equal tx+ty share no edge. The omp-for barrier
            // completes their updates before the next diagonal can start.
            for (long long d = 0; d < static_cast<long long>(bx) + by - 1; ++d) {
                const int first = static_cast<int>(std::max(0LL, d - (by - 1)));
                const int last = static_cast<int>(std::min(d, static_cast<long long>(bx - 1)));
                #pragma omp for schedule(static)
                for (int tx = first; tx <= last; ++tx) {
                    const int ty = static_cast<int>(d - tx);
                    const int i0 = 1 + tx * opt.tile_size, j0 = 1 + ty * opt.tile_size;
                    const int i1 = i0 + std::min(opt.tile_size, nx - i0);
                    const int j1 = j0 + std::min(opt.tile_size, ny - j0);
                    for (int j = j0; j < j1; ++j) {
                        for (int i = i0; i < i1; ++i) {
                            const std::size_t k = std::size_t(j) * stride + i;
                            u[k] += sol.omega * (residual(k) / diagonal);
                        }
                    }
                }
            }
            if (step % opt.check_every == 0 || step == opt.max_iterations) {
                double local_norm = 0.0;
                #pragma omp for schedule(static) nowait
                for (int j = 1; j < ny; ++j)
                    local_norm = std::max(local_norm, row_residual(j));
                thread_norm[omp_get_thread_num()] = local_norm;
                #pragma omp barrier
                #pragma omp single
                {
                    sol.iterations = step;
                    sol.residual_inf = *std::max_element(thread_norm.begin(), thread_norm.end());
                    sol.converged = sol.residual_inf <= tolerance;
                }
                if (sol.converged || !std::isfinite(sol.residual_inf) || step == opt.max_iterations)
                    break;
            }
        }
    }
    if (!std::isfinite(sol.residual_inf))
        throw std::overflow_error("Non-finite residual during relaxation");
    return sol;
}

}  // namespace poisson
#endif
