#include "poisson.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_help() {
    std::cout <<
        "Solves -Delta u = f with Dirichlet data on a rectangle.\n"
        "Test field: u = 1+s+2*t+sin(pi*s)*sin(pi*t), with normalized s,t.\n"
        "Usage: poisson_demo [options]\n"
        "  --nx N --ny N           Numbers of intervals (defaults: 64, 64)\n"
        "  --x0 X --x1 X           x limits (defaults: 0, 1)\n"
        "  --y0 Y --y1 Y           y limits (defaults: 0, 1)\n"
        "  --omega W               Relaxation factor in (0,2), or 0 for auto\n"
        "  --atol A --rtol R       Stop at ||r||inf <= A + R*||r0||inf\n"
        "                          Defaults: A=1e-8, R=0\n"
        "  --max-iterations N      Sweep limit (default: 20000)\n"
        "  --check-every N         Residual check interval (default: 10 sweeps)\n"
        "  --tile-size N           Tile side in interior nodes (default: 16)\n"
        "  --threads N             OpenMP thread limit (default: runtime setting)\n"
        "  --csv PATH              Write x,y,u,u_exact for every mesh node\n";
}

double read_double(const std::string& text) {
    std::size_t end = 0;
    const double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid number: " + text);
    return value;
}

int read_int(const std::string& text) {
    std::size_t end = 0;
    const int value = std::stoi(text, &end);
    if (end != text.size())
        throw std::invalid_argument("Invalid integer: " + text);
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        poisson::Rectangle grid;
        poisson::Options options;
        std::string csv_path;
        for (int arg = 1; arg < argc; ++arg) {
            const std::string key = argv[arg];
            if (key == "--help" || key == "-h") {
                print_help();
                return EXIT_SUCCESS;
            }
            if (++arg >= argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[arg];
            if      (key == "--nx")             grid.nx = read_int(value);
            else if (key == "--ny")             grid.ny = read_int(value);
            else if (key == "--x0")             grid.x0 = read_double(value);
            else if (key == "--x1")             grid.x1 = read_double(value);
            else if (key == "--y0")             grid.y0 = read_double(value);
            else if (key == "--y1")             grid.y1 = read_double(value);
            else if (key == "--omega")          options.omega = read_double(value);
            else if (key == "--atol")           options.atol = read_double(value);
            else if (key == "--rtol")           options.rtol = read_double(value);
            else if (key == "--max-iterations") options.max_iterations = read_int(value);
            else if (key == "--check-every")    options.check_every = read_int(value);
            else if (key == "--threads")        options.threads = read_int(value);
            else if (key == "--tile-size")      options.tile_size = read_int(value);
            else if (key == "--csv")            csv_path = value;
            else throw std::invalid_argument("Unknown option: " + key);
        }

        const double lx = grid.x1 - grid.x0, ly = grid.y1 - grid.y0;
        const double pi = std::acos(-1.0);
        const auto wave = [&](double x, double y) {
            return std::sin(pi * ((x - grid.x0) / lx)) * std::sin(pi * ((y - grid.y0) / ly));
        };
        const auto boundary = [&](double x, double y) {
            return 1.0 + (x - grid.x0) / lx + 2.0 * ((y - grid.y0) / ly);
        };
        const auto exact = [&](double x, double y) { return boundary(x, y) + wave(x, y); };
        const auto rhs = [&](double x, double y) {
            return pi * pi * ((1.0 / lx) / lx + (1.0 / ly) / ly) * wave(x, y);
        };
        const auto sol = poisson::solve(grid, rhs, boundary, options);

        std::ofstream csv;
        if (!csv_path.empty()) {
            csv.exceptions(std::ios::badbit | std::ios::failbit);
            csv.open(csv_path);
            csv << std::setprecision(17) << "x,y,u,u_exact\n";
        }
        double error = 0.0;
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                const double x = grid.x(i), y = grid.y(j);
                const double reference = exact(x, y), numeric = sol.at(i, j);
                error = std::max(error, std::abs(numeric - reference));
                if (csv.is_open()) csv << x << ',' << y << ',' << numeric << ',' << reference << '\n';
            }
        }
        if (csv.is_open()) csv.close();
        const double cx = grid.x0 + lx / 2.0, cy = grid.y0 + ly / 2.0;
        std::cout << std::setprecision(12) << std::boolalpha
                  << "converged=" << sol.converged << '\n'
                  << "iterations=" << sol.iterations << '\n'
                  << "omega=" << sol.omega << '\n'
                  << "residual_inf=" << sol.residual_inf << '\n'
                  << "max_nodal_error=" << error << '\n'
                  << "u(center)=" << sol(cx, cy) << '\n';
        return sol.converged ? EXIT_SUCCESS : 2;
    } catch (const std::exception& err) {
        std::cerr << "poisson_demo: " << err.what() << '\n';
        return EXIT_FAILURE;
    }
}
