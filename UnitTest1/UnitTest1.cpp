#include "pch.h" 
#include "CppUnitTest.h"
#include "../poisson.hpp" // Путь к файлу 

#include <cmath>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PoissonTests
{
    TEST_CLASS(PoissonUnitTests)
    {
    public:
        TEST_METHOD(TestLinearExact)
        {
            poisson::Rectangle grid;
            grid.nx = 64; grid.ny = 64;

            auto exact = [](double x, double y) { return 2.0 * x + 3.0 * y; };
            auto boundary = exact;
            auto rhs = [](double x, double y) { return 0.0; };

            poisson::Options opt;
            opt.atol = 1e-10;

            auto sol = poisson::solve(grid, rhs, boundary, opt);

            double max_error = 0.0;
            for (int j = 0; j <= grid.ny; ++j) {
                for (int i = 0; i <= grid.nx; ++i) {
                    double x = grid.x(i), y = grid.y(j);
                    max_error = std::max(max_error, std::abs(sol.at(i, j) - exact(x, y)));
                }
            }

            Assert::IsTrue(max_error < 1e-10, L"Ошибка линейного теста слишком велика!");
        }

        TEST_METHOD(TestQuadraticExact)
        {
            poisson::Rectangle grid;
            grid.nx = 64; grid.ny = 64;

            auto exact = [](double x, double y) { return x * x + y * y; };
            auto boundary = exact;
            auto rhs = [](double x, double y) { return -4.0; };

            poisson::Options opt;
            opt.atol = 1e-10;

            auto sol = poisson::solve(grid, rhs, boundary, opt);

            double max_error = 0.0;
            for (int j = 0; j <= grid.ny; ++j) {
                for (int i = 0; i <= grid.nx; ++i) {
                    max_error = std::max(max_error, std::abs(sol.at(i, j) - exact(grid.x(i), grid.y(j))));
                }
            }

            Assert::IsTrue(max_error < 1e-10, L"Ошибка квадратичного теста слишком велика!");
        }

        TEST_METHOD(TestInterpolation)
        {
            poisson::Rectangle grid;
            grid.nx = 64; grid.ny = 64;

            auto exact = [](double x, double y) { return x + y; };
            auto rhs = [](double x, double y) { return 0.0; };

            auto sol = poisson::solve(grid, rhs, exact, {});

            double x_mid = grid.x(5) + (grid.x(6) - grid.x(5)) / 2.0;
            double y_mid = grid.y(5) + (grid.y(6) - grid.y(5)) / 2.0;

            double interpolated_val = sol(x_mid, y_mid);
            double exact_val = exact(x_mid, y_mid);

            Assert::IsTrue(std::abs(interpolated_val - exact_val) < 1e-9, L"Ошибка интерполяции!");
        }

        TEST_METHOD(TestInvalidInput)
        {
            poisson::Rectangle grid;
            grid.nx = 1; // Меньше 2 быть не может
            auto valid_func = [](double, double) { return 0.0; };

            // Проверяем, что вызов бросит исключение std::invalid_argument
            Assert::ExpectException<std::invalid_argument>([&] {
                poisson::solve(grid, valid_func, valid_func, {});
                });

            // Проверка неверного omega
            grid.nx = 10;
            poisson::Options opt;
            opt.omega = 2.5;

            Assert::ExpectException<std::invalid_argument>([&] {
                poisson::solve(grid, valid_func, valid_func, opt);
                });
        }

        TEST_METHOD(TestCubicCos)
        {
            poisson::Rectangle grid;
            grid.nx = 16; grid.ny = 16;
            const double pi = std::acos(-1.0);
            grid.x0 = 0.0; grid.x1 = pi / 2.0;
            grid.y0 = 0.0; grid.y1 = pi / 2.0;

            auto exact = [](double x, double y) {
                return x * x * x + x * y + std::cos(y);
                };

            auto boundary = exact;
            auto rhs = [](double x, double y) {
                return std::cos(y) - 6.0 * x;
                };

            poisson::Options opt;
            opt.atol = 1e-10;

            auto sol = poisson::solve(grid, rhs, boundary, opt);

            double max_error = 0.0;
            for (int j = 0; j <= grid.ny; ++j) {
                for (int i = 0; i <= grid.nx; ++i) {
                    double x = grid.x(i);
                    double y = grid.y(j);
                    max_error = std::max(max_error, std::abs(sol.at(i, j) - exact(x, y)));
                }
            }

            Assert::IsTrue(max_error < 1e-3, L"Ошибка кубического теста (схема O(h^2)) превышает ожидаемую!");
        }
    };
}