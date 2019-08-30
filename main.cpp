#include "ThreadPool.h"
#include <armadillo>
#include <chrono>
#include <future>
#include <iostream>

using Clock = std::chrono::steady_clock;
using namespace std::chrono;

#if defined(ARMA_USE_EXTERN_CXX11_RNG)
namespace arma {
thread_local arma_rng_cxx11 arma_rng_cxx11_instance; // NOLINT
} // namespace arma
#endif

arma::mat func(const arma::mat &A, const arma::mat &B) {
  arma::mat U_A;
  arma::vec s_A;
  arma::mat V_A;
  arma::svd(U_A, s_A, V_A, A);

  arma::mat U_B;
  arma::vec s_B;
  arma::mat V_B;
  arma::svd(U_B, s_B, V_B, B);

  arma::mat U_other;
  arma::vec s_other;
  arma::mat V_other;
  arma::svd(U_other, s_other, V_other, (A * B - B * U_A));
  return U_other * (A * B - B * U_A) * (A * (A - U_B) * (A + B) * V_B * A) *
             (-A * B + B * V_A) +
         V_other;
}

int main() {
  unsigned int nummultiplications = 500000;

  ThreadPool pool(4);

  arma::mat A{{1, 2, 3, 7, 2, 8}, {4, 5, 1, 9, 2, 5}, {0, 0, 2, 7, 8, 10},
              {1, 1, 1, 1, 1, 1}, {2, 3, 2, 4, 5, 5}, {5, 8, 9, 0, 9, 9}};
  A = A / arma::norm(A);
  arma::mat B{{1, 2, 2, -4, 3, 1},      {5, -6, -7, 1, 1, -7},
              {3, 3, 9, -10, 11, 12},   {11, 12, 13, 14, 15, 16},
              {17, 18, 19, 20, 21, 22}, {-10, -10, -10, -10, -10, -10}};
  B = B / arma::norm(B);
  arma::mat expectedResult = func(A, B);

  std::vector<std::future<arma::mat>> futures;
  futures.reserve(nummultiplications);

  auto tic = Clock::now();
  unsigned int numDiferentes = 0;

  for (unsigned int i = 0; i < nummultiplications; i++) {
    futures.push_back(pool.enqueue(func, A, B));
     //auto result = func(A, B);
    // bool ok = arma::approx_equal(result, expectedResult, "absdiff", 1e-6);
    // if (!ok) {
    //  ++numDiferentes;
    //  std::cout << "Diferente" << std::endl;
    //  result.print("result");
    //  expectedResult.print("expected");
    //}
  }

  std::cout << "Todos os futuros na fila" << std::endl;

  for (auto &f : futures) {
    auto result = f.get();
    bool ok = arma::approx_equal(result, expectedResult, "absdiff", 1e-6);
    if (!ok) {
      ++numDiferentes;
      //std::cout << "Diferente" << std::endl;
      //result.print("result");
      //expectedResult.print("expected");
    }
  }

  std::cout << "Fim" << std::endl;

  auto toc = Clock::now();
  std::cout << "Elapsed time: "
            << duration_cast<milliseconds>(toc - tic).count() / 1000.0
            << std::endl;

  std::cout << "numDiferentes: " << numDiferentes << "/" << nummultiplications << std::endl;
  return 0;
}
