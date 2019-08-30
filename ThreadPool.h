#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <gsl/gsl_util>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

/**
 * Implementation of a Thread Pool
 *
 * From: http://progsch.net/wordpress/?p=81
 *       https://github.com/progschj/ThreadPool
 *
 */
class ThreadPool {
public:
    /**
     * Thread Pool constructor
     * Starts the Thread Pool by launching the requested amount of workers.
     *
     * @param numOfThreads - The number of threads that will consume the task queue
     */
    explicit ThreadPool(size_t numOfThreads = std::thread::hardware_concurrency());

    /**
     * Thread Pool destructor
     */
    ~ThreadPool();

    /**
     * Add a task in the queue.
     * This task may be a function with parameters, or a lambda function.
     *
     * This method returns a future object which holds the result of the task that may be
     * accessed asynchronously.
     *
     * Example:
     *      class Foo {
     *          public:
     *              output_type bar(input_type input);
     *      }
     *
     *      Foo foo;
     *      input_type in;
     *      future<output_type> futureResult = pool.enqueue(&Foo::bar, foo, in);
     *      ...
     *      ...
     *      ...
     *      output_type res = futureResult.get();
     *
     * @param fun - Task to be executed
     * @param args - Arguments of the task fun
     * @return a future object holding the result of the task fun
     *
     */
    template <class F, class... Args>
    auto enqueue(F&& fun, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * Map the function to each element of a container using an start and an ending iterators and put the result in the
     * an output container with the given iterator.
     *
     * The output iterator must be of a container whose element type is the same as the return type of the mapped
     * function "fun".
     * **Note**: The output container must have enough space for the result.
     *
     * @tparam F The function template type.
     * @tparam InputIt The input iterator template type.
     * @tparam OutputIt The output iterator template type.
     * @param fun The function to call with each element of the container.
     * @param first The initial iterator of the input container.
     * @param last The end iterator of the input container.
     * @param out The initial iterator of the output container.
     */
    template <class F, typename InputIt, typename OutputIt>
    void map_container(F fun, InputIt first, InputIt last, OutputIt out);

    /**
     * Map the function to each element of a container using an start and an ending iterators.
     *
     * The result is a vector whose element types are of the same as the return type of the mapped function.
     *
     * @tparam F The function template type.
     * @tparam InputIt The input iterator template type.
     * @param fun The function to call with each element of the container.
     * @param first The initial iterator of the input container.
     * @param last The end iterator of the input container.
     * @return A vector with the result of applying the function to each element of the container.
     */
    template <class F, typename InputIt>
    auto map_container(F fun, InputIt first, InputIt last) -> std::vector<decltype(fun(*first))>;

    /**
     * Map the function to each element of a container.
     * @tparam F The function template type.
     * @tparam Container The input container template type.
     * @param fun The function to call with each element of the container.
     * @param container The input container.
     * @return A vector with the result of applying the function to each element of the container.
     */
    template <class F, class Container>
    auto map_container(F fun, Container container);

private:
    /// Keep track of threads so we can join them
    std::vector<std::thread> workers;

    /// Task queue
    std::queue<std::function<void()>> tasks;

    /// Mutex used to control the access to the task queue
    std::mutex queue_mutex;

    /// Conditional Variable used to synchronize the access to the task queue
    std::condition_variable condition;

    /// Determine if the Thread Pool is enqueueing jobs
    bool stop;
};

inline ThreadPool::ThreadPool(size_t numOfThreads) : stop(false) {
    // Create the workers, making then consume the tasks of the queue until the pool is released
    for(size_t i = 0; i < numOfThreads; ++i) {
        workers.emplace_back([this] {
            while(true) {
                std::function<void()> task;

                {
                    // Lock the current worker -> we use std::unique_lock instead of lock_guard to lock the mutex in
                    // order to alow the condition.wait method to release the mutex while it waits. Note that when the
                    // wait is done the condition.wait method will lock the mutex again.
                    std::unique_lock<std::mutex> lock(this->queue_mutex);

                    // Keep the worker waiting until the pool is released (destroyed) or some task is added to the queue
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                    // If there is no more tasks to execute and the pool is released, then release the worker
                    if(this->stop && this->tasks.empty()) {
                        return;
                    }

                    // Get the first task in the queue
                    task = move(this->tasks.front());

                    // Remove the retrieved task from the queue
                    this->tasks.pop();
                }

                // Execute task in the worker
                task();
            }
        });
    }
}

template <class F, class... Args>
auto ThreadPool::enqueue(F&& fun, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    // Create an alias to the return type
    using return_type = typename std::invoke_result<F, Args...>::type;

    // Create a shared pointer to the task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(fun), std::forward<Args>(args)...));

    // Get the future object of the task
    std::future<return_type> res = task->get_future();

    {
        // Lock the task queue
        std::lock_guard<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        // Add the task to the queue
        tasks.emplace([task]() { (*task)(); });
    }

    // Notify one thread that is waiting for tasks
    condition.notify_one();

    // Return the future object
    return res;
}

inline ThreadPool::~ThreadPool() {
    // Lock the queue task and stop the pool
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    // Notify all threads
    condition.notify_all();

    // Wait the workers to finish
    for(std::thread& worker : workers) {
        worker.join();
    }
}

template <class F, typename InputIt, typename OutputIt>
void ThreadPool::map_container(F fun, InputIt first, InputIt last, OutputIt out) {
    using Element_t       = decltype(*first);       // Type of the elements of the input container
    using ResultElement_t = decltype(fun(*first));  // Type of the elements of the result we will return

    // Check if the iterator of the output is of correct type (same as the return type of "fun"
    static_assert(std::is_same<ResultElement_t, typename std::iterator_traits<OutputIt>::value_type>::value,
                  "Type of the Output iterator must match the return value of 'fun'.");

    // The second iterator must come after the first iterator
    Ensures(last > first);
    // Number of elements of the input container is given by "last-first"
    auto numElements = gsl::narrow_cast<unsigned long>(last - first);

    // Create a vector for the future results. We use std::transform to enqueue in the ThreadPool a lambda function
    // that calls "fun" for each element in the container (using the iterators)
    std::vector<std::future<ResultElement_t>> future_results(numElements);
    std::transform(first, last, future_results.begin(),
                   [this, &fun](const Element_t& elem) { return enqueue(std::forward<F>(fun), elem); });

    // We use std::transform to call the "get" method of each future result in "future_results" so that we get the
    // actual values that we can return.
    // Note: The output container must have enough space for all results
    std::transform(future_results.begin(), future_results.end(), out, [](auto& e) { return e.get(); });
}

template <class F, typename InputIt>
auto ThreadPool::map_container(F fun, InputIt first, InputIt last) -> std::vector<decltype(fun(*first))> {
    using ResultElement_t = decltype(fun(*first));  // Type of the elements of the result we will return

    auto numElements = std::distance(first, last);
    if(numElements < 0) throw std::runtime_error("The 'last' iterator must come after the 'first' iterator");

    // Output vector with "last - first" elements
    std::vector<ResultElement_t> results(gsl::narrow_cast<unsigned long>(numElements));
    // Call previous implementation of map_container with the input iterators and an iterator to our output vector
    map_container(std::forward<F>(fun), std::forward<InputIt>(first), std::forward<InputIt>(last),
                  std::forward<decltype(results.begin())>(results.begin()));
    return std::forward<std::vector<ResultElement_t>>(results);
}

template <class F, class Container>
auto ThreadPool::map_container(F fun, Container container) {
    using Iter_t = decltype(container.begin());
    return map_container(std::forward<F>(fun), std::forward<Iter_t>(container.begin()),
                         std::forward<Iter_t>(container.end()));
}

#endif
