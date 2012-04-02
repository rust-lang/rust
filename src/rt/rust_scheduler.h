#ifndef RUST_SCHEDULER_H
#define RUST_SCHEDULER_H

#include "rust_internal.h"

class rust_sched_launcher;

class rust_scheduler : public kernel_owned<rust_scheduler> {
    // FIXME: Make these private
public:
    rust_kernel *kernel;
private:
    // Protects live_threads and cur_thread increments
    lock_and_signal lock;
    // When this hits zero we'll tell the kernel to release us
    uintptr_t live_threads;
    // When this hits zero we'll tell the threads to exit
    uintptr_t live_tasks;

    array_list<rust_sched_launcher *> threads;
    const size_t num_threads;
    size_t cur_thread;

    rust_sched_id id;

    void create_task_threads();
    void destroy_task_threads();

    rust_sched_launcher *create_task_thread(int id);
    void destroy_task_thread(rust_sched_launcher *thread);

    void exit();

public:
    rust_scheduler(rust_kernel *kernel, size_t num_threads,
                   rust_sched_id id);
    ~rust_scheduler();

    void start_task_threads();
    void join_task_threads();
    void kill_all_tasks();
    rust_task* create_task(rust_task *spawner, const char *name);

    void release_task();

    size_t number_of_threads();
    // Called by each thread when it terminates. When all threads
    // terminate the scheduler does as well.
    void release_task_thread();

    rust_sched_id get_id() { return id; }
};

#endif /* RUST_SCHEDULER_H */
