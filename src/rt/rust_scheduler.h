#ifndef RUST_SCHEDULER_H
#define RUST_SCHEDULER_H

#include "rust_internal.h"

class rust_scheduler : public kernel_owned<rust_scheduler> {
    // FIXME: Make these private
public:
    rust_kernel *kernel;
    rust_srv *srv;
    rust_env *env;
private:
    // Protects the random number context and live_threads
    lock_and_signal lock;
    // When this hits zero we'll tell the kernel to release us
    uintptr_t live_threads;
    // When this hits zero we'll tell the threads to exit
    uintptr_t live_tasks;
    randctx rctx;

    array_list<rust_task_thread *> threads;
    const size_t num_threads;

    rust_sched_id id;

    void create_task_threads();
    void destroy_task_threads();

    rust_task_thread *create_task_thread(int id);
    void destroy_task_thread(rust_task_thread *thread);

    void exit();

public:
    rust_scheduler(rust_kernel *kernel, rust_srv *srv, size_t num_threads,
		   rust_sched_id id);
    ~rust_scheduler();

    void start_task_threads();
    void join_task_threads();
    void kill_all_tasks();
    rust_task* create_task(rust_task *spawner,
			   const char *name,
			   size_t init_stack_sz);
    rust_task* create_task(rust_task *spawner, const char *name);

    void release_task();

    size_t number_of_threads();
    // Called by each thread when it terminates. When all threads
    // terminate the scheduler does as well.
    void release_task_thread();

    rust_sched_id get_id() { return id; }
};

#endif /* RUST_SCHEDULER_H */
