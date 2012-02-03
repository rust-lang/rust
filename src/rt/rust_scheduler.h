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
    lock_and_signal lock;
    array_list<rust_task_thread *> threads;
    randctx rctx;
    const size_t num_threads;
    int rval;

    void create_task_threads();
    void destroy_task_threads();

    rust_task_thread *create_task_thread(int id);
    void destroy_task_thread(rust_task_thread *thread);

public:
    rust_scheduler(rust_kernel *kernel, rust_srv *srv, size_t num_threads);
    ~rust_scheduler();

    void start_task_threads();
    void kill_all_tasks();
    rust_task_id create_task(rust_task *spawner,
			     const char *name,
			     size_t init_stack_sz);
    rust_task_id create_task(rust_task *spawner, const char *name);
    void exit();
    size_t number_of_threads();
};

#endif /* RUST_SCHEDULER_H */
