/**
   The rust scheduler. Schedulers may be added to the kernel
   dynamically and they run until there are no more tasks to
   schedule. Most of the scheduler work is carried out in worker
   threads by rust_sched_loop.
 */

#ifndef RUST_SCHEDULER_H
#define RUST_SCHEDULER_H

#include "rust_globals.h"
#include "util/array_list.h"
#include "rust_kernel.h"

class rust_sched_launcher;
class rust_sched_launcher_factory;

class rust_scheduler : public kernel_owned<rust_scheduler> {
    // FIXME: Make these private
public:
    rust_kernel *kernel;
private:
    // Protects live_threads, live_tasks, cur_thread, may_exit
    lock_and_signal lock;
    // When this hits zero we'll tell the kernel to release us
    uintptr_t live_threads;
    // When this hits zero we'll tell the threads to exit
    uintptr_t live_tasks;
    size_t cur_thread;
    bool may_exit;

    array_list<rust_sched_launcher *> threads;
    const size_t num_threads;

    rust_sched_id id;

    void create_task_threads(rust_sched_launcher_factory *launchfac);
    void destroy_task_threads();

    rust_sched_launcher *
    create_task_thread(rust_sched_launcher_factory *launchfac, int id);
    void destroy_task_thread(rust_sched_launcher *thread);

    void exit();

public:
    rust_scheduler(rust_kernel *kernel, size_t num_threads,
                   rust_sched_id id, bool allow_exit,
                   rust_sched_launcher_factory *launchfac);
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
    // Tells the scheduler that as soon as it runs out of tasks
    // to run it should exit
    void allow_exit();
    void disallow_exit();
};

#endif /* RUST_SCHEDULER_H */
