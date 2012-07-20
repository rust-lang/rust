
#include "rust_globals.h"
#include "rust_scheduler.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_sched_launcher.h"

rust_scheduler::rust_scheduler(rust_kernel *kernel,
                               size_t num_threads,
                               rust_sched_id id,
                               bool allow_exit,
                               bool killed,
                               rust_sched_launcher_factory *launchfac) :
    ref_count(1),
    kernel(kernel),
    live_threads(num_threads),
    live_tasks(0),
    cur_thread(0),
    may_exit(allow_exit),
    num_threads(num_threads),
    id(id)
{
    create_task_threads(launchfac, killed);
}

void rust_scheduler::delete_this() {
    destroy_task_threads();
    delete this;
}

rust_sched_launcher *
rust_scheduler::create_task_thread(rust_sched_launcher_factory *launchfac,
                                   int id, bool killed) {
    rust_sched_launcher *thread = launchfac->create(this, id, killed);
    KLOG(kernel, kern, "created task thread: " PTR ", id: %d",
          thread, id);
    return thread;
}

void
rust_scheduler::destroy_task_thread(rust_sched_launcher *thread) {
    KLOG(kernel, kern, "deleting task thread: " PTR, thread);
    delete thread;
}

void
rust_scheduler::create_task_threads(rust_sched_launcher_factory *launchfac,
                                    bool killed) {
    KLOG(kernel, kern, "Using %d scheduler threads.", num_threads);

    for(size_t i = 0; i < num_threads; ++i) {
        threads.push(create_task_thread(launchfac, i, killed));
    }
}

void
rust_scheduler::destroy_task_threads() {
    for(size_t i = 0; i < num_threads; ++i) {
        destroy_task_thread(threads[i]);
    }
}

void
rust_scheduler::start_task_threads()
{
    for(size_t i = 0; i < num_threads; ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->start();
    }
}

void
rust_scheduler::join_task_threads()
{
    for(size_t i = 0; i < num_threads; ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->join();
    }
}

void
rust_scheduler::kill_all_tasks() {
    for(size_t i = 0; i < num_threads; ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->get_loop()->kill_all_tasks();
    }
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name) {
    size_t thread_no;
    {
        scoped_lock with(lock);
        live_tasks++;
        thread_no = cur_thread++;
        if (cur_thread >= num_threads)
            cur_thread = 0;
    }
    kernel->register_task();
    rust_sched_launcher *thread = threads[thread_no];
    return thread->get_loop()->create_task(spawner, name);
}

void
rust_scheduler::release_task() {
    bool need_exit = false;
    {
        scoped_lock with(lock);
        live_tasks--;
        if (live_tasks == 0 && may_exit) {
            need_exit = true;
        }
    }
    kernel->unregister_task();
    if (need_exit) {
        exit();
    }
}

void
rust_scheduler::exit() {
    // Take a copy of num_threads. After the last thread exits this
    // scheduler will get destroyed, and our fields will cease to exist.
    size_t current_num_threads = num_threads;
    for(size_t i = 0; i < current_num_threads; ++i) {
        threads[i]->get_loop()->exit();
    }
}

size_t
rust_scheduler::number_of_threads() {
    return num_threads;
}

void
rust_scheduler::release_task_thread() {
    uintptr_t new_live_threads;
    {
        scoped_lock with(lock);
        new_live_threads = --live_threads;
    }
    if (new_live_threads == 0) {
        kernel->release_scheduler_id(id);
    }
}

void
rust_scheduler::allow_exit() {
    bool need_exit = false;
    {
        scoped_lock with(lock);
        may_exit = true;
        need_exit = live_tasks == 0;
    }
    if (need_exit) {
        exit();
    }
}

void
rust_scheduler::disallow_exit() {
    scoped_lock with(lock);
    may_exit = false;
}
