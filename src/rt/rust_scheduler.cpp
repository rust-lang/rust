
#include "rust_globals.h"
#include "rust_scheduler.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_sched_launcher.h"

rust_scheduler::rust_scheduler(rust_kernel *kernel,
                               size_t max_num_threads,
                               rust_sched_id id,
                               bool allow_exit,
                               bool killed,
                               rust_sched_launcher_factory *launchfac) :
    ref_count(1),
    kernel(kernel),
    live_threads(0),
    live_tasks(0),
    cur_thread(0),
    may_exit(allow_exit),
    killed(killed),
    launchfac(launchfac),
    max_num_threads(max_num_threads),
    id(id)
{
    // Create the first thread
    threads.push(create_task_thread(0));
}

void rust_scheduler::delete_this() {
    destroy_task_threads();
    delete launchfac;
    delete this;
}

rust_sched_launcher *
rust_scheduler::create_task_thread(int id) {
    live_threads++;
    rust_sched_launcher *thread = launchfac->create(this, id, killed);
    KLOG(kernel, kern, "created task thread: " PTR
         ", id: %d, live_threads: %d",
         thread, id, live_threads);
    return thread;
}

void
rust_scheduler::destroy_task_thread(rust_sched_launcher *thread) {
    KLOG(kernel, kern, "deleting task thread: " PTR, thread);
    delete thread;
}

void
rust_scheduler::destroy_task_threads() {
    for(size_t i = 0; i < threads.size(); ++i) {
        destroy_task_thread(threads[i]);
    }
}

void
rust_scheduler::start_task_threads()
{
    for(size_t i = 0; i < threads.size(); ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->start();
    }
}

void
rust_scheduler::join_task_threads()
{
    for(size_t i = 0; i < threads.size(); ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->join();
    }
}

void
rust_scheduler::kill_all_tasks() {
    for(size_t i = 0; i < threads.size(); ++i) {
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

        // Find unoccupied thread
        for (thread_no = 0; thread_no < threads.size(); ++thread_no) {
            if (threads[thread_no]->get_loop()->number_of_live_tasks() == 0)
                break;
        }

        if (thread_no == threads.size()) {
            if (threads.size() < max_num_threads) {
                // Else create new thread
                thread_no = threads.size();
                rust_sched_launcher *thread = create_task_thread(thread_no);
                thread->start();
                threads.push(thread);
            } else {
                // Or use round robin allocation
                thread_no = cur_thread++;
                if (cur_thread >= max_num_threads)
                    cur_thread = 0;
            }
        }
    }
    KLOG(kernel, kern, "Creating task %s, on thread %d.", name, thread_no);
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
    // Take a copy of the number of threads. After the last thread exits this
    // scheduler will get destroyed, and our fields will cease to exist.
    size_t current_num_threads = threads.size();
    for(size_t i = 0; i < current_num_threads; ++i) {
        threads[i]->get_loop()->exit();
    }
}

size_t
rust_scheduler::max_number_of_threads() {
    return max_num_threads;
}

size_t
rust_scheduler::number_of_threads() {
    return threads.size();
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
