// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


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
    scoped_lock with(lock);
    threads.push(create_task_thread(0));
}

void rust_scheduler::delete_this() {
    destroy_task_threads();
    delete launchfac;
    delete this;
}

rust_sched_launcher *
rust_scheduler::create_task_thread(int id) {
    lock.must_have_lock();
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
    scoped_lock with(lock);
    for(size_t i = 0; i < threads.size(); ++i) {
        destroy_task_thread(threads[i]);
    }
}

void
rust_scheduler::start_task_threads()
{
    scoped_lock with(lock);
    for(size_t i = 0; i < threads.size(); ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->start();
    }
}

void
rust_scheduler::join_task_threads()
{
    scoped_lock with(lock);
    for(size_t i = 0; i < threads.size(); ++i) {
        rust_sched_launcher *thread = threads[i];
        thread->join();
    }
}

void
rust_scheduler::kill_all_tasks() {
    array_list<rust_sched_launcher *> copied_threads;
    {
        scoped_lock with(lock);
        killed = true;
        for (size_t i = 0; i < threads.size(); ++i) {
            copied_threads.push(threads[i]);
        }
    }
    for(size_t i = 0; i < copied_threads.size(); ++i) {
        rust_sched_launcher *thread = copied_threads[i];
        thread->get_loop()->kill_all_tasks();
    }
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name) {
    size_t thread_no;
    {
        scoped_lock with(lock);
        live_tasks++;

        if (cur_thread < threads.size()) {
            thread_no = cur_thread;
        } else {
            assert(threads.size() < max_num_threads);
            thread_no = threads.size();
            rust_sched_launcher *thread = create_task_thread(thread_no);
            thread->start();
            threads.push(thread);
        }
        cur_thread = (thread_no + 1) % max_num_threads;
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
    //
    // This is also the reason we can't use the lock here (as in the other
    // cases when accessing `threads`), after the loop the lock won't exist
    // anymore. This is safe because this method is only called when all the
    // task are dead, so there is no chance of a task trying to create new
    // threads.
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
    scoped_lock with(lock);
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
