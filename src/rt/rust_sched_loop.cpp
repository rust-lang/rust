// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_sched_loop.h"
#include "rust_util.h"
#include "rust_scheduler.h"

#ifndef _WIN32
pthread_key_t rust_sched_loop::task_key;
#else
DWORD rust_sched_loop::task_key;
#endif

const size_t C_STACK_SIZE = 2*1024*1024;

bool rust_sched_loop::tls_initialized = false;

rust_sched_loop::rust_sched_loop(rust_scheduler *sched, int id, bool killed) :
    _log(this),
    id(id),
    should_exit(false),
    cached_c_stack(NULL),
    extra_c_stack(NULL),
    cached_big_stack(NULL),
    extra_big_stack(NULL),
    dead_task(NULL),
    killed(killed),
    pump_signal(NULL),
    kernel(sched->kernel),
    sched(sched),
    log_lvl(log_debug),
    min_stack_size(kernel->env->min_stack_size),
    local_region(kernel->env, false),
    // FIXME #2891: calculate a per-scheduler name.
    name("main")
{
    LOGPTR(this, "new dom", (uintptr_t)this);
    rng_init(kernel, &rng, NULL, 0);

    if (!tls_initialized)
        init_tls();
}

void
rust_sched_loop::activate(rust_task *task) {
    lock.must_have_lock();
    task->ctx.next = &c_context;
    DLOG(this, task, "descheduling...");
    lock.unlock();
    prepare_c_stack(task);
    task->ctx.swap(c_context);
    task->cleanup_after_turn();
    unprepare_c_stack();
    lock.lock();
    DLOG(this, task, "task has returned");
}


void
rust_sched_loop::fail() {
    _log.log(NULL, log_err, "domain %s @0x%" PRIxPTR " root task failed",
        name, this);
    kernel->fail();
}

void
rust_sched_loop::kill_all_tasks() {
    std::vector<rust_task*> all_tasks;

    {
        scoped_lock with(lock);
        // Any task created after this will be killed. See transition, below.
        killed = true;

        for (size_t i = 0; i < running_tasks.length(); i++) {
            rust_task *t = running_tasks[i];
            t->ref();
            all_tasks.push_back(t);
        }

        for (size_t i = 0; i < blocked_tasks.length(); i++) {
            rust_task *t = blocked_tasks[i];
            t->ref();
            all_tasks.push_back(t);
        }
    }

    while (!all_tasks.empty()) {
        rust_task *task = all_tasks.back();
        all_tasks.pop_back();
        task->kill();
        task->deref();
    }
}

size_t
rust_sched_loop::number_of_live_tasks() {
    lock.must_have_lock();
    return running_tasks.length() + blocked_tasks.length();
}

/**
 * Delete any dead tasks.
 */
void
rust_sched_loop::reap_dead_tasks() {
    lock.must_have_lock();

    if (dead_task == NULL) {
        return;
    }

    // Dereferencing the task will probably cause it to be released
    // from the scheduler, which may end up trying to take this lock
    lock.unlock();

    dead_task->delete_all_stacks();
    // Deref the task, which may cause it to request us to release it
    dead_task->deref();
    dead_task = NULL;

    lock.lock();
}

void
rust_sched_loop::release_task(rust_task *task) {
    // Nobody should have a ref to the task at this point
    assert(task->get_ref_count() == 0);
    // Now delete the task, which will require using this thread's
    // memory region.
    delete task;
    // Now release the task from the scheduler, which may trigger this
    // thread to exit
    sched->release_task();
}

/**
 * Schedules a running task for execution. Only running tasks can be
 * activated.  Blocked tasks have to be unblocked before they can be
 * activated.
 *
 * Returns NULL if no tasks can be scheduled.
 */
rust_task *
rust_sched_loop::schedule_task() {
    lock.must_have_lock();
    size_t tasks = running_tasks.length();
    if (tasks > 0) {
        size_t i = (tasks > 1) ? (rng_gen_u32(kernel, &rng) % tasks) : 0;
        return running_tasks[i];
    }
    return NULL;
}

void
rust_sched_loop::log_state() {
    if (log_rt_task < log_debug) return;

    if (!running_tasks.is_empty()) {
        _log.log(NULL, log_debug, "running tasks:");
        for (size_t i = 0; i < running_tasks.length(); i++) {
            _log.log(NULL, log_debug, "\t task: %s @0x%" PRIxPTR,
                running_tasks[i]->name,
                running_tasks[i]);
        }
    }

    if (!blocked_tasks.is_empty()) {
        _log.log(NULL, log_debug, "blocked tasks:");
        for (size_t i = 0; i < blocked_tasks.length(); i++) {
            _log.log(NULL, log_debug, "\t task: %s @0x%" PRIxPTR
                ", blocked on: 0x%" PRIxPTR " '%s'",
                blocked_tasks[i]->name, blocked_tasks[i],
                blocked_tasks[i]->get_cond(),
                blocked_tasks[i]->get_cond_name());
        }
    }
}

void
rust_sched_loop::on_pump_loop(rust_signal *signal) {
    assert(pump_signal == NULL);
    assert(signal != NULL);
    pump_signal = signal;
}

void
rust_sched_loop::pump_loop() {
    assert(pump_signal != NULL);
    pump_signal->signal();
}

rust_sched_loop_state
rust_sched_loop::run_single_turn() {
    DLOG(this, task,
         "scheduler %d resuming ...", id);

    lock.lock();

    if (!should_exit) {
        assert(dead_task == NULL && "Tasks should only die after running");

        DLOG(this, dom, "worker %d, number_of_live_tasks = %d",
             id, number_of_live_tasks());

        rust_task *scheduled_task = schedule_task();

        if (scheduled_task == NULL) {
            log_state();
            DLOG(this, task,
                 "all tasks are blocked, scheduler id %d yielding ...",
                 id);

            lock.unlock();
            return sched_loop_state_block;
        }

        scheduled_task->assert_is_running();

        DLOG(this, task,
             "activating task %s 0x%" PRIxPTR
             ", state: %s",
             scheduled_task->name,
             (uintptr_t)scheduled_task,
             state_name(scheduled_task->get_state()));

        place_task_in_tls(scheduled_task);

        DLOG(this, task,
             "Running task %p on worker %d",
             scheduled_task, id);
        activate(scheduled_task);

        DLOG(this, task,
             "returned from task %s @0x%" PRIxPTR
             " in state '%s', worker id=%d" PRIxPTR,
             scheduled_task->name,
             (uintptr_t)scheduled_task,
             state_name(scheduled_task->get_state()),
             id);

        reap_dead_tasks();

        lock.unlock();
        return sched_loop_state_keep_going;
    } else {
        assert(running_tasks.is_empty() && "Should have no running tasks");
        assert(blocked_tasks.is_empty() && "Should have no blocked tasks");
        assert(dead_task == NULL && "Should have no dead tasks");

        DLOG(this, dom, "finished main-loop %d", id);

        lock.unlock();

        assert(!extra_c_stack);
        if (cached_c_stack) {
            destroy_exchange_stack(kernel->region(), cached_c_stack);
            cached_c_stack = NULL;
        }
        assert(!extra_big_stack);
        if (cached_big_stack) {
            destroy_exchange_stack(kernel->region(), cached_big_stack);
            cached_big_stack = NULL;
        }

        sched->release_task_thread();
        return sched_loop_state_exit;
    }
}

rust_task *
rust_sched_loop::create_task(rust_task *spawner, const char *name) {
    rust_task *task =
        new (this->kernel, "rust_task")
        rust_task(this, task_state_newborn,
                  name, kernel->env->min_stack_size);
    DLOG(this, task, "created task: " PTR ", spawner: %s, name: %s",
                        task, spawner ? spawner->name : "(none)", name);

    task->id = kernel->generate_task_id();
    return task;
}

rust_task_list *
rust_sched_loop::state_list(rust_task_state state) {
    switch (state) {
    case task_state_running:
        return &running_tasks;
    case task_state_blocked:
        return &blocked_tasks;
    default:
        return NULL;
    }
}

const char *
rust_sched_loop::state_name(rust_task_state state) {
    switch (state) {
    case task_state_newborn:
        return "newborn";
    case task_state_running:
        return "running";
    case task_state_blocked:
        return "blocked";
    case task_state_dead:
        return "dead";
    default:
        assert(false);
        return "";
    }
}

void
rust_sched_loop::transition(rust_task *task,
                             rust_task_state src, rust_task_state dst,
                             rust_cond *cond, const char* cond_name) {
    scoped_lock with(lock);
    DLOG(this, task,
         "task %s " PTR " state change '%s' -> '%s' while in '%s'",
         name, (uintptr_t)this, state_name(src), state_name(dst),
         state_name(task->get_state()));
    assert(task->get_state() == src);
    rust_task_list *src_list = state_list(src);
    if (src_list) {
        src_list->remove(task);
    }
    rust_task_list *dst_list = state_list(dst);
    if (dst_list) {
        dst_list->append(task);
    }
    if (dst == task_state_dead) {
        assert(dead_task == NULL);
        dead_task = task;
    }
    task->set_state(dst, cond, cond_name);

    // If the entire runtime is failing, newborn tasks must be doomed.
    if (src == task_state_newborn && killed) {
        task->kill_inner();
    }

    pump_loop();
}

#ifndef _WIN32
void
rust_sched_loop::init_tls() {
    int result = pthread_key_create(&task_key, NULL);
    assert(!result && "Couldn't create the TLS key!");
    tls_initialized = true;
}

void
rust_sched_loop::place_task_in_tls(rust_task *task) {
    int result = pthread_setspecific(task_key, task);
    assert(!result && "Couldn't place the task in TLS!");
    task->record_stack_limit();
}
#else
void
rust_sched_loop::init_tls() {
    task_key = TlsAlloc();
    assert(task_key != TLS_OUT_OF_INDEXES && "Couldn't create the TLS key!");
    tls_initialized = true;
}

void
rust_sched_loop::place_task_in_tls(rust_task *task) {
    BOOL result = TlsSetValue(task_key, task);
    assert(result && "Couldn't place the task in TLS!");
    task->record_stack_limit();
}
#endif

void
rust_sched_loop::exit() {
    scoped_lock with(lock);
    DLOG(this, dom, "Requesting exit for thread %d", id);
    should_exit = true;
    pump_loop();
}

// Before activating each task, make sure we have a C stack available.
// It needs to be allocated ahead of time (while we're on our own
// stack), because once we're on the Rust stack we won't have enough
// room to do the allocation
void
rust_sched_loop::prepare_c_stack(rust_task *task) {
    assert(!extra_c_stack);
    if (!cached_c_stack && !task->have_c_stack()) {
        cached_c_stack = create_exchange_stack(kernel->region(),
                                               C_STACK_SIZE);
    }
    assert(!extra_big_stack);
    if (!cached_big_stack) {
        cached_big_stack = create_exchange_stack(kernel->region(),
                                                 C_STACK_SIZE +
                                                 (C_STACK_SIZE * 2));
        cached_big_stack->is_big = 1;
    }
}

void
rust_sched_loop::unprepare_c_stack() {
    if (extra_c_stack) {
        destroy_exchange_stack(kernel->region(), extra_c_stack);
        extra_c_stack = NULL;
    }
    if (extra_big_stack) {
        destroy_exchange_stack(kernel->region(), extra_big_stack);
        extra_big_stack = NULL;
    }
}

//
// Local Variables:
// mode: C++
// fill-column: 70;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
