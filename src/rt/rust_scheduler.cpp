
#include <stdarg.h>
#include "rust_internal.h"
#include "globals.h"

rust_scheduler::rust_scheduler(rust_kernel *kernel,
                               rust_srv *srv,
                               int id) :
    interrupt_flag(0),
    _log(srv, this),
    log_lvl(log_note),
    srv(srv),
    // TODO: calculate a per scheduler name.
    name("main"),
    newborn_tasks(this, "newborn"),
    running_tasks(this, "running"),
    blocked_tasks(this, "blocked"),
    dead_tasks(this, "dead"),
    cache(this),
    kernel(kernel),
    id(id),
    min_stack_size(kernel->env->min_stack_size),
    env(kernel->env)
{
    LOGPTR(this, "new dom", (uintptr_t)this);
    isaac_init(this, &rctx);
#ifndef __WIN32__
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);
    pthread_attr_setdetachstate(&attr, true);
#endif
}

rust_scheduler::~rust_scheduler() {
    DLOG(this, dom, "~rust_scheduler %s @0x%" PRIxPTR, name, (uintptr_t)this);

    newborn_tasks.delete_all();
    running_tasks.delete_all();
    blocked_tasks.delete_all();
    dead_tasks.delete_all();
#ifndef __WIN32__
    pthread_attr_destroy(&attr);
#endif
}

void
rust_scheduler::activate(rust_task *task) {
    context ctx;

    task->ctx.next = &ctx;
    DLOG(this, task, "descheduling...");
    lock.unlock();
    task->ctx.swap(ctx);
    lock.lock();
    DLOG(this, task, "task has returned");
}

void
rust_scheduler::log(rust_task* task, uint32_t level, char const *fmt, ...) {
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    _log.trace_ln(task, level, buf);
    va_end(args);
}

void
rust_scheduler::fail() {
    log(NULL, log_err, "domain %s @0x%" PRIxPTR " root task failed",
        name, this);
    I(this, kernel->rval == 0);
    kernel->rval = 1;
    kernel->fail();
}

void
rust_scheduler::kill_all_tasks() {
    I(this, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);

    for (size_t i = 0; i < running_tasks.length(); i++) {
        running_tasks[i]->kill();
    }

    for (size_t i = 0; i < blocked_tasks.length(); i++) {
        blocked_tasks[i]->kill();
    }
}

size_t
rust_scheduler::number_of_live_tasks() {
    return running_tasks.length() + blocked_tasks.length();
}

/**
 * Delete any dead tasks.
 */
void
rust_scheduler::reap_dead_tasks(int id) {
    I(this, lock.lock_held_by_current_thread());
    for (size_t i = 0; i < dead_tasks.length(); ) {
        rust_task *task = dead_tasks[i];
        task->lock.lock();
        // Make sure this task isn't still running somewhere else...
        if (task->can_schedule(id)) {
            I(this, task->tasks_waiting_to_join.is_empty());
            dead_tasks.remove(task);
            DLOG(this, task,
                "deleting unreferenced dead task %s @0x%" PRIxPTR,
                task->name, task);
            task->lock.unlock();
            task->deref();
            sync::decrement(kernel->live_tasks);
            kernel->wakeup_schedulers();
            continue;
        }
        task->lock.unlock();
        ++i;
    }
}

/**
 * Schedules a running task for execution. Only running tasks can be
 * activated.  Blocked tasks have to be unblocked before they can be
 * activated.
 *
 * Returns NULL if no tasks can be scheduled.
 */
rust_task *
rust_scheduler::schedule_task(int id) {
    I(this, this);
    // FIXME: in the face of failing tasks, this is not always right.
    // I(this, n_live_tasks() > 0);
    if (running_tasks.length() > 0) {
        size_t k = rand(&rctx);
        // Look around for a runnable task, starting at k.
        for(size_t j = 0; j < running_tasks.length(); ++j) {
            size_t  i = (j + k) % running_tasks.length();
            if (running_tasks[i]->can_schedule(id)) {
                return (rust_task *)running_tasks[i];
            }
        }
    }
    return NULL;
}

void
rust_scheduler::log_state() {
    if (log_rt_task < log_note) return;

    if (!running_tasks.is_empty()) {
        log(NULL, log_note, "running tasks:");
        for (size_t i = 0; i < running_tasks.length(); i++) {
            log(NULL, log_note, "\t task: %s @0x%" PRIxPTR
                " remaining: %" PRId64 " us",
                running_tasks[i]->name,
                running_tasks[i],
                running_tasks[i]->yield_timer.remaining_us());
        }
    }

    if (!blocked_tasks.is_empty()) {
        log(NULL, log_note, "blocked tasks:");
        for (size_t i = 0; i < blocked_tasks.length(); i++) {
            log(NULL, log_note, "\t task: %s @0x%" PRIxPTR ", blocked on: 0x%"
                PRIxPTR " '%s'",
                blocked_tasks[i]->name, blocked_tasks[i],
                blocked_tasks[i]->cond, blocked_tasks[i]->cond_name);
        }
    }

    if (!dead_tasks.is_empty()) {
        log(NULL, log_note, "dead tasks:");
        for (size_t i = 0; i < dead_tasks.length(); i++) {
            log(NULL, log_note, "\t task: %s 0x%" PRIxPTR,
                dead_tasks[i]->name, dead_tasks[i]);
        }
    }
}
/**
 * Starts the main scheduler loop which performs task scheduling for this
 * domain.
 *
 * Returns once no more tasks can be scheduled and all task ref_counts
 * drop to zero.
 */
void
rust_scheduler::start_main_loop() {
    lock.lock();

    // Make sure someone is watching, to pull us out of infinite loops.
    //
    // FIXME: time-based interruption is not presently working; worked
    // in rustboot and has been completely broken in rustc.
    //
    // rust_timer timer(this);

    DLOG(this, dom, "started domain loop %d", id);

    while (kernel->live_tasks > 0) {
        A(this, kernel->is_deadlocked() == false, "deadlock");

        DLOG(this, dom, "worker %d, number_of_live_tasks = %d, total = %d",
             id, number_of_live_tasks(), kernel->live_tasks);

        rust_task *scheduled_task = schedule_task(id);

        if (scheduled_task == NULL) {
            log_state();
            DLOG(this, task,
                 "all tasks are blocked, scheduler id %d yielding ...",
                 id);
            lock.timed_wait(10);
            reap_dead_tasks(id);
            DLOG(this, task,
                 "scheduler %d resuming ...", id);
            continue;
        }

        I(this, scheduled_task->running());

        DLOG(this, task,
             "activating task %s 0x%" PRIxPTR
             ", sp=0x%" PRIxPTR
             ", state: %s",
             scheduled_task->name,
             (uintptr_t)scheduled_task,
             scheduled_task->rust_sp,
             scheduled_task->state->name);

        interrupt_flag = 0;

        DLOG(this, task,
             "Running task %p on worker %d",
             scheduled_task, id);
        scheduled_task->running_on = id;
        activate(scheduled_task);
        scheduled_task->running_on = -1;

        DLOG(this, task,
             "returned from task %s @0x%" PRIxPTR
             " in state '%s', sp=0x%x, worker id=%d" PRIxPTR,
             scheduled_task->name,
             (uintptr_t)scheduled_task,
             scheduled_task->state->name,
             scheduled_task->rust_sp,
             id);

        reap_dead_tasks(id);
    }

    DLOG(this, dom,
         "terminated scheduler loop, reaping dead tasks ...");

    while (dead_tasks.length() > 0) {
        DLOG(this, dom,
             "waiting for %d dead tasks to become dereferenced, "
             "scheduler yielding ...",
             dead_tasks.length());
        log_state();
        lock.unlock();
        sync::yield();
        lock.lock();
        reap_dead_tasks(id);
    }

    DLOG(this, dom, "finished main-loop %d", id);

    lock.unlock();
}

rust_crate_cache *
rust_scheduler::get_cache() {
    return &cache;
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name) {
    rust_task *task =
        new (this->kernel, "rust_task")
        rust_task (this, &newborn_tasks, spawner, name);
    DLOG(this, task, "created task: " PTR ", spawner: %s, name: %s",
                        task, spawner ? spawner->name : "null", name);
    if(spawner) {
        task->pin(spawner->pinned_on);
        task->on_wakeup(spawner->_on_wakeup);
    }

    {
        scoped_lock with(lock);
        newborn_tasks.append(task);
    }

    sync::increment(kernel->live_tasks);

    return task;
}

void rust_scheduler::run() {
    this->start_main_loop();
}

//
// Local Variables:
// mode: C++
// fill-column: 70;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
