
#include <stdarg.h>
#include "rust_internal.h"
#include "globals.h"

rust_scheduler::rust_scheduler(rust_kernel *kernel,
    rust_message_queue *message_queue, rust_srv *srv,
    const char *name) :
    interrupt_flag(0),
    _log(srv, this),
    log_lvl(log_note),
    srv(srv),
    name(name),
    newborn_tasks(this, "newborn"),
    running_tasks(this, "running"),
    blocked_tasks(this, "blocked"),
    dead_tasks(this, "dead"),
    cache(this),
    root_task(NULL),
    curr_task(NULL),
    rval(0),
    kernel(kernel),
    message_queue(message_queue)
{
    LOGPTR(this, "new dom", (uintptr_t)this);
    isaac_init(this, &rctx);
#ifndef __WIN32__
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);
    pthread_attr_setdetachstate(&attr, true);
#endif
    root_task = create_task(NULL, name);
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
    kernel->scheduler_lock.unlock();
    task->ctx.swap(ctx);
    kernel->scheduler_lock.lock();
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
    I(this, rval == 0);
    rval = 1;
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
    I(this, kernel->scheduler_lock.lock_held_by_current_thread());
    for (size_t i = 0; i < dead_tasks.length(); ) {
        rust_task *task = dead_tasks[i];
        // Make sure this task isn't still running somewhere else...
        if (task->ref_count == 0 && task->can_schedule(id)) {
            I(this, task->tasks_waiting_to_join.is_empty());
            dead_tasks.remove(task);
            DLOG(this, task,
                "deleting unreferenced dead task %s @0x%" PRIxPTR,
                task->name, task);
            delete task;
            continue;
        }
        ++i;
    }
}

/**
 * Drains and processes incoming pending messages.
 */
void rust_scheduler::drain_incoming_message_queue(bool process) {
    rust_message *message;
    while (message_queue->dequeue(&message)) {
        DLOG(this, comm, "<== receiving \"%s\" " PTR,
            message->label, message);
        if (process) {
            message->process();
        }
        delete message;
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
            log(NULL, log_note, "\t task: %s @0x%" PRIxPTR " timeout: %d",
                running_tasks[i]->name,
                running_tasks[i],
                running_tasks[i]->yield_timer.get_timeout());
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
            log(NULL, log_note, "\t task: %s 0x%" PRIxPTR ", ref_count: %d",
                dead_tasks[i]->name, dead_tasks[i],
                dead_tasks[i]->ref_count);
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
int
rust_scheduler::start_main_loop(int id) {
    kernel->scheduler_lock.lock();

    // Make sure someone is watching, to pull us out of infinite loops.
    //
    // FIXME: time-based interruption is not presently working; worked
    // in rustboot and has been completely broken in rustc.
    //
    // rust_timer timer(this);

    DLOG(this, dom, "started domain loop %d", id);

    while (number_of_live_tasks() > 0) {
        A(this, kernel->is_deadlocked() == false, "deadlock");

        DLOG(this, dom, "worker %d, number_of_live_tasks = %d",
             id, number_of_live_tasks());

        drain_incoming_message_queue(true);

        rust_task *scheduled_task = schedule_task(id);

        // The scheduler busy waits until a task is available for scheduling.
        // Eventually we'll want a smarter way to do this, perhaps sleep
        // for a minimum amount of time.

        if (scheduled_task == NULL) {
            log_state();
            DLOG(this, task,
                 "all tasks are blocked, scheduler id %d yielding ...",
                 id);
            kernel->scheduler_lock.unlock();
            sync::sleep(100);
            kernel->scheduler_lock.lock();
            DLOG(this, task,
                "scheduler resuming ...");
            continue;
        }

        I(this, scheduled_task->running());

        DLOG(this, task,
            "activating task %s 0x%" PRIxPTR
            ", sp=0x%" PRIxPTR
            ", ref_count=%d"
            ", state: %s",
            scheduled_task->name,
            (uintptr_t)scheduled_task,
            scheduled_task->rust_sp,
            scheduled_task->ref_count,
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
        if (message_queue->is_empty()) {
            DLOG(this, dom,
                "waiting for %d dead tasks to become dereferenced, "
                "scheduler yielding ...",
                dead_tasks.length());
            log_state();
            kernel->scheduler_lock.unlock();
            sync::yield();
            kernel->scheduler_lock.lock();
        } else {
            drain_incoming_message_queue(true);
        }
        reap_dead_tasks(id);
    }

    DLOG(this, dom, "finished main-loop %d (dom.rval = %d)", id, rval);

    kernel->scheduler_lock.unlock();
    return rval;
}

rust_crate_cache *
rust_scheduler::get_cache() {
    return &cache;
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name) {
    rust_task *task =
        new (this->kernel) rust_task (this, &newborn_tasks, spawner, name);
    DLOG(this, task, "created task: " PTR ", spawner: %s, name: %s",
                        task, spawner ? spawner->name : "null", name);
    if(spawner) {
        task->pin(spawner->pinned_on);
        task->on_wakeup(spawner->_on_wakeup);
    }
    newborn_tasks.append(task);
    return task;
}

//
// Local Variables:
// mode: C++
// fill-column: 70;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
