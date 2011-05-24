
#include <stdarg.h>
#include "rust_internal.h"

rust_dom::rust_dom(rust_kernel *kernel,
    rust_message_queue *message_queue, rust_srv *srv,
    rust_crate const *root_crate, const char *name) :
    interrupt_flag(0),
    root_crate(root_crate),
    _log(srv, this),
    log_lvl(log_note),
    srv(srv),
    local_region(&srv->local_region),
    synchronized_region(&srv->synchronized_region),
    name(name),
    newborn_tasks(this, "newborn"),
    running_tasks(this, "running"),
    blocked_tasks(this, "blocked"),
    dead_tasks(this, "dead"),
    caches(this),
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

rust_dom::~rust_dom() {
    DLOG(this, dom, "~rust_dom %s @0x%" PRIxPTR, name, (uintptr_t)this);
    newborn_tasks.delete_all();
    running_tasks.delete_all();
    blocked_tasks.delete_all();
    dead_tasks.delete_all();
#ifndef __WIN32__
    pthread_attr_destroy(&attr);
#endif
    while (caches.length()) {
        delete caches.pop();
    }
}

extern "C" void new_rust_activate_glue(rust_task *)
    asm("new_rust_activate_glue");

void
rust_dom::activate(rust_task *task) {
    curr_task = task;
    new_rust_activate_glue(task);
    curr_task = NULL;
}

void
rust_dom::log(rust_task* task, uint32_t level, char const *fmt, ...) {
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    _log.trace_ln(task, level, buf);
    va_end(args);
}

void
rust_dom::fail() {
    log(NULL, log_err, "domain %s @0x%" PRIxPTR " root task failed",
        name, this);
    I(this, rval == 0);
    rval = 1;
}

void *
rust_dom::malloc(size_t size) {
    return malloc(size, memory_region::LOCAL);
}

void *
rust_dom::malloc(size_t size, memory_region::memory_region_type type) {
    if (type == memory_region::LOCAL) {
        return local_region.malloc(size);
    } else if (type == memory_region::SYNCHRONIZED) {
        return synchronized_region.malloc(size);
    }
    return NULL;
}

void *
rust_dom::calloc(size_t size) {
    return calloc(size, memory_region::LOCAL);
}

void *
rust_dom::calloc(size_t size, memory_region::memory_region_type type) {
    if (type == memory_region::LOCAL) {
        return local_region.calloc(size);
    } else if (type == memory_region::SYNCHRONIZED) {
        return synchronized_region.calloc(size);
    }
    return NULL;
}

void *
rust_dom::realloc(void *mem, size_t size) {
    return realloc(mem, size, memory_region::LOCAL);
}

void *
rust_dom::realloc(void *mem, size_t size,
    memory_region::memory_region_type type) {
    if (type == memory_region::LOCAL) {
        return local_region.realloc(mem, size);
    } else if (type == memory_region::SYNCHRONIZED) {
        return synchronized_region.realloc(mem, size);
    }
    return NULL;
}

void
rust_dom::free(void *mem) {
    free(mem, memory_region::LOCAL);
}

void
rust_dom::free(void *mem, memory_region::memory_region_type type) {
    DLOG(this, mem, "rust_dom::free(0x%" PRIxPTR ")", mem);
    if (type == memory_region::LOCAL) {
        local_region.free(mem);
    } else if (type == memory_region::SYNCHRONIZED) {
        synchronized_region.free(mem);
    }
    return;
}

#ifdef __WIN32__
void
rust_dom::win32_require(LPCTSTR fn, BOOL ok) {
    if (!ok) {
        LPTSTR buf;
        DWORD err = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                      FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf, 0, NULL );
        DLOG_ERR(this, dom, "%s failed with error %ld: %s", fn, err, buf);
        LocalFree((HLOCAL)buf);
        I(this, ok);
    }
}
#endif

size_t
rust_dom::number_of_live_tasks() {
    return running_tasks.length() + blocked_tasks.length();
}

/**
 * Delete any dead tasks.
 */
void
rust_dom::reap_dead_tasks() {
    for (size_t i = 0; i < dead_tasks.length(); ) {
        rust_task *task = dead_tasks[i];
        if (task->ref_count == 0) {
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
void rust_dom::drain_incoming_message_queue(bool process) {
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
rust_dom::schedule_task() {
    I(this, this);
    // FIXME: in the face of failing tasks, this is not always right.
    // I(this, n_live_tasks() > 0);
    if (running_tasks.length() > 0) {
        size_t i = rand(&rctx);
        i %= running_tasks.length();
        if (running_tasks[i]->yield_timer.has_timed_out()) {
            return (rust_task *)running_tasks[i];
        }
    }
    return NULL;
}

void
rust_dom::log_state() {
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
rust_dom::start_main_loop() {
    // Make sure someone is watching, to pull us out of infinite loops.
    rust_timer timer(this);

    DLOG(this, dom, "started domain loop");

    while (number_of_live_tasks() > 0) {
        A(this, kernel->is_deadlocked() == false, "deadlock");

        drain_incoming_message_queue(true);

        rust_task *scheduled_task = schedule_task();

        // The scheduler busy waits until a task is available for scheduling.
        // Eventually we'll want a smarter way to do this, perhaps sleep
        // for a minimum amount of time.

        if (scheduled_task == NULL) {
            log_state();
            DLOG(this, task,
                "all tasks are blocked, scheduler yielding ...");
            sync::sleep(100);
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

        activate(scheduled_task);

        DLOG(this, task,
                 "returned from task %s @0x%" PRIxPTR
                 " in state '%s', sp=0x%" PRIxPTR,
                 scheduled_task->name,
                 (uintptr_t)scheduled_task,
                 scheduled_task->state->name,
                 scheduled_task->rust_sp);

        I(this, scheduled_task->rust_sp >=
          (uintptr_t) &scheduled_task->stk->data[0]);
        I(this, scheduled_task->rust_sp < scheduled_task->stk->limit);

        reap_dead_tasks();
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
            sync::yield();
        } else {
            drain_incoming_message_queue(true);
        }
        reap_dead_tasks();
    }

    DLOG(this, dom, "finished main-loop (dom.rval = %d)", rval);
    return rval;
}


rust_crate_cache *
rust_dom::get_cache(rust_crate const *crate) {
    DLOG(this, cache, "looking for crate-cache for crate 0x%" PRIxPTR, crate);
    rust_crate_cache *cache = NULL;
    for (size_t i = 0; i < caches.length(); ++i) {
        rust_crate_cache *c = caches[i];
        if (c->crate == crate) {
            cache = c;
            break;
        }
    }
    if (!cache) {
        DLOG(this, cache,
            "making new crate-cache for crate 0x%" PRIxPTR, crate);
        cache = new (this) rust_crate_cache(this, crate);
        caches.push(cache);
    }
    cache->ref();
    return cache;
}

rust_task *
rust_dom::create_task(rust_task *spawner, const char *name) {
    rust_task *task =
        new (this) rust_task (this, &newborn_tasks, spawner, name);
    DLOG(this, task, "created task: " PTR ", spawner: %s, name: %s",
                        task, spawner ? spawner->name : "null", name);
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
