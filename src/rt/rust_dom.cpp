
#include <stdarg.h>
#include "rust_internal.h"

template class ptr_vec<rust_task>;

// Keeps track of all live domains, for debugging purposes.
array_list<rust_dom*> _live_domains;

rust_dom::rust_dom(rust_srv *srv, rust_crate const *root_crate,
                   const char *name) :
    interrupt_flag(0),
    root_crate(root_crate),
    _log(srv, this),
    srv(srv),
    local_region(&srv->local_region),
    synchronized_region(&srv->synchronized_region),
    name(name),
    running_tasks(this),
    blocked_tasks(this),
    dead_tasks(this),
    caches(this),
    root_task(NULL),
    curr_task(NULL),
    rval(0)
{
    logptr("new dom", (uintptr_t)this);
    isaac_init(this, &rctx);
#ifndef __WIN32__
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);
    pthread_attr_setdetachstate(&attr, true);
#endif
    root_task = new (this) rust_task(this, NULL, name);

    if (_live_domains.replace(NULL, this) == false) {
        _live_domains.append(this);
    }
}

static void
del_all_tasks(rust_dom *dom, ptr_vec<rust_task> *v) {
    I(dom, v);
    while (v->length()) {
        dom->log(rust_log::TASK, "deleting task 0x%" PRIdPTR,
                 v->length() - 1);
        delete v->pop();
    }
}

void
rust_dom::delete_proxies() {
    rust_task *task;
    rust_proxy<rust_task> *task_proxy;
    while (_task_proxies.pop(&task, &task_proxy)) {
        log(rust_log::TASK,
            "deleting proxy 0x%" PRIxPTR " in dom %s 0x%" PRIxPTR,
            task_proxy, task_proxy->dom->name, task_proxy->dom);
        delete task_proxy;
    }

    rust_port *port;
    rust_proxy<rust_port> *port_proxy;
    while (_port_proxies.pop(&port, &port_proxy)) {
        log(rust_log::TASK,
            "deleting proxy 0x%" PRIxPTR " in dom %s 0x%" PRIxPTR,
            port_proxy, port_proxy->dom->name, port_proxy->dom);
        delete port_proxy;
    }
}

rust_dom::~rust_dom() {
    log(rust_log::MEM | rust_log::DOM,
        "~rust_dom %s @0x%" PRIxPTR, name, (uintptr_t)this);

    log(rust_log::TASK, "deleting all proxies");
    delete_proxies();
    log(rust_log::TASK, "deleting all running tasks");
    del_all_tasks(this, &running_tasks);
    log(rust_log::TASK, "deleting all blocked tasks");
    del_all_tasks(this, &blocked_tasks);
    log(rust_log::TASK, "deleting all dead tasks");
    del_all_tasks(this, &dead_tasks);
#ifndef __WIN32__
    pthread_attr_destroy(&attr);
#endif
    while (caches.length())
        delete caches.pop();

    _live_domains.replace(this, NULL);
}

void
rust_dom::activate(rust_task *task) {
    curr_task = task;
    root_crate->get_activate_glue()(task);
    curr_task = NULL;
}

void
rust_dom::log(rust_task *task, uint32_t type_bits, char const *fmt, ...) {
    char buf[256];
    if (_log.is_tracing(type_bits)) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        _log.trace_ln(task, type_bits, buf);
        va_end(args);
    }
}

void
rust_dom::log(uint32_t type_bits, char const *fmt, ...) {
    char buf[256];
    if (_log.is_tracing(type_bits)) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        _log.trace_ln(NULL, type_bits, buf);
        va_end(args);
    }
}

rust_log &
rust_dom::get_log() {
    return _log;
}

void
rust_dom::logptr(char const *msg, uintptr_t ptrval) {
    log(rust_log::MEM, "%s 0x%" PRIxPTR, msg, ptrval);
}

template<typename T> void
rust_dom::logptr(char const *msg, T* ptrval) {
    log(rust_log::MEM, "%s 0x%" PRIxPTR, msg, (uintptr_t)ptrval);
}


void
rust_dom::fail() {
    log(rust_log::DOM, "domain %s @0x%" PRIxPTR " root task failed",
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
    log(rust_log::MEM, "rust_dom::free(0x%" PRIxPTR ")", mem);
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
        log(rust_log::ERR, "%s failed with error %ld: %s", fn, err, buf);
        LocalFree((HLOCAL)buf);
        I(this, ok);
    }
}
#endif

size_t
rust_dom::n_live_tasks()
{
    return running_tasks.length() + blocked_tasks.length();
}

void
rust_dom::add_task_to_state_vec(ptr_vec<rust_task> *v, rust_task *task)
{
    log(rust_log::MEM|rust_log::TASK,
        "adding task %s @0x%" PRIxPTR " in state '%s' to vec 0x%" PRIxPTR,
        task->name, (uintptr_t)task, state_vec_name(v), (uintptr_t)v);
    v->push(task);
}


void
rust_dom::remove_task_from_state_vec(ptr_vec<rust_task> *v, rust_task *task)
{
    log(rust_log::MEM|rust_log::TASK,
        "removing task %s @0x%" PRIxPTR " in state '%s' from vec 0x%" PRIxPTR,
        task->name, (uintptr_t)task, state_vec_name(v), (uintptr_t)v);
    I(this, (*v)[task->idx] == task);
    v->swap_delete(task);
}

const char *
rust_dom::state_vec_name(ptr_vec<rust_task> *v)
{
    if (v == &running_tasks)
        return "running";
    if (v == &blocked_tasks)
        return "blocked";
    I(this, v == &dead_tasks);
    return "dead";
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
            dead_tasks.swap_delete(task);
            log(rust_log::TASK,
                "deleting unreferenced dead task %s @0x%" PRIxPTR,
                task->name, task);
            delete task;
            continue;
        }
        ++i;
    }
}

/**
 * Enqueues a message in this domain's incoming message queue. It's the
 * responsibility of the receiver to free the message once it's processed.
 */
void rust_dom::send_message(rust_message *message) {
    log(rust_log::COMM, "==> enqueueing \"%s\" 0x%" PRIxPTR
                        " in queue 0x%" PRIxPTR
                        " in domain 0x%" PRIxPTR,
                        message->label,
                        message,
                        &_incoming_message_queue,
                        this);
    _incoming_message_queue.enqueue(message);
}

/**
 * Drains and processes incoming pending messages.
 */
void rust_dom::drain_incoming_message_queue() {
    rust_message *message;
    while ((message = (rust_message *) _incoming_message_queue.dequeue())) {
        log(rust_log::COMM, "<== processing incoming message \"%s\" 0x%"
            PRIxPTR, message->label, message);
        message->process();
        message->~rust_message();
        this->synchronized_region.free(message);
    }
}

rust_proxy<rust_task> *
rust_dom::get_task_proxy(rust_task *task) {
    rust_proxy<rust_task> *proxy = NULL;
    if (_task_proxies.get(task, &proxy)) {
        return proxy;
    }
    log(rust_log::COMM, "no proxy for %s @0x%" PRIxPTR, task->name, task);
    proxy = new (this) rust_proxy<rust_task> (this, task, false);
    _task_proxies.put(task, proxy);
    return proxy;
}

/**
 * Gets a proxy for this port.
 *
 * TODO: This method needs to be synchronized since it's usually called
 * during upcall_clone_chan in a different thread. However, for now
 * since this usually happens before the thread actually starts,
 * we may get lucky without synchronizing.
 *
 */
rust_proxy<rust_port> *
rust_dom::get_port_proxy_synchronized(rust_port *port) {
    rust_proxy<rust_port> *proxy = NULL;
    if (_port_proxies.get(port, &proxy)) {
        return proxy;
    }
    log(rust_log::COMM, "no proxy for 0x%" PRIxPTR, port);
    proxy = new (this) rust_proxy<rust_port> (this, port, false);
    _port_proxies.put(port, proxy);
    return proxy;
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
    // log(rust_log::DOM|rust_log::TASK, "no schedulable tasks");
    return NULL;
}

/**
 * Checks for simple deadlocks.
 */
bool
rust_dom::is_deadlocked() {
    if (_live_domains.size() != 1) {
        // We cannot tell if we are deadlocked if other domains exists.
        return false;
    }

    if (running_tasks.length() != 0) {
        // We are making progress and therefore we are not deadlocked.
        return false;
    }

    if (_incoming_message_queue.is_empty() && blocked_tasks.length() > 0) {
        // We have no messages to process, no running tasks to schedule
        // and some blocked tasks therefore we are likely in a deadlock.
        log_state();
        return true;
    }

    return false;
}

void
rust_dom::log_all_state() {
    for (uint32_t i = 0; i < _live_domains.size(); i++) {
        _live_domains[i]->log_state();
    }
}

void
rust_dom::log_state() {
    if (!running_tasks.is_empty()) {
        log(rust_log::TASK, "running tasks:");
        for (size_t i = 0; i < running_tasks.length(); i++) {
            log(rust_log::TASK,
                "\t task: %s @0x%" PRIxPTR
                " timeout: %d",
                running_tasks[i]->name,
                running_tasks[i],
                running_tasks[i]->yield_timer.get_timeout());
        }
    }

    if (!blocked_tasks.is_empty()) {
        log(rust_log::TASK, "blocked tasks:");
        for (size_t i = 0; i < blocked_tasks.length(); i++) {
            log(rust_log::TASK,
                "\t task: %s @0x%" PRIxPTR ", blocked on: 0x%" PRIxPTR
                " '%s'",
                blocked_tasks[i]->name, blocked_tasks[i],
                blocked_tasks[i]->cond, blocked_tasks[i]->cond_name);
        }
    }

    if (!dead_tasks.is_empty()) {
        log(rust_log::TASK, "dead tasks:");
        for (size_t i = 0; i < dead_tasks.length(); i++) {
            log(rust_log::TASK, "\t task: %s 0x%" PRIxPTR ", ref_count: %d",
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
rust_dom::start_main_loop()
{
    // Make sure someone is watching, to pull us out of infinite loops.
    rust_timer timer(this);

    log(rust_log::DOM, "running main-loop on domain %s @0x%" PRIxPTR,
        name, this);
    logptr("exit-task glue", root_crate->get_exit_task_glue());

    while (n_live_tasks() > 0) {
        A(this, is_deadlocked() == false, "deadlock");

        drain_incoming_message_queue();

        rust_task *scheduled_task = schedule_task();

        // The scheduler busy waits until a task is available for scheduling.
        // Eventually we'll want a smarter way to do this, perhaps sleep
        // for a minimum amount of time.

        if (scheduled_task == NULL) {
            if (_log.is_tracing(rust_log::TASK)) {
                log_state();
            }
            log(rust_log::TASK,
                "all tasks are blocked, scheduler yielding ...");
            sync::yield();
            log(rust_log::TASK,
                "scheduler resuming ...");
            continue;
        }

        I(this, scheduled_task->running());

        log(rust_log::TASK,
            "activating task %s 0x%" PRIxPTR
            ", sp=0x%" PRIxPTR
            ", ref_count=%d"
            ", state: %s",
            scheduled_task->name,
            (uintptr_t)scheduled_task,
            scheduled_task->rust_sp,
            scheduled_task->ref_count,
            scheduled_task->state_str());

        interrupt_flag = 0;

        activate(scheduled_task);

        log(rust_log::TASK,
                 "returned from task %s @0x%" PRIxPTR
                 " in state '%s', sp=0x%" PRIxPTR,
                 scheduled_task->name,
                 (uintptr_t)scheduled_task,
                 state_vec_name(scheduled_task->state),
                 scheduled_task->rust_sp);

        I(this, scheduled_task->rust_sp >=
          (uintptr_t) &scheduled_task->stk->data[0]);
        I(this, scheduled_task->rust_sp < scheduled_task->stk->limit);

        reap_dead_tasks();
    }

    log(rust_log::DOM, "terminated scheduler loop, reaping dead tasks ...");

    while (dead_tasks.length() > 0) {
        if (_incoming_message_queue.is_empty()) {
            log(rust_log::DOM,
                "waiting for %d dead tasks to become dereferenced, "
                "scheduler yielding ...",
                dead_tasks.length());
            if (_log.is_tracing(rust_log::TASK)) {
                log_state();
            }
            sync::yield();
        } else {
            drain_incoming_message_queue();
        }
        reap_dead_tasks();
    }

    log(rust_log::DOM, "finished main-loop (dom.rval = %d)", rval);
    return rval;
}


rust_crate_cache *
rust_dom::get_cache(rust_crate const *crate) {
    log(rust_log::CACHE,
        "looking for crate-cache for crate 0x%" PRIxPTR, crate);
    rust_crate_cache *cache = NULL;
    for (size_t i = 0; i < caches.length(); ++i) {
        rust_crate_cache *c = caches[i];
        if (c->crate == crate) {
            cache = c;
            break;
        }
    }
    if (!cache) {
        log(rust_log::CACHE,
            "making new crate-cache for crate 0x%" PRIxPTR, crate);
        cache = new (this) rust_crate_cache(this, crate);
        caches.push(cache);
    }
    cache->ref();
    return cache;
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
