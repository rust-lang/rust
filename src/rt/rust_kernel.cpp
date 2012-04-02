#include "rust_internal.h"
#include "rust_util.h"
#include "rust_scheduler.h"

#include <vector>

#define KLOG_(...)                              \
    KLOG(this, kern, __VA_ARGS__)
#define KLOG_ERR_(field, ...)                   \
    KLOG_LVL(this, field, log_err, __VA_ARGS__)

rust_kernel::rust_kernel(rust_env *env) :
    _region(env, true),
    _log(NULL),
    max_task_id(0),
    max_port_id(0),
    rval(0),
    max_sched_id(0),
    sched_reaper(this),
    env(env)
{
}

void
rust_kernel::log(uint32_t level, char const *fmt, ...) {
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    _log.trace_ln(NULL, level, buf);
    va_end(args);
}

void
rust_kernel::fatal(char const *fmt, ...) {
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    _log.trace_ln(NULL, (uint32_t)0, buf);
    exit(1);
    va_end(args);
}

void *
rust_kernel::malloc(size_t size, const char *tag) {
    return _region.malloc(size, tag);
}

void *
rust_kernel::realloc(void *mem, size_t size) {
    return _region.realloc(mem, size);
}

void rust_kernel::free(void *mem) {
    _region.free(mem);
}

rust_sched_id
rust_kernel::create_scheduler(size_t num_threads) {
    rust_sched_id id;
    rust_scheduler *sched;
    {
        scoped_lock with(sched_lock);
        // If this is the first scheduler then we need to launch
        // the scheduler reaper.
        bool start_reaper = sched_table.empty();
        id = max_sched_id++;
        assert(id != INTPTR_MAX && "Hit the maximum scheduler id");
        sched = new (this, "rust_scheduler")
            rust_scheduler(this, num_threads, id);
        bool is_new = sched_table
            .insert(std::pair<rust_sched_id,
                              rust_scheduler*>(id, sched)).second;
        assert(is_new && "Reusing a sched id?");
        if (start_reaper) {
            sched_reaper.start();
        }
    }
    sched->start_task_threads();
    return id;
}

rust_scheduler *
rust_kernel::get_scheduler_by_id(rust_sched_id id) {
    scoped_lock with(sched_lock);
    sched_map::iterator iter = sched_table.find(id);
    if (iter != sched_table.end()) {
        return iter->second;
    } else {
        return NULL;
    }
}

void
rust_kernel::release_scheduler_id(rust_sched_id id) {
    scoped_lock with(sched_lock);
    // This list will most likely only ever have a single element in it, but
    // it's an actual list because we could potentially get here multiple
    // times before the main thread ever calls wait_for_schedulers()
    join_list.push_back(id);
    sched_lock.signal();
}

/*
Called by rust_sched_reaper to join every every terminating scheduler thread,
so that we can be sure they have completely exited before the process exits.
If we don't join them then we can see valgrind errors due to un-freed pthread
memory.
 */
void
rust_kernel::wait_for_schedulers()
{
    scoped_lock with(sched_lock);
    while (!sched_table.empty()) {
        while (!join_list.empty()) {
            rust_sched_id id = join_list.back();
            join_list.pop_back();
            sched_map::iterator iter = sched_table.find(id);
            assert(iter != sched_table.end());
            rust_scheduler *sched = iter->second;
            sched_table.erase(iter);
            sched->join_task_threads();
            delete sched;
        }
        if (!sched_table.empty()) {
            sched_lock.wait();
        }
    }
}

/* Called on the main thread to wait for the kernel to exit */
int
rust_kernel::wait_for_exit() {
    sched_reaper.join();
    return rval;
}

// FIXME: Fix all these FIXMEs
void
rust_kernel::fail() {
    // FIXME: On windows we're getting "Application has requested the
    // Runtime to terminate it in an unusual way" when trying to shutdown
    // cleanly.
    set_exit_status(PROC_FAIL_CODE);
#if defined(__WIN32__)
    exit(rval);
#endif
    // Copy the list of schedulers so that we don't hold the lock while
    // running kill_all_tasks.
    // FIXME: There's a lot that happens under kill_all_tasks, and I don't
    // know that holding sched_lock here is ok, but we need to hold the
    // sched lock to prevent the scheduler from being destroyed while
    // we are using it. Probably we need to make rust_scheduler atomicly
    // reference counted.
    std::vector<rust_scheduler*> scheds;
    {
        scoped_lock with(sched_lock);
        for (sched_map::iterator iter = sched_table.begin();
             iter != sched_table.end(); iter++) {
            scheds.push_back(iter->second);
        }
    }

    // FIXME: This is not a foolproof way to kill all tasks while ensuring
    // that no new tasks or schedulers are created in the meantime that
    // keep the scheduler alive.
    for (std::vector<rust_scheduler*>::iterator iter = scheds.begin();
         iter != scheds.end(); iter++) {
        (*iter)->kill_all_tasks();
    }
}

rust_task_id
rust_kernel::generate_task_id() {
    rust_task_id id = sync::increment(max_task_id);
    assert(id != INTPTR_MAX && "Hit the maximum task id");
    return id;
}

rust_port_id
rust_kernel::register_port(rust_port *port) {
    uintptr_t new_live_ports;
    rust_port_id new_port_id;
    {
        scoped_lock with(port_lock);
        new_port_id = max_port_id++;
        port_table.put(new_port_id, port);
        new_live_ports = port_table.count();
    }
    assert(new_port_id != INTPTR_MAX && "Hit the maximum port id");
    KLOG_("Registered port %" PRIdPTR, new_port_id);
    KLOG_("Total outstanding ports: %d", new_live_ports);
    return new_port_id;
}

void
rust_kernel::release_port_id(rust_port_id id) {
    KLOG_("Releasing port %" PRIdPTR, id);
    uintptr_t new_live_ports;
    {
        scoped_lock with(port_lock);
        port_table.remove(id);
        new_live_ports = port_table.count();
    }
    KLOG_("Total outstanding ports: %d", new_live_ports);
}

rust_port *
rust_kernel::get_port_by_id(rust_port_id id) {
    scoped_lock with(port_lock);
    rust_port *port = NULL;
    // get leaves port unchanged if not found.
    port_table.get(id, &port);
    if(port) {
        port->ref();
    }
    return port;
}

#ifdef __WIN32__
void
rust_kernel::win32_require(LPCTSTR fn, BOOL ok) {
    if (!ok) {
        LPTSTR buf;
        DWORD err = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                      FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf, 0, NULL );
        KLOG_ERR_(dom, "%s failed with error %ld: %s", fn, err, buf);
        LocalFree((HLOCAL)buf);
        assert(ok);
    }
}
#endif

void
rust_kernel::set_exit_status(int code) {
    scoped_lock with(rval_lock);
    // If we've already failed then that's the code we're going to use
    if (rval != PROC_FAIL_CODE) {
        rval = code;
    }
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
