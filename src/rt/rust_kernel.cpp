#include "rust_internal.h"
#include "rust_util.h"
#include "rust_scheduler.h"

#define KLOG_(...)                              \
    KLOG(this, kern, __VA_ARGS__)
#define KLOG_ERR_(field, ...)                   \
    KLOG_LVL(this, field, log_err, __VA_ARGS__)

rust_kernel::rust_kernel(rust_srv *srv, size_t num_threads) :
    _region(srv, true),
    _log(srv, NULL),
    srv(srv),
    live_tasks(0),
    max_task_id(0),
    rval(0),
    env(srv->env)
{
    sched = new (this, "rust_scheduler")
        rust_scheduler(this, srv, num_threads);
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

rust_kernel::~rust_kernel() {
    delete sched;
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

int rust_kernel::start_schedulers()
{
    sched->start_task_threads();
    return rval;
}

rust_scheduler *
rust_kernel::get_default_scheduler() {
    return sched;
}

void
rust_kernel::fail() {
    // FIXME: On windows we're getting "Application has requested the
    // Runtime to terminate it in an unusual way" when trying to shutdown
    // cleanly.
    set_exit_status(PROC_FAIL_CODE);
#if defined(__WIN32__)
    exit(rval);
#endif
    sched->kill_all_tasks();
}

void
rust_kernel::register_task(rust_task *task) {
    int new_live_tasks;
    {
        scoped_lock with(task_lock);
        task->user.id = max_task_id++;
        task_table.put(task->user.id, task);
        new_live_tasks = ++live_tasks;
    }
    K(srv, task->user.id != INTPTR_MAX, "Hit the maximum task id");
    KLOG_("Registered task %" PRIdPTR, task->user.id);
    KLOG_("Total outstanding tasks: %d", new_live_tasks);
}

void
rust_kernel::release_task_id(rust_task_id id) {
    KLOG_("Releasing task %" PRIdPTR, id);
    int new_live_tasks;
    {
        scoped_lock with(task_lock);
        task_table.remove(id);
        new_live_tasks = --live_tasks;
    }
    KLOG_("Total outstanding tasks: %d", new_live_tasks);
    if (new_live_tasks == 0) {
        // There are no more tasks and there never will be.
        // Tell all the schedulers to exit.
        sched->exit();
    }
}

rust_task *
rust_kernel::get_task_by_id(rust_task_id id) {
    scoped_lock with(task_lock);
    rust_task *task = NULL;
    // get leaves task unchanged if not found.
    task_table.get(id, &task);
    if(task) {
        if(task->get_ref_count() == 0) {
            // this means the destructor is running, since the destructor
            // grabs the kernel lock to unregister the task. Pretend this
            // doesn't actually exist.
            return NULL;
        }
        else {
            task->ref();
        }
    }
    return task;
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
        I(this, ok);
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
