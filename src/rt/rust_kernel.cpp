#include "rust_internal.h"
#include "rust_util.h"

#define KLOG_(...)                              \
    KLOG(this, kern, __VA_ARGS__)
#define KLOG_ERR_(field, ...)                   \
    KLOG_LVL(this, field, log_err, __VA_ARGS__)

rust_kernel::rust_kernel(rust_srv *srv, size_t num_threads) :
    _region(srv, true),
    _log(srv, NULL),
    srv(srv),
    max_id(0),
    num_threads(num_threads),
    rval(0),
    live_tasks(0),
    env(srv->env)
{
    isaac_init(this, &rctx);
    create_schedulers();
}

rust_scheduler *
rust_kernel::create_scheduler(int id) {
    _kernel_lock.lock();
    rust_srv *srv = this->srv->clone();
    rust_scheduler *sched =
        new (this, "rust_scheduler") rust_scheduler(this, srv, id);
    KLOG_("created scheduler: " PTR ", id: %d, index: %d",
          sched, id, sched->list_index);
    _kernel_lock.signal_all();
    _kernel_lock.unlock();
    return sched;
}

void
rust_kernel::destroy_scheduler(rust_scheduler *sched) {
    _kernel_lock.lock();
    KLOG_("deleting scheduler: " PTR ", name: %s, index: %d",
        sched, sched->name, sched->list_index);
    rust_srv *srv = sched->srv;
    delete sched;
    delete srv;
    _kernel_lock.signal_all();
    _kernel_lock.unlock();
}

void rust_kernel::create_schedulers() {
    KLOG_("Using %d scheduler threads.", num_threads);

    for(size_t i = 0; i < num_threads; ++i) {
        threads.push(create_scheduler(i));
    }
}

void rust_kernel::destroy_schedulers() {
    for(size_t i = 0; i < num_threads; ++i) {
        destroy_scheduler(threads[i]);
    }
}

void
rust_kernel::log_all_scheduler_state() {
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i]->log_state();
    }
}

/**
 * Checks for simple deadlocks.
 */
bool
rust_kernel::is_deadlocked() {
    return false;
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
    destroy_schedulers();
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

void
rust_kernel::signal_kernel_lock() {
    _kernel_lock.lock();
    _kernel_lock.signal_all();
    _kernel_lock.unlock();
}

int rust_kernel::start_task_threads()
{
    for(size_t i = 0; i < num_threads; ++i) {
        rust_scheduler *thread = threads[i];
        thread->start();
    }

    for(size_t i = 0; i < num_threads; ++i) {
        rust_scheduler *thread = threads[i];
        thread->join();
    }

    return rval;
}

void
rust_kernel::fail() {
    // FIXME: On windows we're getting "Application has requested the
    // Runtime to terminate it in an unusual way" when trying to shutdown
    // cleanly.
#if defined(__WIN32__)
    exit(rval);
#endif
    for(size_t i = 0; i < num_threads; ++i) {
        rust_scheduler *thread = threads[i];
        thread->kill_all_tasks();
    }
}

rust_task_id
rust_kernel::create_task(rust_task *spawner, const char *name) {
    scoped_lock with(_kernel_lock);
    rust_scheduler *thread = threads[isaac_rand(&rctx) % num_threads];
    rust_task *t = thread->create_task(spawner, name);
    t->user.id = max_id++;
    task_table.put(t->user.id, t);
    return t->user.id;
}

rust_task *
rust_kernel::get_task_by_id(rust_task_id id) {
    scoped_lock with(_kernel_lock);
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

void
rust_kernel::release_task_id(rust_task_id id) {
    scoped_lock with(_kernel_lock);
    task_table.remove(id);
}

void rust_kernel::wakeup_schedulers() {
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i]->lock.signal_all();
    }
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

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
