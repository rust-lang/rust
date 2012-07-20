

#include "rust_kernel.h"
#include "rust_port.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "rust_sched_launcher.h"
#include <algorithm>

#define KLOG_(...)                              \
    KLOG(this, kern, __VA_ARGS__)
#define KLOG_ERR_(field, ...)                   \
    KLOG_LVL(this, field, log_err, __VA_ARGS__)

rust_kernel::rust_kernel(rust_env *env) :
    _region(env, true),
    _log(NULL),
    max_task_id(INIT_TASK_ID-1), // sync_add_and_fetch increments first
    max_port_id(1),
    rval(0),
    max_sched_id(1),
    killed(false),
    sched_reaper(this),
    osmain_driver(NULL),
    non_weak_tasks(0),
    global_loop_chan(0),
    global_env_chan(0),
    env(env)

{

    // Create the single threaded scheduler that will run on the platform's
    // main thread
    rust_manual_sched_launcher_factory launchfac;
    osmain_scheduler = create_scheduler(&launchfac, 1, false);
    osmain_driver = launchfac.get_driver();
    sched_reaper.start();
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
rust_kernel::calloc(size_t size, const char *tag) {
    return _region.calloc(size, tag);
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
    rust_thread_sched_launcher_factory launchfac;
    return create_scheduler(&launchfac, num_threads, true);
}

rust_sched_id
rust_kernel::create_scheduler(rust_sched_launcher_factory *launchfac,
                              size_t num_threads, bool allow_exit) {
    rust_sched_id id;
    rust_scheduler *sched;
    {
        scoped_lock with(sched_lock);

        if (sched_table.size() == 1) {
            // The OS main scheduler may not exit while there are other
            // schedulers
            KLOG_("Disallowing osmain scheduler to exit");
            rust_scheduler *sched =
                get_scheduler_by_id_nolock(osmain_scheduler);
            assert(sched != NULL);
            sched->disallow_exit();
        }

        id = max_sched_id++;
        assert(id != INTPTR_MAX && "Hit the maximum scheduler id");
        sched = new (this, "rust_scheduler")
            rust_scheduler(this, num_threads, id, allow_exit, killed,
                           launchfac);
        bool is_new = sched_table
            .insert(std::pair<rust_sched_id,
                              rust_scheduler*>(id, sched)).second;
        assert(is_new && "Reusing a sched id?");
    }
    sched->start_task_threads();
    return id;
}

rust_scheduler *
rust_kernel::get_scheduler_by_id(rust_sched_id id) {
    scoped_lock with(sched_lock);
    return get_scheduler_by_id_nolock(id);
}

rust_scheduler *
rust_kernel::get_scheduler_by_id_nolock(rust_sched_id id) {
    if (id == 0) {
        return NULL;
    }
    sched_lock.must_have_lock();
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
            KLOG_("Deleting scheduler %d", id);
            join_list.pop_back();
            sched_map::iterator iter = sched_table.find(id);
            assert(iter != sched_table.end());
            rust_scheduler *sched = iter->second;
            sched_table.erase(iter);
            sched->join_task_threads();
            sched->deref();
            if (sched_table.size() == 1) {
                KLOG_("Allowing osmain scheduler to exit");
                // It's only the osmain scheduler left. Tell it to exit
                rust_scheduler *sched =
                    get_scheduler_by_id_nolock(osmain_scheduler);
                assert(sched != NULL);
                sched->allow_exit();
            }
        }
        if (!sched_table.empty()) {
            sched_lock.wait();
        }
    }
}

/* Called on the main thread to run the osmain scheduler to completion,
   then wait for schedulers to exit */
int
rust_kernel::run() {
    assert(osmain_driver != NULL);
    osmain_driver->start_main_loop();
    sched_reaper.join();
    return rval;
}

void
rust_kernel::fail() {
    // FIXME (#2671): On windows we're getting "Application has
    // requested the Runtime to terminate it in an unusual way" when
    // trying to shutdown cleanly.
    set_exit_status(PROC_FAIL_CODE);
#if defined(__WIN32__)
    exit(rval);
#endif
    // I think this only needs to be done by one task ever; as it is,
    // multiple tasks invoking kill_all might get here. Currently libcore
    // ensures only one task will ever invoke it, but this would really be
    // fine either way, so I'm leaving it as it is. -- bblum

    // Copy the list of schedulers so that we don't hold the lock while
    // running kill_all_tasks. Refcount to ensure they stay alive.
    std::vector<rust_scheduler*> scheds;
    {
        scoped_lock with(sched_lock);
        // All schedulers created after this flag is set will be doomed.
        killed = true;
        for (sched_map::iterator iter = sched_table.begin();
             iter != sched_table.end(); iter++) {
            iter->second->ref();
            scheds.push_back(iter->second);
        }
    }

    for (std::vector<rust_scheduler*>::iterator iter = scheds.begin();
         iter != scheds.end(); iter++) {
        (*iter)->kill_all_tasks();
        (*iter)->deref();
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
    assert(id != 0 && "invalid port id");
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

void
rust_kernel::register_task() {
    KLOG_("Registering task");
    uintptr_t new_non_weak_tasks = sync::increment(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
}

void
rust_kernel::unregister_task() {
    KLOG_("Unregistering task");
    uintptr_t new_non_weak_tasks = sync::decrement(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
    if (new_non_weak_tasks == 0) {
        end_weak_tasks();
    }
}

void
rust_kernel::weaken_task(rust_port_id chan) {
    {
        scoped_lock with(weak_task_lock);
        KLOG_("Weakening task with channel %" PRIdPTR, chan);
        weak_task_chans.push_back(chan);
    }
    uintptr_t new_non_weak_tasks = sync::decrement(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
    if (new_non_weak_tasks == 0) {
        end_weak_tasks();
    }
}

void
rust_kernel::unweaken_task(rust_port_id chan) {
    uintptr_t new_non_weak_tasks = sync::increment(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
    {
        scoped_lock with(weak_task_lock);
        KLOG_("Unweakening task with channel %" PRIdPTR, chan);
        std::vector<rust_port_id>::iterator iter =
            std::find(weak_task_chans.begin(), weak_task_chans.end(), chan);
        if (iter != weak_task_chans.end()) {
            weak_task_chans.erase(iter);
        }
    }
}

void
rust_kernel::end_weak_tasks() {
    std::vector<rust_port_id> chancopies;
    {
        scoped_lock with(weak_task_lock);
        chancopies = weak_task_chans;
        weak_task_chans.clear();
    }
    while (!chancopies.empty()) {
        rust_port_id chan = chancopies.back();
        chancopies.pop_back();
        KLOG_("Notifying weak task " PRIdPTR, chan);
        uintptr_t token = 0;
        send_to_port(chan, &token);
    }
}

bool
rust_kernel::send_to_port(rust_port_id chan, void *sptr) {
    KLOG_("rust_port_id*_send port: 0x%" PRIxPTR, (uintptr_t) chan);

    rust_port *port = get_port_by_id(chan);
    if(port) {
        port->send(sptr);
        port->deref();
        return true;
    } else {
        KLOG_("didn't get the port");
        return false;
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
