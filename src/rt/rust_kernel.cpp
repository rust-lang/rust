// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



#include "rust_kernel.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "rust_sched_launcher.h"
#include <algorithm>

#define KLOG_(...)                              \
    KLOG(this, kern, __VA_ARGS__)
#define KLOG_ERR_(field, ...)                   \
    KLOG_LVL(this, field, log_err, __VA_ARGS__)

rust_kernel::rust_kernel(rust_env *env) :
    _log(NULL),
    max_task_id(INIT_TASK_ID-1), // sync_add_and_fetch increments first
    rval(0),
    max_sched_id(1),
    killed(false),
    already_exiting(false),
    sched_reaper(this),
    osmain_driver(NULL),
    non_weak_tasks(0),
    at_exit_runner(NULL),
    at_exit_started(false),
    env(env),
    global_data(0)
{
    // Create the single threaded scheduler that will run on the platform's
    // main thread
    rust_manual_sched_launcher_factory *osmain_launchfac =
        new rust_manual_sched_launcher_factory();
    osmain_scheduler = create_scheduler(osmain_launchfac, 1, false);
    osmain_driver = osmain_launchfac->get_driver();

    // Create the primary scheduler
    rust_thread_sched_launcher_factory *main_launchfac =
        new rust_thread_sched_launcher_factory();
    main_scheduler = create_scheduler(main_launchfac,
                                      env->num_sched_threads,
                                      false);

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
    return exchange_alloc.malloc(size);
}

void *
rust_kernel::realloc(void *mem, size_t size) {
    return exchange_alloc.realloc(mem, size);
}

void rust_kernel::free(void *mem) {
    exchange_alloc.free(mem);
}

rust_sched_id
rust_kernel::create_scheduler(size_t num_threads) {
    rust_thread_sched_launcher_factory *launchfac =
        new rust_thread_sched_launcher_factory();
    return create_scheduler(launchfac, num_threads, true);
}

rust_sched_id
rust_kernel::create_scheduler(rust_sched_launcher_factory *launchfac,
                              size_t num_threads, bool allow_exit) {
    rust_sched_id id;
    rust_scheduler *sched;
    {
        scoped_lock with(sched_lock);

        /*if (sched_table.size() == 2) {
            // The main and OS main schedulers may not exit while there are
            // other schedulers
            KLOG_("Disallowing main scheduler to exit");
            rust_scheduler *main_sched =
                get_scheduler_by_id_nolock(main_scheduler);
            assert(main_sched != NULL);
            main_sched->disallow_exit();
        }
        if (sched_table.size() == 1) {
            KLOG_("Disallowing osmain scheduler to exit");
            rust_scheduler *osmain_sched =
                get_scheduler_by_id_nolock(osmain_scheduler);
            assert(osmain_sched != NULL);
            osmain_sched->disallow_exit();
            }*/

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
Called by rust_sched_reaper to join every terminating scheduler thread,
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
            /*if (sched_table.size() == 2) {
                KLOG_("Allowing main scheduler to exit");
                // It's only the main schedulers left. Tell them to exit
                rust_scheduler *main_sched =
                    get_scheduler_by_id_nolock(main_scheduler);
                assert(main_sched != NULL);
                main_sched->allow_exit();
            }
            if (sched_table.size() == 1) {
                KLOG_("Allowing osmain scheduler to exit");
                rust_scheduler *osmain_sched =
                    get_scheduler_by_id_nolock(osmain_scheduler);
                assert(osmain_sched != NULL);
                osmain_sched->allow_exit();
            }*/
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
    // FIXME (#908): On windows we're getting "Application has
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

void
rust_kernel::set_exit_status(int code) {
    scoped_lock with(rval_lock);
    // If we've already failed then that's the code we're going to use
    if (rval != PROC_FAIL_CODE) {
        rval = code;
    }
}

void
rust_kernel::inc_live_count() {
    uintptr_t new_non_weak_tasks = sync::increment(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
}

void
rust_kernel::dec_live_count() {
    uintptr_t new_non_weak_tasks = sync::decrement(non_weak_tasks);
    KLOG_("New non-weak tasks %" PRIdPTR, new_non_weak_tasks);
    if (new_non_weak_tasks == 0) {
        begin_shutdown();
    }
}

void
rust_kernel::allow_scheduler_exit() {
    scoped_lock with(sched_lock);

    KLOG_("Allowing main scheduler to exit");
    // It's only the main schedulers left. Tell them to exit
    rust_scheduler *main_sched =
        get_scheduler_by_id_nolock(main_scheduler);
    assert(main_sched != NULL);
    main_sched->allow_exit();

    KLOG_("Allowing osmain scheduler to exit");
    rust_scheduler *osmain_sched =
        get_scheduler_by_id_nolock(osmain_scheduler);
    assert(osmain_sched != NULL);
    osmain_sched->allow_exit();
}

void
rust_kernel::begin_shutdown() {
    {
        scoped_lock with(sched_lock);
        // FIXME #4410: This shouldn't be necessary, but because of
        // unweaken_task this may end up getting called multiple times.
        if (already_exiting) {
            return;
        } else {
            already_exiting = true;
        }
    }

    run_exit_functions();
    allow_scheduler_exit();
}

void
rust_kernel::register_exit_function(spawn_fn runner, fn_env_pair *f) {
    scoped_lock with(at_exit_lock);

    assert(!at_exit_started && "registering at_exit function after exit");

    if (at_exit_runner) {
        // FIXME #2912 Would be very nice to assert this but we can't because
        // of the way coretest works (the test case ends up using its own
        // function)
        //assert(runner == at_exit_runner
        //       && "there can be only one at_exit_runner");
    }

    at_exit_runner = runner;
    at_exit_fns.push_back(f);
}

void
rust_kernel::run_exit_functions() {
    rust_task *task;

    {
        scoped_lock with(at_exit_lock);

        assert(!at_exit_started && "running exit functions twice?");

        at_exit_started = true;

        if (at_exit_runner == NULL) {
            return;
        }

        rust_scheduler *sched = get_scheduler_by_id(main_sched_id());
        assert(sched);
        task = sched->create_task(NULL, "at_exit");

        final_exit_fns.count = at_exit_fns.size();
        final_exit_fns.start = at_exit_fns.data();
    }

    task->start(at_exit_runner, NULL, &final_exit_fns);
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
