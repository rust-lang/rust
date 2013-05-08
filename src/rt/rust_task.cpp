// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#ifndef __WIN32__
#ifdef __ANDROID__
#include "rust_android_dummy.h"
#else
#include <execinfo.h>
#endif
#endif
#include <iostream>
#include <algorithm>

#include "rust_task.h"
#include "rust_env.h"
#include "rust_globals.h"
#include "rust_crate_map.h"

// Tasks
rust_task::rust_task(rust_sched_loop *sched_loop, rust_task_state state,
                     const char *name, size_t init_stack_sz) :
    ref_count(1),
    id(0),
    stk(NULL),
    runtime_sp(0),
    sched(sched_loop->sched),
    sched_loop(sched_loop),
    kernel(sched_loop->kernel),
    name(name),
    list_index(-1),
    boxed(&local_region, sched_loop->kernel->env->poison_on_free),
    local_region(&sched_loop->local_region),
    unwinding(false),
    total_stack_sz(0),
    task_local_data(NULL),
    task_local_data_cleanup(NULL),
    borrow_list(NULL),
    state(state),
    cond(NULL),
    cond_name("none"),
    event_reject(false),
    event(NULL),
    killed(false),
    reentered_rust_stack(false),
    disallow_kill(0),
    disallow_yield(0),
    c_stack(NULL),
    next_c_sp(0),
    next_rust_sp(0),
    big_stack(NULL)
{
    LOGPTR(sched_loop, "new task", (uintptr_t)this);
    DLOG(sched_loop, task, "sizeof(task) = %d (0x%x)",
         sizeof *this, sizeof *this);

    new_stack(init_stack_sz);
}

// NB: This does not always run on the task's scheduler thread
void
rust_task::delete_this()
{
    DLOG(sched_loop, task, "~rust_task %s @0x%" PRIxPTR ", refcnt=%d",
         name, (uintptr_t)this, ref_count);

    /* FIXME (#2677): tighten this up, there are some more
       assertions that hold at task-lifecycle events. */
    assert(ref_count == 0); // ||
    //   (ref_count == 1 && this == sched->root_task));

    // The borrow list should be freed in the task annihilator
    assert(!borrow_list);

    sched_loop->release_task(this);
}

// All failure goes through me. Put your breakpoints here!
extern "C" void
rust_task_fail(rust_task *task,
               char const *expr,
               char const *file,
               size_t line) {
    assert(task != NULL);
    task->begin_failure(expr, file, line);
}

struct spawn_args {
    rust_task *task;
    spawn_fn f;
    rust_opaque_box *envptr;
    void *argptr;
};

struct cleanup_args {
    spawn_args *spargs;
    bool threw_exception;
};

void
annihilate_boxes(rust_task *task);

void
cleanup_task(cleanup_args *args) {
    spawn_args *a = args->spargs;
    bool threw_exception = args->threw_exception;
    rust_task *task = a->task;

    {
        scoped_lock with(task->lifecycle_lock);
        if (task->killed && !threw_exception) {
            LOG(task, task, "Task killed during termination");
            threw_exception = true;
        }
    }

    // Clean up TLS. This will only be set if TLS was used to begin with.
    // Because this is a crust function, it must be called from the C stack.
    if (task->task_local_data_cleanup != NULL) {
        // This assert should hold but it's not our job to ensure it (and
        // the condition might change). Handled in libcore/task.rs.
        // assert(task->task_local_data != NULL);
        task->task_local_data_cleanup(task->task_local_data);
        task->task_local_data = NULL;
    } else if (threw_exception && task->id == INIT_TASK_ID) {
        // Edge case: If main never spawns any tasks, but fails anyway, TLS
        // won't be around to take down the kernel (task.rs:kill_taskgroup,
        // rust_task_kill_all). Do it here instead.
        // (Note that children tasks can not init their TLS if they were
        // killed too early, so we need to check main's task id too.)
        task->fail_sched_loop();
        // This must not happen twice.
        static bool main_task_failed_without_spawning = false;
        assert(!main_task_failed_without_spawning);
        main_task_failed_without_spawning = true;
    }

    // Call the box annihilator.
    cratemap* map = reinterpret_cast<cratemap*>(global_crate_map);
    task->call_on_rust_stack(NULL, const_cast<void*>(map->annihilate_fn()));

    task->die();

#ifdef __WIN32__
    assert(!threw_exception && "No exception-handling yet on windows builds");
#endif
}

extern "C" CDECL void upcall_exchange_free(void *ptr);

// This runs on the Rust stack
void task_start_wrapper(spawn_args *a)
{
    rust_task *task = a->task;

    bool threw_exception = false;
    try {
        // The first argument is the return pointer; as the task fn
        // must have void return type, we can safely pass 0.
        a->f(0, a->envptr, a->argptr);
    } catch (rust_task *ex) {
        assert(ex == task && "Expected this task to be thrown for unwinding");
        threw_exception = true;

        if (task->c_stack) {
            task->return_c_stack();
        }

        // Since we call glue code below we need to make sure we
        // have the stack limit set up correctly
        task->reset_stack_limit();
    }

    // We should have returned any C stack by now
    assert(task->c_stack == NULL);

    rust_opaque_box* env = a->envptr;
    if(env) {
        // free the environment (which should be a unique closure).
        const type_desc *td = env->td;
        td->drop_glue(NULL, NULL, NULL, box_body(env));
        task->kernel->region()->free(env);
    }

    // The cleanup work needs lots of stack
    cleanup_args ca = {a, threw_exception};
    task->call_on_c_stack(&ca, (void*)cleanup_task);

    task->ctx.next->swap(task->ctx);
}

void
rust_task::start(spawn_fn spawnee_fn,
                 rust_opaque_box *envptr,
                 void *argptr)
{
    LOG(this, task, "starting task from fn 0x%" PRIxPTR
        " with env 0x%" PRIxPTR " and arg 0x%" PRIxPTR,
        spawnee_fn, envptr, argptr);

    assert(stk->data != NULL);

    char *sp = (char *)stk->end;

    sp -= sizeof(spawn_args);

    spawn_args *a = (spawn_args *)sp;

    a->task = this;
    a->envptr = envptr;
    a->argptr = argptr;
    a->f = spawnee_fn;

    ctx.call((void *)task_start_wrapper, a, sp);

    this->start();
}

void rust_task::start()
{
    transition(task_state_newborn, task_state_running, NULL, "none");
}

bool
rust_task::must_fail_from_being_killed() {
    scoped_lock with(lifecycle_lock);
    return must_fail_from_being_killed_inner();
}

bool
rust_task::must_fail_from_being_killed_inner() {
    lifecycle_lock.must_have_lock();
    return killed && !reentered_rust_stack && disallow_kill == 0;
}

void rust_task_yield_fail(rust_task *task) {
    LOG_ERR(task, task, "task %" PRIxPTR " yielded in an atomic section",
            task);
    task->fail();
}

// Only run this on the rust stack
MUST_CHECK bool rust_task::yield() {
    bool killed = false;

    if (disallow_yield > 0) {
        call_on_c_stack(this, (void *)rust_task_yield_fail);
    }

    // This check is largely superfluous; it's the one after the context swap
    // that really matters. This one allows us to assert a useful invariant.

    // NB: This takes lifecycle_lock three times, and I believe that none of
    // them are actually necessary, as per #3213. Removing the locks here may
    // cause *harmless* races with a killer... but I didn't observe any
    // substantial performance improvement from removing them, even with
    // msgsend-ring-pipes, and also it's my last day, so I'm not about to
    // remove them.  -- bblum
    if (must_fail_from_being_killed()) {
        {
            scoped_lock with(lifecycle_lock);
            assert(!(state == task_state_blocked));
        }
        killed = true;
    }

    // Return to the scheduler.
    ctx.next->swap(ctx);

    if (must_fail_from_being_killed()) {
        killed = true;
    }
    return killed;
}

void
rust_task::kill() {
    scoped_lock with(lifecycle_lock);
    kill_inner();
}

void rust_task::kill_inner() {
    lifecycle_lock.must_have_lock();

    // Multiple kills should be able to safely race, but check anyway.
    if (killed) {
        LOG(this, task, "task %s @0x%" PRIxPTR " already killed", name, this);
        return;
    }

    // Note the distinction here: kill() is when you're in an upcall
    // from task A and want to force-fail task B, you do B->kill().
    // If you want to fail yourself you do self->fail().
    LOG(this, task, "killing task %s @0x%" PRIxPTR, name, this);
    // When the task next goes to yield or resume it will fail
    killed = true;
    // Unblock the task so it can unwind.

    if (state == task_state_blocked &&
        must_fail_from_being_killed_inner()) {
        wakeup_inner(cond);
    }

    LOG(this, task, "preparing to unwind task: 0x%" PRIxPTR, this);
}

void
rust_task::fail() {
    // See note in ::kill() regarding who should call this.
    fail(NULL, NULL, 0);
}

void
rust_task::fail(char const *expr, char const *file, size_t line) {
    rust_task_fail(this, expr, file, line);
}

// Called only by rust_task_fail
void
rust_task::begin_failure(char const *expr, char const *file, size_t line) {

    if (expr) {
        LOG_ERR(this, task, "task failed at '%s', %s:%" PRIdPTR,
                expr, file, line);
    }

    DLOG(sched_loop, task, "task %s @0x%" PRIxPTR " failing", name, this);
    backtrace();
    unwinding = true;
#ifndef __WIN32__
    throw this;
#else
    die();
    // FIXME (#908): Need unwinding on windows. This will end up aborting
    fail_sched_loop();
#endif
}

void rust_task::fail_sched_loop() {
    sched_loop->fail();
}

void rust_task::assert_is_running()
{
    scoped_lock with(lifecycle_lock);
    assert(state == task_state_running);
}

// FIXME (#2851) Remove this code when rust_port goes away?
bool
rust_task::blocked_on(rust_cond *on)
{
    lifecycle_lock.must_have_lock();
    return cond == on;
}

void *
rust_task::malloc(size_t sz, const char *tag, type_desc *td)
{
    return local_region.malloc(sz, tag);
}

void *
rust_task::realloc(void *data, size_t sz)
{
    return local_region.realloc(data, sz);
}

void
rust_task::free(void *p)
{
    local_region.free(p);
}

void
rust_task::transition(rust_task_state src, rust_task_state dst,
                      rust_cond *cond, const char* cond_name) {
    scoped_lock with(lifecycle_lock);
    transition_inner(src, dst, cond, cond_name);
}

void rust_task::transition_inner(rust_task_state src, rust_task_state dst,
                                  rust_cond *cond, const char* cond_name) {
    lifecycle_lock.must_have_lock();
    sched_loop->transition(this, src, dst, cond, cond_name);
}

void
rust_task::set_state(rust_task_state state,
                     rust_cond *cond, const char* cond_name) {
    lifecycle_lock.must_have_lock();
    this->state = state;
    this->cond = cond;
    this->cond_name = cond_name;
}

bool
rust_task::block(rust_cond *on, const char* name) {
    scoped_lock with(lifecycle_lock);
    return block_inner(on, name);
}

bool
rust_task::block_inner(rust_cond *on, const char* name) {
    if (must_fail_from_being_killed_inner()) {
        // We're already going to die. Don't block. Tell the task to fail
        return false;
    }

    LOG(this, task, "Blocking on 0x%" PRIxPTR ", cond: 0x%" PRIxPTR,
                         (uintptr_t) on, (uintptr_t) cond);
    assert(cond == NULL && "Cannot block an already blocked task.");
    assert(on != NULL && "Cannot block on a NULL object.");

    transition_inner(task_state_running, task_state_blocked, on, name);

    return true;
}

void
rust_task::wakeup(rust_cond *from) {
    scoped_lock with(lifecycle_lock);
    wakeup_inner(from);
}

void
rust_task::wakeup_inner(rust_cond *from) {
    assert(cond != NULL && "Cannot wake up unblocked task.");
    LOG(this, task, "Blocked on 0x%" PRIxPTR " woken up on 0x%" PRIxPTR,
                        (uintptr_t) cond, (uintptr_t) from);
    assert(cond == from && "Cannot wake up blocked task on wrong condition.");

    transition_inner(task_state_blocked, task_state_running, NULL, "none");
}

void
rust_task::die() {
    transition(task_state_running, task_state_dead, NULL, "none");
}

void
rust_task::backtrace() {
    if (log_rt_backtrace <= log_err) return;
#ifndef __WIN32__
    void *call_stack[256];
    int nframes = ::backtrace(call_stack, 256);
    backtrace_symbols_fd(call_stack + 1, nframes - 1, 2);
#endif
}

size_t
rust_task::get_next_stack_size(size_t min, size_t current, size_t requested) {
    LOG(this, mem, "calculating new stack size for 0x%" PRIxPTR, this);
    LOG(this, mem,
        "min: %" PRIdPTR " current: %" PRIdPTR " requested: %" PRIdPTR,
        min, current, requested);

    // Allocate at least enough to accomodate the next frame, plus a little
    // slack to avoid thrashing
    size_t sz = std::max(min, requested + (requested / 2));

    // And double the stack size each allocation
    const size_t max = 1024 * 1024;
    size_t next = std::min(max, current * 2);

    sz = std::max(sz, next);

    LOG(this, mem, "next stack size: %" PRIdPTR, sz);
    assert(requested <= sz);
    return sz;
}

void
rust_task::free_stack(stk_seg *stk) {
    LOGPTR(sched_loop, "freeing stk segment", (uintptr_t)stk);
    total_stack_sz -= user_stack_size(stk);
    destroy_stack(&local_region, stk);
}

void
new_stack_slow(new_stack_args *args) {
    args->task->new_stack(args->requested_sz);
}

void
rust_task::new_stack(size_t requested_sz) {
    LOG(this, mem, "creating new stack for task %" PRIxPTR, this);
    if (stk) {
        ::check_stack_canary(stk);
    }

    // The minimum stack size, in bytes, of a Rust stack, excluding red zone
    size_t min_sz = sched_loop->min_stack_size;

    // Try to reuse an existing stack segment
    while (stk != NULL && stk->next != NULL) {
        size_t next_sz = user_stack_size(stk->next);
        if (min_sz <= next_sz && requested_sz <= next_sz) {
            LOG(this, mem, "reusing existing stack");
            stk = stk->next;
            return;
        } else {
            LOG(this, mem, "existing stack is not big enough");
            stk_seg *new_next = stk->next->next;
            free_stack(stk->next);
            stk->next = new_next;
            if (new_next) {
                new_next->prev = stk;
            }
        }
    }

    // The size of the current stack segment, excluding red zone
    size_t current_sz = 0;
    if (stk != NULL) {
        current_sz = user_stack_size(stk);
    }
    // The calculated size of the new stack, excluding red zone
    size_t rust_stk_sz = get_next_stack_size(min_sz,
                                             current_sz, requested_sz);

    size_t max_stack = kernel->env->max_stack_size;
    size_t used_stack = total_stack_sz + rust_stk_sz;

    // Don't allow stacks to grow forever. During unwinding we have to allow
    // for more stack than normal in order to allow destructors room to run,
    // arbitrarily selected as 2x the maximum stack size.
    if (!unwinding && used_stack > max_stack) {
        LOG_ERR(this, task, "task %" PRIxPTR " ran out of stack", this);
        abort();
    } else if (unwinding && used_stack > max_stack * 2) {
        LOG_ERR(this, task,
                "task %" PRIxPTR " ran out of stack during unwinding", this);
        abort();
    }

    size_t sz = rust_stk_sz + RED_ZONE_SIZE;
    stk_seg *new_stk = create_stack(&local_region, sz);
    LOGPTR(sched_loop, "new stk", (uintptr_t)new_stk);
    new_stk->task = this;
    new_stk->next = NULL;
    new_stk->prev = stk;
    if (stk) {
        stk->next = new_stk;
    }
    LOGPTR(sched_loop, "stk end", new_stk->end);

    stk = new_stk;
    total_stack_sz += user_stack_size(new_stk);
}

void
rust_task::cleanup_after_turn() {
    // Delete any spare stack segments that were left
    // behind by calls to prev_stack
    assert(stk);

    while (stk->next) {
        stk_seg *new_next = stk->next->next;

        if (stk->next->is_big) {
            assert (big_stack == stk->next);
            sched_loop->return_big_stack(big_stack);
            big_stack = NULL;
        } else {
            free_stack(stk->next);
        }

        stk->next = new_next;
    }
}

// NB: Runs on the Rust stack. Returns true if we successfully allocated the big
// stack and false otherwise.
bool
rust_task::new_big_stack() {
    // If we have a cached big stack segment, use it.
    if (big_stack) {
        // Check to see if we're already on the big stack.
        stk_seg *ss = stk;
        while (ss != NULL) {
            if (ss == big_stack)
                return false;
            ss = ss->prev;
        }

        // Unlink the big stack.
        if (big_stack->next)
            big_stack->next->prev = big_stack->prev;
        if (big_stack->prev)
            big_stack->prev->next = big_stack->next;
    } else {
        stk_seg *borrowed_big_stack = sched_loop->borrow_big_stack();
        if (!borrowed_big_stack) {
            abort();
        } else {
            big_stack = borrowed_big_stack;
        }
    }

    big_stack->task = this;
    big_stack->next = stk->next;
    if (big_stack->next)
        big_stack->next->prev = big_stack;
    big_stack->prev = stk;
    if (stk)
        stk->next = big_stack;

    stk = big_stack;

    return true;
}

static bool
sp_in_stk_seg(uintptr_t sp, stk_seg *stk) {
    // Not positive these bounds for sp are correct.  I think that the first
    // possible value for esp on a new stack is stk->end, which points to the
    // address before the first value to be pushed onto a new stack. The last
    // possible address we can push data to is stk->data.  Regardless, there's
    // so much slop at either end that we should never hit one of these
    // boundaries.
    return (uintptr_t)stk->data <= sp && sp <= stk->end;
}

/*
Called by landing pads during unwinding to figure out which stack segment we
are currently running on and record the stack limit (which was not restored
when unwinding through __morestack).
 */
void
rust_task::reset_stack_limit() {
    uintptr_t sp = get_sp();
    while (!sp_in_stk_seg(sp, stk)) {
        stk = stk->prev;
        assert(stk != NULL && "Failed to find the current stack");
    }
    record_stack_limit();
}

void
rust_task::check_stack_canary() {
    ::check_stack_canary(stk);
}

void
rust_task::delete_all_stacks() {
    assert(!on_rust_stack());
    // Delete all the stacks. There may be more than one if the task failed
    // and no landing pads stopped to clean up.
    assert(stk->next == NULL);
    while (stk != NULL) {
        stk_seg *prev = stk->prev;

        if (stk->is_big)
            sched_loop->return_big_stack(stk);
        else
            free_stack(stk);

        stk = prev;
    }

    big_stack = NULL;
}

/*
Returns true if we're currently running on the Rust stack
 */
bool
rust_task::on_rust_stack() {
    if (stk == NULL) {
        // This only happens during construction
        return false;
    }

    uintptr_t sp = get_sp();
    bool in_first_segment = sp_in_stk_seg(sp, stk);
    if (in_first_segment) {
        return true;
    } else if (stk->prev != NULL) {
        // This happens only when calling the upcall to delete
        // a stack segment
        bool in_second_segment = sp_in_stk_seg(sp, stk->prev);
        return in_second_segment;
    } else {
        return false;
    }
}

// NB: In inhibit_kill and allow_kill, helgrind would complain that we need to
// hold lifecycle_lock while accessing disallow_kill. Even though another
// killing task may access disallow_kill concurrently, this is not racy
// because the killer only cares if this task is blocking, and block() already
// uses proper locking. See https://github.com/mozilla/rust/issues/3213 .

void
rust_task::inhibit_kill() {
    // Here might be good, though not mandatory, to check if we have to die.
    disallow_kill++;
}

void
rust_task::allow_kill() {
    assert(disallow_kill > 0 && "Illegal allow_kill(): already killable!");
    disallow_kill--;
}

void rust_task::inhibit_yield() {
    disallow_yield++;
}

void rust_task::allow_yield() {
    assert(disallow_yield > 0 && "Illegal allow_yield(): already yieldable!");
    disallow_yield--;
}

MUST_CHECK bool rust_task::wait_event(void **result) {
    bool killed = false;
    scoped_lock with(lifecycle_lock);

    if(!event_reject) {
        block_inner(&event_cond, "waiting on event");
        lifecycle_lock.unlock();
        killed = yield();
        lifecycle_lock.lock();
    } else if (must_fail_from_being_killed_inner()) {
        // If the deschedule was rejected, yield won't do our killed check for
        // us. For thoroughness, do it here. FIXME (#524)
        killed = true;
    }

    event_reject = false;
    *result = event;
    return killed;
}

void
rust_task::signal_event(void *event) {
    scoped_lock with(lifecycle_lock);

    this->event = event;
    event_reject = true;
    if(task_state_blocked == state) {
        wakeup_inner(&event_cond);
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
