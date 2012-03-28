
#include "rust_internal.h"
#include "rust_cc.h"

#ifndef __WIN32__
#include <execinfo.h>
#endif
#include <iostream>
#include <cassert>
#include <cstring>
#include <algorithm>

#include "rust_upcall.h"

// Tasks
rust_task::rust_task(rust_task_thread *thread, rust_task_state state,
                     rust_task *spawner, const char *name,
                     size_t init_stack_sz) :
    ref_count(1),
    id(0),
    notify_enabled(false),
    stk(NULL),
    runtime_sp(0),
    sched(thread->sched),
    thread(thread),
    kernel(thread->kernel),
    name(name),
    list_index(-1),
    rendezvous_ptr(0),
    local_region(&thread->srv->local_region),
    boxed(&local_region),
    unwinding(false),
    propagate_failure(true),
    cc_counter(0),
    total_stack_sz(0),
    state(state),
    cond(NULL),
    cond_name("none"),
    killed(false),
    reentered_rust_stack(false),
    c_stack(NULL),
    next_c_sp(0),
    next_rust_sp(0),
    supervisor(spawner)
{
    LOGPTR(thread, "new task", (uintptr_t)this);
    DLOG(thread, task, "sizeof(task) = %d (0x%x)", sizeof *this, sizeof *this);

    new_stack(init_stack_sz);
    if (supervisor) {
        supervisor->ref();
    }
}

// NB: This does not always run on the task's scheduler thread
void
rust_task::delete_this()
{
    DLOG(thread, task, "~rust_task %s @0x%" PRIxPTR ", refcnt=%d",
         name, (uintptr_t)this, ref_count);

    // FIXME: We should do this when the task exits, not in the destructor
    {
        scoped_lock with(supervisor_lock);
        if (supervisor) {
            supervisor->deref();
        }
    }

    /* FIXME: tighten this up, there are some more
       assertions that hold at task-lifecycle events. */
    I(thread, ref_count == 0); // ||
    //   (ref_count == 1 && this == sched->root_task));

    thread->release_task(this);
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
cleanup_task(cleanup_args *args) {
    spawn_args *a = args->spargs;
    bool threw_exception = args->threw_exception;
    rust_task *task = a->task;

    cc::do_final_cc(task);

    task->die();

    {
        scoped_lock with(task->kill_lock);
        if (task->killed && !threw_exception) {
            LOG(task, task, "Task killed during termination");
            threw_exception = true;
        }
    }

    task->notify(!threw_exception);

    if (threw_exception) {
#ifndef __WIN32__
        task->conclude_failure();
#else
        A(task->thread, false, "Shouldn't happen");
#endif
    }
}

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
        A(task->thread, ex == task,
          "Expected this task to be thrown for unwinding");
        threw_exception = true;

        if (task->c_stack) {
            task->return_c_stack();
        }

        // Since we call glue code below we need to make sure we
        // have the stack limit set up correctly
        task->reset_stack_limit();
    }

    // We should have returned any C stack by now
    I(task->thread, task->c_stack == NULL);

    rust_opaque_box* env = a->envptr;
    if(env) {
        // free the environment (which should be a unique closure).
        const type_desc *td = env->td;
        td->drop_glue(NULL, NULL, td->first_param, box_body(env));
        upcall_shared_free(env);
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

    I(thread, stk->data != NULL);

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
    scoped_lock with(kill_lock);
    return must_fail_from_being_killed_unlocked();
}

bool
rust_task::must_fail_from_being_killed_unlocked() {
    return killed && !reentered_rust_stack;
}

// Only run this on the rust stack
void
rust_task::yield(bool *killed) {
    if (must_fail_from_being_killed()) {
        I(thread, !blocked());
        *killed = true;
    }

    // Return to the scheduler.
    ctx.next->swap(ctx);

    if (must_fail_from_being_killed()) {
        *killed = true;
    }
}

void
rust_task::kill() {
    scoped_lock with(kill_lock);

    if (dead()) {
        // Task is already dead, can't kill what's already dead.
        fail_parent();
    }

    // Note the distinction here: kill() is when you're in an upcall
    // from task A and want to force-fail task B, you do B->kill().
    // If you want to fail yourself you do self->fail().
    LOG(this, task, "killing task %s @0x%" PRIxPTR, name, this);
    // When the task next goes to yield or resume it will fail
    killed = true;
    // Unblock the task so it can unwind.

    if (blocked()) {
        wakeup(cond);
    }

    LOG(this, task, "preparing to unwind task: 0x%" PRIxPTR, this);
}

extern "C" CDECL
bool rust_task_is_unwinding(rust_task *rt) {
    return rt->unwinding;
}

void
rust_task::fail() {
    // See note in ::kill() regarding who should call this.
    DLOG(thread, task, "task %s @0x%" PRIxPTR " failing", name, this);
    backtrace();
    unwinding = true;
#ifndef __WIN32__
    throw this;
#else
    die();
    conclude_failure();
    // FIXME: Need unwinding on windows. This will end up aborting
    thread->fail();
#endif
}

void
rust_task::conclude_failure() {
    fail_parent();
}

void
rust_task::fail_parent() {
    scoped_lock with(supervisor_lock);
    if (supervisor) {
        DLOG(thread, task,
             "task %s @0x%" PRIxPTR
             " propagating failure to supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
        supervisor->kill();
    }
    if (NULL == supervisor && propagate_failure)
        thread->fail();
}

void
rust_task::unsupervise()
{
    scoped_lock with(supervisor_lock);
    if (supervisor) {
        DLOG(thread, task,
             "task %s @0x%" PRIxPTR
             " disconnecting from supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
        supervisor->deref();
    }
    supervisor = NULL;
    propagate_failure = false;
}

frame_glue_fns*
rust_task::get_frame_glue_fns(uintptr_t fp) {
    fp -= sizeof(uintptr_t);
    return *((frame_glue_fns**) fp);
}

bool
rust_task::running()
{
    scoped_lock with(state_lock);
    return state == task_state_running;
}

bool
rust_task::blocked()
{
    scoped_lock with(state_lock);
    return state == task_state_blocked;
}

bool
rust_task::blocked_on(rust_cond *on)
{
    scoped_lock with(state_lock);
    return cond == on;
}

bool
rust_task::dead()
{
    scoped_lock with(state_lock);
    return state == task_state_dead;
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
    thread->transition(this, src, dst, cond, cond_name);
}

void
rust_task::set_state(rust_task_state state,
                     rust_cond *cond, const char* cond_name) {
    scoped_lock with(state_lock);
    this->state = state;
    this->cond = cond;
    this->cond_name = cond_name;
}

bool
rust_task::block(rust_cond *on, const char* name) {
    scoped_lock with(kill_lock);

    if (must_fail_from_being_killed_unlocked()) {
        // We're already going to die. Don't block. Tell the task to fail
        return false;
    }

    LOG(this, task, "Blocking on 0x%" PRIxPTR ", cond: 0x%" PRIxPTR,
                         (uintptr_t) on, (uintptr_t) cond);
    A(thread, cond == NULL, "Cannot block an already blocked task.");
    A(thread, on != NULL, "Cannot block on a NULL object.");

    transition(task_state_running, task_state_blocked, on, name);

    return true;
}

void
rust_task::wakeup(rust_cond *from) {
    A(thread, cond != NULL, "Cannot wake up unblocked task.");
    LOG(this, task, "Blocked on 0x%" PRIxPTR " woken up on 0x%" PRIxPTR,
                        (uintptr_t) cond, (uintptr_t) from);
    A(thread, cond == from, "Cannot wake up blocked task on wrong condition.");

    transition(task_state_blocked, task_state_running, NULL, "none");
}

void
rust_task::die() {
    transition(task_state_running, task_state_dead, NULL, "none");
}

void
rust_task::backtrace() {
    if (!log_rt_backtrace) return;
#ifndef __WIN32__
    void *call_stack[256];
    int nframes = ::backtrace(call_stack, 256);
    backtrace_symbols_fd(call_stack + 1, nframes - 1, 2);
#endif
}

void *
rust_task::calloc(size_t size, const char *tag) {
    return local_region.calloc(size, tag);
}

void
rust_task::notify(bool success) {
    // FIXME (1078) Do this in rust code
    if(notify_enabled) {
        rust_port *target_port =
            kernel->get_port_by_id(notify_port);
        if(target_port) {
            task_notification msg;
            msg.id = id;
            msg.result = !success ? tr_failure : tr_success;

            target_port->send(&msg);
            target_port->deref();
        }
    }
}

size_t
rust_task::get_next_stack_size(size_t min, size_t current, size_t requested) {
    LOG(this, mem, "calculating new stack size for 0x%" PRIxPTR, this);
    LOG(this, mem,
        "min: %" PRIdPTR " current: %" PRIdPTR " requested: %" PRIdPTR,
        min, current, requested);

    // Allocate at least enough to accomodate the next frame
    size_t sz = std::max(min, requested);

    // And double the stack size each allocation
    const size_t max = 1024 * 1024;
    size_t next = std::min(max, current * 2);

    sz = std::max(sz, next);

    LOG(this, mem, "next stack size: %" PRIdPTR, sz);
    I(thread, requested <= sz);
    return sz;
}

void
rust_task::free_stack(stk_seg *stk) {
    LOGPTR(thread, "freeing stk segment", (uintptr_t)stk);
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
    size_t min_sz = thread->min_stack_size;

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

    if (total_stack_sz + rust_stk_sz > thread->env->max_stack_size) {
        LOG_ERR(this, task, "task %" PRIxPTR " ran out of stack", this);
        fail();
    }

    size_t sz = rust_stk_sz + RED_ZONE_SIZE;
    stk_seg *new_stk = create_stack(&local_region, sz);
    LOGPTR(thread, "new stk", (uintptr_t)new_stk);
    new_stk->task = this;
    new_stk->next = NULL;
    new_stk->prev = stk;
    if (stk) {
        stk->next = new_stk;
    }
    LOGPTR(thread, "stk end", new_stk->end);

    stk = new_stk;
    total_stack_sz += user_stack_size(new_stk);
}

void
rust_task::cleanup_after_turn() {
    // Delete any spare stack segments that were left
    // behind by calls to prev_stack
    I(thread, stk);
    while (stk->next) {
        stk_seg *new_next = stk->next->next;
        free_stack(stk->next);
        stk->next = new_next;
    }
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

struct reset_args {
    rust_task *task;
    uintptr_t sp;
};

void
reset_stack_limit_on_c_stack(reset_args *args) {
    rust_task *task = args->task;
    uintptr_t sp = args->sp;
    while (!sp_in_stk_seg(sp, task->stk)) {
        task->stk = task->stk->prev;
        A(task->thread, task->stk != NULL,
          "Failed to find the current stack");
    }
    task->record_stack_limit();
}

/*
Called by landing pads during unwinding to figure out which stack segment we
are currently running on and record the stack limit (which was not restored
when unwinding through __morestack).
 */
void
rust_task::reset_stack_limit() {
    I(thread, on_rust_stack());
    uintptr_t sp = get_sp();
    // Have to do the rest on the C stack because it involves
    // freeing stack segments, logging, etc.
    // FIXME: This probably doesn't need to happen on the C
    // stack now
    reset_args ra = {this, sp};
    call_on_c_stack(&ra, (void*)reset_stack_limit_on_c_stack);
}

void
rust_task::check_stack_canary() {
    ::check_stack_canary(stk);
}

void
rust_task::delete_all_stacks() {
    I(thread, !on_rust_stack());
    // Delete all the stacks. There may be more than one if the task failed
    // and no landing pads stopped to clean up.
    I(thread, stk->next == NULL);
    while (stk != NULL) {
        stk_seg *prev = stk->prev;
        free_stack(stk);
        stk = prev;
    }
}

void
rust_task::config_notify(rust_port_id port) {
    notify_enabled = true;
    notify_port = port;
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

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
