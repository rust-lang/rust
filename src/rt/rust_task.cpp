
#include "rust_internal.h"
#include "rust_cc.h"

#include "valgrind.h"
#include "memcheck.h"

#ifndef __WIN32__
#include <execinfo.h>
#endif
#include <cassert>
#include <cstring>

#include "globals.h"

// Stack size
size_t g_custom_min_stack_size = 0;

static size_t
get_min_stk_size(size_t default_size) {
    if (g_custom_min_stack_size != 0) {
        return g_custom_min_stack_size;
    } else {
        return default_size;
    }
}


// Task stack segments. Heap allocated and chained together.

static stk_seg*
new_stk(rust_scheduler *sched, rust_task *task, size_t minsz)
{
    size_t min_stk_bytes = get_min_stk_size(sched->min_stack_size);
    if (minsz < min_stk_bytes)
        minsz = min_stk_bytes;
    size_t sz = sizeof(stk_seg) + minsz;
    stk_seg *stk = (stk_seg *)task->malloc(sz, "stack");
    LOGPTR(task->sched, "new stk", (uintptr_t)stk);
    memset(stk, 0, sizeof(stk_seg));
    stk->next = task->stk;
    stk->limit = (uintptr_t) &stk->data[minsz];
    LOGPTR(task->sched, "stk limit", stk->limit);
    stk->valgrind_id =
        VALGRIND_STACK_REGISTER(&stk->data[0],
                                &stk->data[minsz]);
    task->stk = stk;
    return stk;
}

static void
del_stk(rust_task *task, stk_seg *stk)
{
    assert(stk == task->stk && "Freeing stack segments out of order!");

    task->stk = stk->next;

    VALGRIND_STACK_DEREGISTER(stk->valgrind_id);
    LOGPTR(task->sched, "freeing stk segment", (uintptr_t)stk);
    task->free(stk);
}

// Entry points for `__morestack` (see arch/*/morestack.S).
extern "C" void *
rust_new_stack(size_t stk_sz, void *args_addr, size_t args_sz) {
    rust_task *task = rust_scheduler::get_task();
    stk_seg *stk_seg = new_stk(task->sched, task, stk_sz);
    memcpy(stk_seg->data, args_addr, args_sz);
    return stk_seg->data;
}

extern "C" void *
rust_del_stack() {
    rust_task *task = rust_scheduler::get_task();
    stk_seg *next_seg = task->stk->next;
    del_stk(task, task->stk);
    return next_seg->data;
}


// Tasks
rust_task::rust_task(rust_scheduler *sched, rust_task_list *state,
                     rust_task *spawner, const char *name) :
    ref_count(1),
    stk(NULL),
    runtime_sp(0),
    sched(sched),
    cache(NULL),
    kernel(sched->kernel),
    name(name),
    state(state),
    cond(NULL),
    cond_name("none"),
    supervisor(spawner),
    list_index(-1),
    next_port_id(0),
    rendezvous_ptr(0),
    running_on(-1),
    pinned_on(-1),
    local_region(&sched->srv->local_region),
    _on_wakeup(NULL),
    failed(false),
    killed(false),
    propagate_failure(true),
    dynastack(this),
    cc_counter(0)
{
    LOGPTR(sched, "new task", (uintptr_t)this);
    DLOG(sched, task, "sizeof(task) = %d (0x%x)", sizeof *this, sizeof *this);

    assert((void*)this == (void*)&user);

    user.notify_enabled = 0;

    stk = new_stk(sched, this, 0);
    user.rust_sp = stk->limit;
    if (supervisor) {
        supervisor->ref();
    }
}

rust_task::~rust_task()
{
    I(sched, !sched->lock.lock_held_by_current_thread());
    I(sched, port_table.is_empty());
    DLOG(sched, task, "~rust_task %s @0x%" PRIxPTR ", refcnt=%d",
         name, (uintptr_t)this, ref_count);

    if (supervisor) {
        supervisor->deref();
    }

    kernel->release_task_id(user.id);

    /* FIXME: tighten this up, there are some more
       assertions that hold at task-lifecycle events. */
    I(sched, ref_count == 0); // ||
    //   (ref_count == 1 && this == sched->root_task));

    del_stk(this, stk);
}

struct spawn_args {
    rust_task *task;
    uintptr_t a3;
    uintptr_t a4;
    void (*CDECL f)(int *, uintptr_t, uintptr_t);
};

struct rust_closure_env {
    intptr_t ref_count;
    type_desc *td;
};

extern "C" CDECL
void task_start_wrapper(spawn_args *a)
{
    rust_task *task = a->task;
    int rval = 42;

    bool failed = false;
    try {
        a->f(&rval, a->a3, a->a4);
    } catch (rust_task *ex) {
        A(task->sched, ex == task,
          "Expected this task to be thrown for unwinding");
        failed = true;
    }

#   ifndef __x86_64__ // FIXME: temp. hack on X86-64 (NDM)
    cc::do_cc(task);
#   endif

    rust_closure_env* env = (rust_closure_env*)a->a3;
    if(env) {
        // free the environment.
        I(task->sched, 1 == env->ref_count); // the ref count better be 1
        //env->td->drop_glue(NULL, task, NULL, env->td->first_param, env);
        //env->td->free_glue(NULL, task, NULL, env->td->first_param, env);
        task->free(env);
    }

    task->die();

    if (task->killed && !failed) {
        LOG(task, task, "Task killed during termination");
        failed = true;
    }

    task->notify(!failed);

    if (failed) {
#ifndef __WIN32__
        task->conclude_failure();
#else
        A(task->sched, false, "Shouldn't happen");
#endif
    } else {
        task->lock.lock();
        task->lock.unlock();
        task->yield(1);
    }
}

void
rust_task::start(uintptr_t spawnee_fn,
                 uintptr_t args,
                 uintptr_t env)
{
    LOG(this, task, "starting task from fn 0x%" PRIxPTR
        " with args 0x%" PRIxPTR, spawnee_fn, args);

    I(sched, stk->data != NULL);

    char *sp = (char *)user.rust_sp;

    sp -= sizeof(spawn_args);

    spawn_args *a = (spawn_args *)sp;

    a->task = this;
    a->a3 = env;
    a->a4 = args;
    void **f = (void **)&a->f;
    *f = (void *)spawnee_fn;

    ctx.call((void *)task_start_wrapper, a, sp);

    this->start();
}

void
rust_task::start(uintptr_t spawnee_fn,
                 uintptr_t args)
{
    start(spawnee_fn, args, 0);
}

void rust_task::start()
{
    yield_timer.reset_us(0);
    transition(&sched->newborn_tasks, &sched->running_tasks);
    sched->lock.signal();
}

void
rust_task::grow(size_t n_frame_bytes)
{
    // FIXME (issue #151): Just fail rather than almost certainly crashing
    // mysteriously later. The commented-out logic below won't work at all in
    // the presence of non-word-aligned pointers.
    abort();

}

void
rust_task::yield() {
    yield(0);
}

void
rust_task::yield(size_t time_in_us) {
    LOG(this, task, "task %s @0x%" PRIxPTR " yielding for %d us",
        name, this, time_in_us);

    if (killed && !dead()) {
        if (blocked()) {
            unblock();
        }
        killed = false;
        fail();
    }
    yield_timer.reset_us(time_in_us);

    // Return to the scheduler.
    ctx.next->swap(ctx);

    if (killed) {
        killed = false;
        fail();
    }
}

void
rust_task::kill() {
    if (dead()) {
        // Task is already dead, can't kill what's already dead.
        fail_parent();
        return;
    }

    // Note the distinction here: kill() is when you're in an upcall
    // from task A and want to force-fail task B, you do B->kill().
    // If you want to fail yourself you do self->fail().
    LOG(this, task, "killing task %s @0x%" PRIxPTR, name, this);
    // When the task next goes to yield or resume it will fail
    killed = true;
    // Unblock the task so it can unwind.
    unblock();

    sched->lock.signal();

    LOG(this, task, "preparing to unwind task: 0x%" PRIxPTR, this);
    // run_on_resume(rust_unwind_glue);
}

void
rust_task::fail() {
    // See note in ::kill() regarding who should call this.
    DLOG(sched, task, "task %s @0x%" PRIxPTR " failing", name, this);
    backtrace();
#ifndef __WIN32__
    throw this;
#else
    die();
    conclude_failure();
#endif
}

void
rust_task::conclude_failure() {
    // Unblock the task so it can unwind.
    unblock();
    fail_parent();
    failed = true;
    yield(4);
}

void
rust_task::fail_parent() {
    if (supervisor) {
        DLOG(sched, task,
             "task %s @0x%" PRIxPTR
             " propagating failure to supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
        supervisor->kill();
    }
    // FIXME: implement unwinding again.
    if (NULL == supervisor && propagate_failure)
        sched->fail();
}

void
rust_task::unsupervise()
{
    DLOG(sched, task,
             "task %s @0x%" PRIxPTR
             " disconnecting from supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
    if (supervisor) {
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
    return state == &sched->running_tasks;
}

bool
rust_task::blocked()
{
    return state == &sched->blocked_tasks;
}

bool
rust_task::blocked_on(rust_cond *on)
{
    return blocked() && cond == on;
}

bool
rust_task::dead()
{
    return state == &sched->dead_tasks;
}

void *
rust_task::malloc(size_t sz, const char *tag, type_desc *td)
{
    return local_region.malloc(sz, tag);
}

void *
rust_task::realloc(void *data, size_t sz, bool is_gc)
{
    return local_region.realloc(data, sz);
}

void
rust_task::free(void *p, bool is_gc)
{
    local_region.free(p);
}

void
rust_task::transition(rust_task_list *src, rust_task_list *dst) {
    bool unlock = false;
    if(!sched->lock.lock_held_by_current_thread()) {
        unlock = true;
        sched->lock.lock();
    }
    DLOG(sched, task,
         "task %s " PTR " state change '%s' -> '%s' while in '%s'",
         name, (uintptr_t)this, src->name, dst->name, state->name);
    I(sched, state == src);
    src->remove(this);
    dst->append(this);
    state = dst;
    if(unlock)
        sched->lock.unlock();
}

void
rust_task::block(rust_cond *on, const char* name) {
    I(sched, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);
    LOG(this, task, "Blocking on 0x%" PRIxPTR ", cond: 0x%" PRIxPTR,
                         (uintptr_t) on, (uintptr_t) cond);
    A(sched, cond == NULL, "Cannot block an already blocked task.");
    A(sched, on != NULL, "Cannot block on a NULL object.");

    transition(&sched->running_tasks, &sched->blocked_tasks);
    cond = on;
    cond_name = name;
}

void
rust_task::wakeup(rust_cond *from) {
    I(sched, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);
    A(sched, cond != NULL, "Cannot wake up unblocked task.");
    LOG(this, task, "Blocked on 0x%" PRIxPTR " woken up on 0x%" PRIxPTR,
                        (uintptr_t) cond, (uintptr_t) from);
    A(sched, cond == from, "Cannot wake up blocked task on wrong condition.");

    transition(&sched->blocked_tasks, &sched->running_tasks);
    I(sched, cond == from);
    cond = NULL;
    cond_name = "none";

    if(_on_wakeup) {
        _on_wakeup->on_wakeup();
    }

    sched->lock.signal();
}

void
rust_task::die() {
    I(sched, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);
    transition(&sched->running_tasks, &sched->dead_tasks);
    sched->lock.signal();
}

void
rust_task::unblock() {
    if (blocked())
        wakeup(cond);
}

rust_crate_cache *
rust_task::get_crate_cache()
{
    if (!cache) {
        DLOG(sched, task, "fetching cache for current crate");
        cache = sched->get_cache();
    }
    return cache;
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

bool rust_task::can_schedule(int id)
{
    return yield_timer.has_timed_out() &&
        running_on == -1 &&
        (pinned_on == -1 || pinned_on == id);
}

void *
rust_task::calloc(size_t size, const char *tag) {
    return local_region.calloc(size, tag);
}

void rust_task::pin() {
    I(this->sched, running_on != -1);
    pinned_on = running_on;
}

void rust_task::pin(int id) {
    I(this->sched, running_on == -1);
    pinned_on = id;
}

void rust_task::unpin() {
    pinned_on = -1;
}

void rust_task::on_wakeup(rust_task::wakeup_callback *callback) {
    _on_wakeup = callback;
}

rust_port_id rust_task::register_port(rust_port *port) {
    I(sched, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);

    rust_port_id id = next_port_id++;
    port_table.put(id, port);
    return id;
}

void rust_task::release_port(rust_port_id id) {
    I(sched, lock.lock_held_by_current_thread());
    port_table.remove(id);
}

rust_port *rust_task::get_port_by_id(rust_port_id id) {
    I(sched, !lock.lock_held_by_current_thread());
    scoped_lock with(lock);
    rust_port *port = NULL;
    port_table.get(id, &port);
    if (port) {
        port->ref();
    }
    return port;
}


// Temporary routine to allow boxes on one task's shared heap to be reparented
// to another.
const type_desc *
rust_task::release_alloc(void *alloc) {
    I(sched, !lock.lock_held_by_current_thread());
    lock.lock();

    assert(local_allocs.find(alloc) != local_allocs.end());
    const type_desc *tydesc = local_allocs[alloc];
    local_allocs.erase(alloc);

    local_region.release_alloc(alloc);

    lock.unlock();
    return tydesc;
}

// Temporary routine to allow boxes from one task's shared heap to be
// reparented to this one.
void
rust_task::claim_alloc(void *alloc, const type_desc *tydesc) {
    I(sched, !lock.lock_held_by_current_thread());
    lock.lock();

    assert(local_allocs.find(alloc) == local_allocs.end());
    local_allocs[alloc] = tydesc;
    local_region.claim_alloc(alloc);

    lock.unlock();
}

void
rust_task::notify(bool success) {
    // FIXME (1078) Do this in rust code
    if(user.notify_enabled) {
        rust_task *target_task = kernel->get_task_by_id(user.notify_chan.task);
        if (target_task) {
            rust_port *target_port =
                target_task->get_port_by_id(user.notify_chan.port);
            if(target_port) {
                task_notification msg;
                msg.id = user.id;
                msg.result = !success ? tr_failure : tr_success;

                target_port->send(&msg);
                scoped_lock with(target_task->lock);
                target_port->deref();
            }
            target_task->deref();
        }
    }
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
