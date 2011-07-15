
#include "rust_internal.h"

#include "valgrind.h"
#include "memcheck.h"

#ifndef __WIN32__
#include <execinfo.h>
#endif

#include "globals.h"

// Stacks

// FIXME (issue #151): This should be 0x300; the change here is for
// practicality's sake until stack growth is working.

static size_t get_min_stk_size() {
    char *stack_size = getenv("RUST_MIN_STACK");
    if(stack_size) {
        return atoi(stack_size);
    }
    else {
        return 0x200000;
    }
}

// Task stack segments. Heap allocated and chained together.

static stk_seg*
new_stk(rust_task *task, size_t minsz)
{
    size_t min_stk_bytes = get_min_stk_size();
    if (minsz < min_stk_bytes)
        minsz = min_stk_bytes;
    size_t sz = sizeof(stk_seg) + minsz;
    stk_seg *stk = (stk_seg *)task->malloc(sz);
    LOGPTR(task->sched, "new stk", (uintptr_t)stk);
    memset(stk, 0, sizeof(stk_seg));
    stk->limit = (uintptr_t) &stk->data[minsz];
    LOGPTR(task->sched, "stk limit", stk->limit);
    stk->valgrind_id =
        VALGRIND_STACK_REGISTER(&stk->data[0],
                                &stk->data[minsz]);
    return stk;
}

static void
del_stk(rust_task *task, stk_seg *stk)
{
    VALGRIND_STACK_DEREGISTER(stk->valgrind_id);
    LOGPTR(task->sched, "freeing stk segment", (uintptr_t)stk);
    task->free(stk);
}

// Tasks

// FIXME (issue #31): ifdef by platform. This is getting absurdly
// x86-specific.

size_t const n_callee_saves = 4;
size_t const callee_save_fp = 0;

rust_task::rust_task(rust_scheduler *sched, rust_task_list *state,
                     rust_task *spawner, const char *name) :
    maybe_proxy<rust_task>(this),
    stk(NULL),
    runtime_sp(0),
    rust_sp(0),
    gc_alloc_chain(0),
    sched(sched),
    cache(NULL),
    kernel(sched->kernel),
    name(name),
    state(state),
    cond(NULL),
    cond_name("none"),
    supervisor(spawner),
    list_index(-1),
    rendezvous_ptr(0),
    handle(NULL),
    running_on(-1),
    pinned_on(-1),
    local_region(&sched->srv->local_region),
    _on_wakeup(NULL),
    failed(false)
{
    LOGPTR(sched, "new task", (uintptr_t)this);
    DLOG(sched, task, "sizeof(task) = %d (0x%x)", sizeof *this, sizeof *this);

    stk = new_stk(this, 0);
    rust_sp = stk->limit;

    if (spawner == NULL) {
        ref_count = 0;
    }
}

rust_task::~rust_task()
{
    DLOG(sched, task, "~rust_task %s @0x%" PRIxPTR ", refcnt=%d",
         name, (uintptr_t)this, ref_count);

    /* FIXME: tighten this up, there are some more
       assertions that hold at task-lifecycle events. */
    I(sched, ref_count == 0 ||
      (ref_count == 1 && this == sched->root_task));

    del_stk(this, stk);
}

extern "C" void rust_new_exit_task_glue();

struct spawn_args {
    rust_task *task;
    uintptr_t a3;
    uintptr_t a4;
    void (*CDECL f)(int *, rust_task *,
                       uintptr_t, uintptr_t);
};

extern "C" CDECL
void task_start_wrapper(spawn_args *a)
{
    rust_task *task = a->task;
    int rval = 42;

    a->f(&rval, task, a->a3, a->a4);

    LOG(task, task, "task exited with value %d", rval);


    LOG(task, task, "task ref_count: %d", task->ref_count);
    A(task->sched, task->ref_count >= 0,
      "Task ref_count should not be negative on exit!");
    task->die();
    task->lock.lock();
    task->notify_tasks_waiting_to_join();
    task->lock.unlock();

    task->yield(1);
}

void
rust_task::start(uintptr_t spawnee_fn,
                 uintptr_t args)
{
    LOGPTR(sched, "from spawnee", spawnee_fn);

    I(sched, stk->data != NULL);

    char *sp = (char *)rust_sp;

    sp -= sizeof(spawn_args);

    spawn_args *a = (spawn_args *)sp;

    a->task = this;
    a->a3 = 0;
    a->a4 = args;
    void **f = (void **)&a->f;
    *f = (void *)spawnee_fn;

    ctx.call((void *)task_start_wrapper, a, sp);

    yield_timer.reset_us(0);
    transition(&sched->newborn_tasks, &sched->running_tasks);
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

    yield_timer.reset_us(time_in_us);

    // Return to the scheduler.
    ctx.next->swap(ctx);
}

void
rust_task::kill() {
    if (dead()) {
        // Task is already dead, can't kill what's already dead.
        return;
    }

    // Note the distinction here: kill() is when you're in an upcall
    // from task A and want to force-fail task B, you do B->kill().
    // If you want to fail yourself you do self->fail().
    LOG(this, task, "killing task %s @0x%" PRIxPTR, name, this);
    // Unblock the task so it can unwind.
    unblock();

    if (this == sched->root_task)
        sched->fail();

    LOG(this, task, "preparing to unwind task: 0x%" PRIxPTR, this);
    // run_on_resume(rust_unwind_glue);
}

void
rust_task::fail() {
    // See note in ::kill() regarding who should call this.
    DLOG(sched, task, "task %s @0x%" PRIxPTR " failing", name, this);
    backtrace();
    // Unblock the task so it can unwind.
    unblock();
    if (supervisor) {
        DLOG(sched, task,
             "task %s @0x%" PRIxPTR
             " propagating failure to supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
        supervisor->kill();
    }
    // FIXME: implement unwinding again.
    if (this == sched->root_task)
        sched->fail();
    failed = true;
}

void
rust_task::gc()
{
    // FIXME: not presently implemented; was broken by rustc.
    DLOG(sched, task,
             "task %s @0x%" PRIxPTR " garbage collecting", name, this);
}

void
rust_task::unsupervise()
{
    DLOG(sched, task,
             "task %s @0x%" PRIxPTR
             " disconnecting from supervisor %s @0x%" PRIxPTR,
             name, this, supervisor->name, supervisor);
    supervisor = NULL;
}

void
rust_task::notify_tasks_waiting_to_join() {
    while (tasks_waiting_to_join.is_empty() == false) {
        LOG(this, task, "notify_tasks_waiting_to_join: %d",
            tasks_waiting_to_join.size());
        maybe_proxy<rust_task> *waiting_task = 0;
        tasks_waiting_to_join.pop(&waiting_task);
        if (waiting_task->is_proxy()) {
            notify_message::send(notify_message::WAKEUP, "wakeup",
                get_handle(), waiting_task->as_proxy()->handle());
            delete waiting_task;
        } else {
            rust_task *task = waiting_task->referent();
            if (task->blocked() == true) {
                task->wakeup(this);
            }
        }
    }
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

void
rust_task::link_gc(gc_alloc *gcm) {
    I(sched, gcm->prev == NULL);
    I(sched, gcm->next == NULL);
    gcm->prev = NULL;
    gcm->next = gc_alloc_chain;
    gc_alloc_chain = gcm;
    if (gcm->next)
        gcm->next->prev = gcm;
}

void
rust_task::unlink_gc(gc_alloc *gcm) {
    if (gcm->prev)
        gcm->prev->next = gcm->next;
    if (gcm->next)
        gcm->next->prev = gcm->prev;
    if (gc_alloc_chain == gcm)
        gc_alloc_chain = gcm->next;
    gcm->prev = NULL;
    gcm->next = NULL;
}

void *
rust_task::malloc(size_t sz, type_desc *td)
{
    // FIXME: GC is disabled for now.
    // GC-memory classification is all wrong.
    td = NULL;

    if (td) {
        sz += sizeof(gc_alloc);
    }
    void *mem = local_region.malloc(sz);
    if (!mem)
        return mem;
    if (td) {
        gc_alloc *gcm = (gc_alloc*) mem;
        DLOG(sched, task, "task %s @0x%" PRIxPTR
             " allocated %d GC bytes = 0x%" PRIxPTR,
             name, (uintptr_t)this, sz, gcm);
        memset((void*) gcm, 0, sizeof(gc_alloc));
        link_gc(gcm);
        gcm->ctrl_word = (uintptr_t)td;
        gc_alloc_accum += sz;
        mem = (void*) &(gcm->data);
    }
    return mem;;
}

void *
rust_task::realloc(void *data, size_t sz, bool is_gc)
{
    // FIXME: GC is disabled for now.
    // Effects, GC-memory classification is all wrong.
    is_gc = false;
    if (is_gc) {
        gc_alloc *gcm = (gc_alloc*)(((char *)data) - sizeof(gc_alloc));
        unlink_gc(gcm);
        sz += sizeof(gc_alloc);
        gcm = (gc_alloc*) local_region.realloc((void*)gcm, sz);
        DLOG(sched, task, "task %s @0x%" PRIxPTR
             " reallocated %d GC bytes = 0x%" PRIxPTR,
             name, (uintptr_t)this, sz, gcm);
        if (!gcm)
            return gcm;
        link_gc(gcm);
        data = (void*) &(gcm->data);
    } else {
        data = local_region.realloc(data, sz);
    }
    return data;
}

void
rust_task::free(void *p, bool is_gc)
{
    // FIXME: GC is disabled for now.
    // GC-memory classification is all wrong.
    is_gc = false;
    if (is_gc) {
        gc_alloc *gcm = (gc_alloc*)(((char *)p) - sizeof(gc_alloc));
        unlink_gc(gcm);
        DLOG(sched, mem,
             "task %s @0x%" PRIxPTR " freeing GC memory = 0x%" PRIxPTR,
             name, (uintptr_t)this, gcm);
        DLOG(sched, mem, "rust_task::free(0x%" PRIxPTR ")", gcm);
        local_region.free(gcm);
    } else {
        DLOG(sched, mem, "rust_task::free(0x%" PRIxPTR ")", p);
        local_region.free(p);
    }
}

void
rust_task::transition(rust_task_list *src, rust_task_list *dst) {
    I(sched, !kernel->scheduler_lock.lock_held_by_current_thread());
    scoped_lock with(kernel->scheduler_lock);
    DLOG(sched, task,
         "task %s " PTR " state change '%s' -> '%s' while in '%s'",
         name, (uintptr_t)this, src->name, dst->name, state->name);
    I(sched, state == src);
    src->remove(this);
    dst->append(this);
    state = dst;
}

void
rust_task::block(rust_cond *on, const char* name) {
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
}

void
rust_task::die() {
    scoped_lock with(lock);
    transition(&sched->running_tasks, &sched->dead_tasks);
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

rust_handle<rust_task> *
rust_task::get_handle() {
    if (handle == NULL) {
        handle = sched->kernel->get_task_handle(this);
    }
    return handle;
}

bool rust_task::can_schedule(int id)
{
    return yield_timer.has_timed_out() &&
        running_on == -1 &&
        (pinned_on == -1 || pinned_on == id);
}

void *
rust_task::calloc(size_t size) {
    return local_region.calloc(size);
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
