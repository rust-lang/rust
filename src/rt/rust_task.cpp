
#include "rust_internal.h"

#include "valgrind.h"
#include "memcheck.h"

// Stacks

static size_t const min_stk_bytes = 0x300;

// Task stack segments. Heap allocated and chained together.

static stk_seg*
new_stk(rust_dom *dom, size_t minsz)
{
    if (minsz < min_stk_bytes)
        minsz = min_stk_bytes;
    size_t sz = sizeof(stk_seg) + minsz;
    stk_seg *stk = (stk_seg *)dom->malloc(sz);
    dom->logptr("new stk", (uintptr_t)stk);
    memset(stk, 0, sizeof(stk_seg));
    stk->limit = (uintptr_t) &stk->data[minsz];
    dom->logptr("stk limit", stk->limit);
    stk->valgrind_id =
        VALGRIND_STACK_REGISTER(&stk->data[0],
                                &stk->data[minsz]);
    return stk;
}

static void
del_stk(rust_dom *dom, stk_seg *stk)
{
    VALGRIND_STACK_DEREGISTER(stk->valgrind_id);
    dom->logptr("freeing stk segment", (uintptr_t)stk);
    dom->free(stk);
}

// Tasks

// FIXME (issue #31): ifdef by platform. This is getting absurdly
// x86-specific.

size_t const n_callee_saves = 4;
size_t const callee_save_fp = 0;

static uintptr_t
align_down(uintptr_t sp)
{
    // There is no platform we care about that needs more than a
    // 16-byte alignment.
    return sp & ~(16 - 1);
}


rust_task::rust_task(rust_dom *dom, rust_task *spawner) :
    stk(new_stk(dom, 0)),
    runtime_sp(0),
    rust_sp(stk->limit),
    gc_alloc_chain(0),
    dom(dom),
    cache(NULL),
    state(&dom->running_tasks),
    cond(NULL),
    dptr(0),
    spawner(spawner),
    idx(0),
    waiting_tasks(dom),
    alarm(this)
{
    dom->logptr("new task", (uintptr_t)this);
}

rust_task::~rust_task()
{
    dom->log(rust_log::MEM|rust_log::TASK,
             "~rust_task 0x%" PRIxPTR ", refcnt=%d",
             (uintptr_t)this, refcnt);

    /*
      for (uintptr_t fp = get_fp(); fp; fp = get_previous_fp(fp)) {
      frame_glue_fns *glue_fns = get_frame_glue_fns(fp);
      dom->log(rust_log::MEM|rust_log::TASK,
      "~rust_task, frame fp=0x%" PRIxPTR ", glue_fns=0x%" PRIxPTR,
      fp, glue_fns);
      if (glue_fns) {
      dom->log(rust_log::MEM|rust_log::TASK,
               "~rust_task, mark_glue=0x%" PRIxPTR,
               glue_fns->mark_glue);
      dom->log(rust_log::MEM|rust_log::TASK,
               "~rust_task, drop_glue=0x%" PRIxPTR,
               glue_fns->drop_glue);
      dom->log(rust_log::MEM|rust_log::TASK,
               "~rust_task, reloc_glue=0x%" PRIxPTR,
               glue_fns->reloc_glue);
      }
      }
    */

    /* FIXME: tighten this up, there are some more
       assertions that hold at task-lifecycle events. */
    I(dom, refcnt == 0 ||
      (refcnt == 1 && this == dom->root_task));

    del_stk(dom, stk);
    if (cache)
        cache->deref();
}

void
rust_task::start(uintptr_t exit_task_glue,
                 uintptr_t spawnee_fn,
                 uintptr_t args,
                 size_t callsz)
{
    dom->logptr("exit-task glue", exit_task_glue);
    dom->logptr("from spawnee", spawnee_fn);

    // Set sp to last uintptr_t-sized cell of segment and align down.
    rust_sp -= sizeof(uintptr_t);
    rust_sp = align_down(rust_sp);

    // Begin synthesizing frames. There are two: a "fully formed"
    // exit-task frame at the top of the stack -- that pretends to be
    // mid-execution -- and a just-starting frame beneath it that
    // starts executing the first instruction of the spawnee. The
    // spawnee *thinks* it was called by the exit-task frame above
    // it. It wasn't; we put that fake frame in place here, but the
    // illusion is enough for the spawnee to return to the exit-task
    // frame when it's done, and exit.
    uintptr_t *spp = (uintptr_t *)rust_sp;

    // The exit_task_glue frame we synthesize above the frame we activate:
    *spp-- = (uintptr_t) this;       // task
    *spp-- = (uintptr_t) 0;          // output
    *spp-- = (uintptr_t) 0;          // retpc
    for (size_t j = 0; j < n_callee_saves; ++j) {
        *spp-- = 0;
    }

    // We want 'frame_base' to point to the last callee-save in this
    // (exit-task) frame, because we're going to inject this
    // frame-pointer into the callee-save frame pointer value in the
    // *next* (spawnee) frame. A cheap trick, but this means the
    // spawnee frame will restore the proper frame pointer of the glue
    // frame as it runs its epilogue.
    uintptr_t frame_base = (uintptr_t) (spp+1);

    *spp-- = (uintptr_t) dom->root_crate;  // crate ptr
    *spp-- = (uintptr_t) 0;                // frame_glue_fns

    // Copy args from spawner to spawnee.
    if (args)  {
        uintptr_t *src = (uintptr_t *)args;
        src += 1;                  // spawn-call output slot
        src += 1;                  // spawn-call task slot
        // Memcpy all but the task and output pointers
        callsz -= (2 * sizeof(uintptr_t));
        spp = (uintptr_t*) (((uintptr_t)spp) - callsz);
        memcpy(spp, src, callsz);

        // Move sp down to point to task cell.
        spp--;
    } else {
        // We're at root, starting up.
        I(dom, callsz==0);
    }

    // The *implicit* incoming args to the spawnee frame we're
    // activating:

    *spp-- = (uintptr_t) this;            // task
    *spp-- = (uintptr_t) 0;               // output addr
    *spp-- = (uintptr_t) exit_task_glue;  // retpc

    // The context the activate_glue needs to switch stack.
    *spp-- = (uintptr_t) spawnee_fn;      // instruction to start at
    for (size_t j = 0; j < n_callee_saves; ++j) {
        // callee-saves to carry in when we activate
        if (j == callee_save_fp)
            *spp-- = frame_base;
        else
            *spp-- = NULL;
    }

    // Back up one, we overshot where sp should be.
    rust_sp = (uintptr_t) (spp+1);

    dom->add_task_to_state_vec(&dom->running_tasks, this);
}

void
rust_task::grow(size_t n_frame_bytes)
{
    stk_seg *old_stk = this->stk;
    uintptr_t old_top = (uintptr_t) old_stk->limit;
    uintptr_t old_bottom = (uintptr_t) &old_stk->data[0];
    uintptr_t rust_sp_disp = old_top - this->rust_sp;
    size_t ssz = old_top - old_bottom;
    dom->log(rust_log::MEM|rust_log::TASK|rust_log::UPCALL,
             "upcall_grow_task(%" PRIdPTR
             "), old size %" PRIdPTR
             " bytes (old lim: 0x%" PRIxPTR ")",
             n_frame_bytes, ssz, old_top);
    ssz *= 2;
    if (ssz < n_frame_bytes)
        ssz = n_frame_bytes;
    ssz = next_power_of_two(ssz);

    dom->log(rust_log::MEM|rust_log::TASK, "upcall_grow_task growing stk 0x%"
             PRIxPTR " to %d bytes", old_stk, ssz);

    stk_seg *nstk = new_stk(dom, ssz);
    uintptr_t new_top = (uintptr_t) &nstk->data[ssz];
    size_t n_copy = old_top - old_bottom;
    dom->log(rust_log::MEM|rust_log::TASK,
             "copying %d bytes of stack from [0x%" PRIxPTR ", 0x%" PRIxPTR "]"
             " to [0x%" PRIxPTR ", 0x%" PRIxPTR "]",
             n_copy,
             old_bottom, old_bottom + n_copy,
             new_top - n_copy, new_top);

    VALGRIND_MAKE_MEM_DEFINED((void*)old_bottom, n_copy);
    memcpy((void*)(new_top - n_copy), (void*)old_bottom, n_copy);

    nstk->limit = new_top;
    this->stk = nstk;
    this->rust_sp = new_top - rust_sp_disp;

    dom->log(rust_log::MEM|rust_log::TASK, "processing relocations");

    // FIXME (issue #32): this is the most ridiculously crude
    // relocation scheme ever. Try actually, you know, writing out
    // reloc descriptors?
    size_t n_relocs = 0;
    for (uintptr_t* p = (uintptr_t*)(new_top - n_copy);
         p < (uintptr_t*)new_top; ++p) {
        if (old_bottom <= *p && *p < old_top) {
            //dom->log(rust_log::MEM, "relocating pointer 0x%" PRIxPTR
            //        " by %d bytes", *p, (new_top - old_top));
            n_relocs++;
            *p += (new_top - old_top);
        }
    }
    dom->log(rust_log::MEM|rust_log::TASK,
             "processed %d relocations", n_relocs);
    del_stk(dom, old_stk);
    dom->logptr("grown stk limit", new_top);
}

void
push_onto_thread_stack(uintptr_t &sp, uintptr_t value)
{
    asm("xchgl %0, %%esp\n"
        "push %2\n"
        "xchgl %0, %%esp\n"
        : "=r" (sp)
        : "0" (sp), "r" (value)
        : "eax");
}

void
rust_task::run_after_return(size_t nargs, uintptr_t glue)
{
    // This is only safe to call if we're the currently-running task.
    check_active();

    uintptr_t sp = runtime_sp;

    // The compiler reserves nargs + 1 word for oldsp on the stack and
    // then aligns it.
    sp = align_down(sp - nargs * sizeof(uintptr_t));

    uintptr_t *retpc = ((uintptr_t *) sp) - 1;
    dom->log(rust_log::TASK|rust_log::MEM,
             "run_after_return: overwriting retpc=0x%" PRIxPTR
             " @ runtime_sp=0x%" PRIxPTR
             " with glue=0x%" PRIxPTR,
             *retpc, sp, glue);

    // Move the current return address (which points into rust code)
    // onto the rust stack and pretend we just called into the glue.
    push_onto_thread_stack(rust_sp, *retpc);
    *retpc = glue;
}

void
rust_task::run_on_resume(uintptr_t glue)
{
    // This is only safe to call if we're suspended.
    check_suspended();

    // Inject glue as resume address in the suspended frame.
    uintptr_t* rsp = (uintptr_t*) rust_sp;
    rsp += n_callee_saves;
    dom->log(rust_log::TASK|rust_log::MEM,
             "run_on_resume: overwriting retpc=0x%" PRIxPTR
             " @ rust_sp=0x%" PRIxPTR
             " with glue=0x%" PRIxPTR,
             *rsp, rsp, glue);
    *rsp = glue;
}

void
rust_task::yield(size_t nargs)
{
    dom->log(rust_log::TASK,
             "task 0x%" PRIxPTR " yielding", this);
    run_after_return(nargs, dom->root_crate->get_yield_glue());
}

static inline uintptr_t
get_callee_save_fp(uintptr_t *top_of_callee_saves)
{
    return top_of_callee_saves[n_callee_saves - (callee_save_fp + 1)];
}

void
rust_task::kill() {
    // Note the distinction here: kill() is when you're in an upcall
    // from task A and want to force-fail task B, you do B->kill().
    // If you want to fail yourself you do self->fail(upcall_nargs).
    dom->log(rust_log::TASK, "killing task 0x%" PRIxPTR, this);
    // Unblock the task so it can unwind.
    unblock();
    if (this == dom->root_task)
        dom->fail();
    run_on_resume(dom->root_crate->get_unwind_glue());
}

void
rust_task::fail(size_t nargs) {
    // See note in ::kill() regarding who should call this.
    dom->log(rust_log::TASK, "task 0x%" PRIxPTR " failing", this);
    // Unblock the task so it can unwind.
    unblock();
    if (this == dom->root_task)
        dom->fail();
    run_after_return(nargs, dom->root_crate->get_unwind_glue());
    if (spawner) {
        dom->log(rust_log::TASK,
                 "task 0x%" PRIxPTR
                 " propagating failure to parent 0x%" PRIxPTR,
                 this, spawner);
        spawner->kill();
    }
}

void
rust_task::gc(size_t nargs)
{
    dom->log(rust_log::TASK|rust_log::MEM,
             "task 0x%" PRIxPTR " garbage collecting", this);
    run_after_return(nargs, dom->root_crate->get_gc_glue());
}

void
rust_task::notify_waiting_tasks()
{
    while (waiting_tasks.length() > 0) {
        rust_task *t = waiting_tasks.pop()->receiver;
        if (!t->dead())
            t->wakeup(this);
    }
}

uintptr_t
rust_task::get_fp() {
    // sp in any suspended task points to the last callee-saved reg on
    // the task stack.
    return get_callee_save_fp((uintptr_t*)rust_sp);
}

uintptr_t
rust_task::get_previous_fp(uintptr_t fp) {
    // fp happens to, coincidentally (!) also point to the last
    // callee-save on the task stack.
    return get_callee_save_fp((uintptr_t*)fp);
}

frame_glue_fns*
rust_task::get_frame_glue_fns(uintptr_t fp) {
    fp -= sizeof(uintptr_t);
    return *((frame_glue_fns**) fp);
}

bool
rust_task::running()
{
    return state == &dom->running_tasks;
}

bool
rust_task::blocked()
{
    return state == &dom->blocked_tasks;
}

bool
rust_task::blocked_on(rust_cond *on)
{
    return blocked() && cond == on;
}

bool
rust_task::dead()
{
    return state == &dom->dead_tasks;
}

void
rust_task::transition(ptr_vec<rust_task> *src, ptr_vec<rust_task> *dst)
{
    I(dom, state == src);
    dom->log(rust_log::TASK,
             "task 0x%" PRIxPTR " state change '%s' -> '%s'",
             (uintptr_t)this,
             dom->state_vec_name(src),
             dom->state_vec_name(dst));
    dom->remove_task_from_state_vec(src, this);
    dom->add_task_to_state_vec(dst, this);
    state = dst;
}

void
rust_task::block(rust_cond *on)
{
    I(dom, on);
    transition(&dom->running_tasks, &dom->blocked_tasks);
    dom->log(rust_log::TASK,
             "task 0x%" PRIxPTR " blocking on 0x%" PRIxPTR,
             (uintptr_t)this,
             (uintptr_t)on);
    cond = on;
}

void
rust_task::wakeup(rust_cond *from)
{
    transition(&dom->blocked_tasks, &dom->running_tasks);
    I(dom, cond == from);
}

void
rust_task::die()
{
    transition(&dom->running_tasks, &dom->dead_tasks);
}

void
rust_task::unblock()
{
    if (blocked())
        wakeup(cond);
}

rust_crate_cache *
rust_task::get_crate_cache(rust_crate const *curr_crate)
{
    if (cache && cache->crate != curr_crate) {
        dom->log(rust_log::TASK, "switching task crate-cache to crate 0x%"
                 PRIxPTR, curr_crate);
        cache->deref();
        cache = NULL;
    }

    if (!cache) {
        dom->log(rust_log::TASK, "fetching cache for current crate");
        cache = dom->get_cache(curr_crate);
    }
    return cache;
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
