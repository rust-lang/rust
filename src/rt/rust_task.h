// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
   The rust task is a cooperatively-scheduled green thread that executes
   Rust code on a segmented stack.

   This class has too many responsibilities:

   * Working with the scheduler loop to signal and respond to state changes,
   and dealing with all the thread synchronization issues involved

   * Managing the dynamically resizing list of Rust stack segments

   * Switching between running Rust code on the Rust segmented stack and
   foreign C code on large stacks owned by the scheduler

   # Lifetime

   The lifetime of a rust_task object closely mirrors that of a running Rust
   task object, but they are not identical. In particular, the rust_task is an
   atomically reference counted object that might be accessed from arbitrary
   threads at any time. This may keep the task from being destroyed even after
   the task is dead from a Rust task lifecycle perspective. The rust_tasks are
   reference counted in the following places:

   * By the task's lifetime (i.e., running tasks hold a reference to themself)

   * In the rust_task_kill_all -> rust_kernel::fail ->
     rust_sched_loop::kill_all_tasks path. When a task brings down the whole
     runtime, each sched_loop must use refcounts to take a 'snapshot' of all
     existing tasks so it can be sure to kill all of them.

   * In core::pipes, tasks that use select() use reference counts to avoid
     use-after-free races with multiple different signallers.

   # Death

   All task death goes through a single central path: The task invokes
   rust_task::die(), which invokes transition(task_state_dead), which pumps
   the scheduler loop, which switches to rust_sched_loop::run_single_turn(),
   which calls reap_dead_tasks(), which cleans up the task's stack segments
   and drops the reference count.

   When a task's reference count hits zero, rust_sched_loop::release_task()
   is called. This frees the memory and deregisters the task from the kernel,
   which may trigger the sched_loop, the scheduler, and/or the kernel to exit
   completely in the case it was the last task alive.

   die() is called from two places: the successful exit path, in cleanup_task,
   and on failure (on linux, this is also in cleanup_task, after unwinding
   completes; on windows, it is in begin_failure).

   Tasks do not force-quit other tasks; a task die()s only itself. However...

   # Killing

   Tasks may kill each other. This happens when propagating failure between
   tasks (see the task::spawn options interface). The code path for this is
   rust_task_kill_other() -> rust_task::kill().

   It also happens when the main ("root") task (or any task in that task's
   linked-failure-group) fails: this brings down the whole runtime, and kills
   all tasks in all groups. The code path for this is rust_task_kill_all() ->
   rust_kernel::fail() -> rust_scheduler::kill_all_tasks() ->
   rust_sched_loop::kill_all_tasks() -> rust_task::kill().

   In either case, killing a task involves, under the protection of its
   lifecycle_lock, (a) setting the 'killed' flag, and (b) checking if it is
   'blocked'* and if so punting it awake.
   (* and also isn't unkillable, which may happen via task::unkillable()
   or via calling an extern rust function from C.)

   The killed task will then (wake up if it was asleep, and) eventually call
   yield() (or wait_event()), which will check the killed flag, see that it is
   true, and then invoke 'fail', which begins the death process described
   above.

   Three things guarantee concurrency safety in this whole affair:

   * The lifecycle_lock protects tasks accessing each other's state: it makes
     killing-and-waking up atomic with respect to a task in block() deciding
     whether it's allowed to go to sleep, so tasks can't 'escape' being woken.

   * In the case of linked failure propagation, we ensure (in task.rs) that
     tasks can only see another task's rust_task pointer if that task is
     already alive. Even before entering the runtime failure path, a task will
     access (locked) the linked-failure data structures to remove its task
     pointer so that no subsequently-failing tasks will do a use-after-free.

   * In the case of bringing down the whole runtime, each sched_loop takes an
     "atomic snapshot" of all its tasks, protected by the sched_loop's lock,
     and also sets a 'failing' flag so that any subsequently-failing task will
     know that it must fail immediately upon creation (which is also checked
     under the same lock). A similar process exists at the one-step-higher
     level of the kernel killing all the schedulers (the kernel snapshots all
     the schedulers and sets a 'failing' flag in the scheduler table).
 */

#ifndef RUST_TASK_H
#define RUST_TASK_H

#include <map>

#include "rust_globals.h"
#include "util/array_list.h"
#include "context.h"
#include "rust_debug.h"
#include "rust_kernel.h"
#include "boxed_region.h"
#include "rust_stack.h"
#include "rust_type.h"
#include "rust_sched_loop.h"
#include "sp.h"

// The amount of extra space at the end of each stack segment, available
// to the rt, compiler and dynamic linker for running small functions
// FIXME (#1509): We want this to be 128 but need to slim the red zone calls
// down, disable lazy symbol relocation, and other things we haven't
// discovered yet
#define RZ_LINUX_32 (1024*2)
#define RZ_LINUX_64 (1024*2)
#define RZ_MAC_32   (1024*20)
#define RZ_MAC_64   (1024*20)
#define RZ_WIN_32   (1024*20)
#define RZ_BSD_32   (1024*20)
#define RZ_BSD_64   (1024*20)

// The threshold beyond which we switch to the C stack.
#define STACK_THRESHOLD (1024 * 1024)

#ifdef __linux__
#ifdef __i386__
#define RED_ZONE_SIZE RZ_LINUX_32
#endif
#ifdef __x86_64__
#define RED_ZONE_SIZE RZ_LINUX_64
#endif
#ifdef __mips__
#define RED_ZONE_SIZE RZ_MAC_32
#endif
#endif
#ifdef __APPLE__
#ifdef __i386__
#define RED_ZONE_SIZE RZ_MAC_32
#endif
#ifdef __x86_64__
#define RED_ZONE_SIZE RZ_MAC_64
#endif
#endif
#ifdef __WIN32__
#ifdef __i386__
#define RED_ZONE_SIZE RZ_WIN_32
#endif
#ifdef __x86_64__
#define RED_ZONE_SIZE RZ_WIN_64
#endif
#endif
#ifdef __FreeBSD__
#ifdef __i386__
#define RED_ZONE_SIZE RZ_BSD_32
#endif
#ifdef __x86_64__
#define RED_ZONE_SIZE RZ_BSD_64
#endif
#endif
#ifdef __ANDROID__
#define RED_ZONE_SIZE RZ_MAC_32
#endif

#ifndef RED_ZONE_SIZE
# error "Red zone not defined for this platform"
#endif

struct frame_glue_fns {
    uintptr_t mark_glue_off;
    uintptr_t drop_glue_off;
    uintptr_t reloc_glue_off;
};

// std::lib::task::task_result
typedef unsigned long task_result;
#define tr_success 0
#define tr_failure 1

struct spawn_args;
struct cleanup_args;
struct reset_args;
struct new_stack_args;

// std::lib::task::task_notification
//
// since it's currently a unary tag, we only add the fields.
struct task_notification {
    rust_task_id id;
    task_result result; // task_result
};

extern "C" void
rust_task_fail(rust_task *task,
               char const *expr,
               char const *file,
               size_t line);

struct
rust_task : public kernel_owned<rust_task>
{
    RUST_ATOMIC_REFCOUNT();

    rust_task_id id;

    context ctx;
    stk_seg *stk;
    uintptr_t runtime_sp;      // Runtime sp while task running.
    rust_scheduler *sched;
    rust_sched_loop *sched_loop;

    // Fields known only to the runtime.
    rust_kernel *kernel;
    const char *const name;
    int32_t list_index;

    boxed_region boxed;
    memory_region local_region;

    // Indicates that fail() has been called and we are cleaning up.
    // We use this to suppress the "killed" flag during calls to yield.
    bool unwinding;

    bool propagate_failure;

    debug::task_debug_info debug;

    // The amount of stack we're using, excluding red zones
    size_t total_stack_sz;

    // Used by rust task management routines in libcore/task.rs.
    void *task_local_data;
    void (*task_local_data_cleanup)(void *data);

    // Contains a ~[BorrowRecord] pointer, or NULL.
    //
    // Used by borrow management code in libcore/unstable/lang.rs.
    void *borrow_list;

private:

    // Protects state, cond, cond_name
    // Protects the killed flag, disallow_kill flag, reentered_rust_stack
    lock_and_signal lifecycle_lock;
    rust_task_state state;
    rust_cond *cond;
    const char *cond_name;

    bool event_reject;
    rust_cond event_cond;
    void *event;

    // Indicates that the task was killed and needs to unwind
    bool killed;
    // Indicates that we've called back into Rust from C
    bool reentered_rust_stack;
    unsigned long disallow_kill;
    unsigned long disallow_yield;

    // The stack used for running C code, borrowed from the scheduler thread
    stk_seg *c_stack;
    uintptr_t next_c_sp;
    uintptr_t next_rust_sp;

    // The big stack.
    stk_seg *big_stack;

    // Called when the atomic refcount reaches zero
    void delete_this();

    bool new_big_stack();
    void new_stack_fast(size_t requested_sz);
    void new_stack(size_t requested_sz);
    void free_stack(stk_seg *stk);
    size_t get_next_stack_size(size_t min, size_t current, size_t requested);

    void return_c_stack();

    void transition(rust_task_state src, rust_task_state dst,
                    rust_cond *cond, const char* cond_name);
    void transition_inner(rust_task_state src, rust_task_state dst,
                           rust_cond *cond, const char* cond_name);

    bool must_fail_from_being_killed_inner();
    // Called by rust_task_fail to unwind on failure
    void begin_failure(char const *expr,
                       char const *file,
                       size_t line);

    friend void task_start_wrapper(spawn_args *a);
    friend void cleanup_task(cleanup_args *a);
    friend void reset_stack_limit_on_c_stack(reset_args *a);
    friend void new_stack_slow(new_stack_args *a);
    friend void rust_task_fail(rust_task *task,
                               char const *expr,
                               char const *file,
                               size_t line);

    bool block_inner(rust_cond *on, const char* name);
    void wakeup_inner(rust_cond *from);
    bool blocked_on(rust_cond *cond);

private:
    // private and undefined to disable copying
    rust_task(const rust_task& rhs);
    rust_task& operator=(const rust_task& rhs);

public:

    // Only a pointer to 'name' is kept, so it must live as long as this task.
    rust_task(rust_sched_loop *sched_loop,
              rust_task_state state,
              const char *name,
              size_t init_stack_sz);

    void start(spawn_fn spawnee_fn,
               rust_opaque_box *env,
               void *args);
    void start();
    void assert_is_running();

    void *malloc(size_t sz, const char *tag, type_desc *td=0);
    void *realloc(void *data, size_t sz);
    void free(void *p);

    void set_state(rust_task_state state,
                   rust_cond *cond, const char* cond_name);

    bool block(rust_cond *on, const char* name);
    void wakeup(rust_cond *from);
    void die();

    // Print a backtrace, if the "bt" logging option is on.
    void backtrace();

    // Yields control to the scheduler. Called from the Rust stack
    // Returns TRUE if the task was killed and needs to fail.
    MUST_CHECK bool yield();

    // Fail this task (assuming caller-on-stack is different task).
    void kill();
    void kill_inner();

    // Indicates that we've been killed and now is an apropriate
    // time to fail as a result
    bool must_fail_from_being_killed();

    // Fail self, assuming caller-on-stack is this task.
    void fail();
    void fail(char const *expr, char const *file, size_t line);

    // Propagate failure to the entire rust runtime.
    void fail_sched_loop();

    void *calloc(size_t size, const char *tag);

    // Use this function sparingly. Depending on the ref count is generally
    // not at all safe.
    intptr_t get_ref_count() const { return ref_count; }

    void *next_stack(size_t stk_sz, void *args_addr, size_t args_sz);
    void prev_stack();
    void record_stack_limit();
    void reset_stack_limit();

    bool on_rust_stack();
    void check_stack_canary();
    void delete_all_stacks();

    void call_on_c_stack(void *args, void *fn_ptr);
    void call_on_rust_stack(void *args, void *fn_ptr);
    bool have_c_stack() { return c_stack != NULL; }
    stk_seg *get_c_stack() { return c_stack; }

    rust_task_state get_state() { return state; }
    rust_cond *get_cond() { return cond; }
    const char *get_cond_name() { return cond_name; }

    void clear_event_reject() {
        this->event_reject = false;
    }

    // Returns TRUE if the task was killed and needs to fail.
    MUST_CHECK bool wait_event(void **result);
    void signal_event(void *event);

    void cleanup_after_turn();

    void inhibit_kill();
    void allow_kill();
    void inhibit_yield();
    void allow_yield();
};

template <typename T> struct task_owned {
    inline void *operator new(size_t size, rust_task *task,
                                             const char *tag) {
        return task->malloc(size, tag);
    }

    inline void *operator new[](size_t size, rust_task *task,
                                               const char *tag) {
        return task->malloc(size, tag);
    }

    inline void *operator new(size_t size, rust_task &task,
                                             const char *tag) {
        return task.malloc(size, tag);
    }

    inline void *operator new[](size_t size, rust_task &task,
                                               const char *tag) {
        return task.malloc(size, tag);
    }

    void operator delete(void *ptr) {
        ((T *)ptr)->task->free(ptr);
    }
};

// This is the function that switches between the C and the Rust stack by
// calling another function with a single void* argument while changing the
// stack pointer. It has a funny name because gdb doesn't normally like to
// backtrace through split stacks (thinks it indicates a bug), but has a
// special case to allow functions named __morestack to move the stack pointer
// around.
extern "C" void __morestack(void *args, void *fn_ptr, uintptr_t stack_ptr);

inline static uintptr_t
sanitize_next_sp(uintptr_t next_sp) {

    // Since I'm not precisely sure where the next stack pointer sits in
    // relation to where the context switch actually happened, nor in relation
    // to the amount of stack needed for calling __morestack I've added some
    // extra bytes here.

    // FIXME (#2698): On the rust stack this potentially puts is quite far
    // into the red zone. Might want to just allocate a new rust stack every
    // time we switch back to rust.
    const uintptr_t padding = 16;

    return align_down(next_sp - padding);
}

inline void
rust_task::call_on_c_stack(void *args, void *fn_ptr) {
    // Too expensive to check
    // assert(on_rust_stack());

    // The shim functions generated by rustc contain the morestack prologue,
    // so we need to let them know they have enough stack.
    record_sp_limit(0);

    uintptr_t prev_rust_sp = next_rust_sp;
    next_rust_sp = get_sp();

    bool borrowed_a_c_stack = false;
    uintptr_t sp;
    if (c_stack == NULL) {
        c_stack = sched_loop->borrow_c_stack();
        next_c_sp = align_down(c_stack->end);
        sp = next_c_sp;
        borrowed_a_c_stack = true;
    } else {
        sp = sanitize_next_sp(next_c_sp);
    }

    __morestack(args, fn_ptr, sp);

    // Note that we may not actually get here if we threw an exception,
    // in which case we will return the c stack when the exception is caught.
    if (borrowed_a_c_stack) {
        return_c_stack();
    }

    next_rust_sp = prev_rust_sp;

    record_stack_limit();
}

inline void
rust_task::call_on_rust_stack(void *args, void *fn_ptr) {
    // Too expensive to check
    // assert(!on_rust_stack());

    // Because of the hack in the other function that disables the stack limit
    // when entering the C stack, here we restore the stack limit again.
    record_stack_limit();

    assert(get_sp_limit() != 0 && "Stack must be configured");
    assert(next_rust_sp);

    // Unlocked access. Might "race" a killer, but harmlessly. This code is
    // only run by the task itself, so cannot race itself. See the comment
    // above inhibit_kill (or #3213) in rust_task.cpp for justification.
    bool had_reentered_rust_stack = reentered_rust_stack;
    reentered_rust_stack = true;

    uintptr_t prev_c_sp = next_c_sp;
    next_c_sp = get_sp();

    uintptr_t sp = sanitize_next_sp(next_rust_sp);

    // FIXME (#2047): There are times when this is called and needs
    // to be able to throw, and we don't account for that.
    __morestack(args, fn_ptr, sp);

    next_c_sp = prev_c_sp;
    reentered_rust_stack = had_reentered_rust_stack;

    record_sp_limit(0);
}

inline void
rust_task::return_c_stack() {
    // Too expensive to check
    // assert(on_rust_stack());
    assert(c_stack != NULL);
    sched_loop->return_c_stack(c_stack);
    c_stack = NULL;
    next_c_sp = 0;
}

// NB: This runs on the Rust stack
inline void *
rust_task::next_stack(size_t stk_sz, void *args_addr, size_t args_sz) {
    new_stack_fast(stk_sz + args_sz);
    assert(stk->end - (uintptr_t)stk->data >= stk_sz + args_sz
      && "Did not receive enough stack");
    uint8_t *new_sp = (uint8_t*)stk->end;
    // Push the function arguments to the new stack
    new_sp = align_down(new_sp - args_sz);

    // I don't know exactly where the region ends that valgrind needs us
    // to mark accessible. On x86_64 these extra bytes aren't needed, but
    // on i386 we get errors without.
    const int fudge_bytes = 16;
    reuse_valgrind_stack(stk, new_sp - fudge_bytes);

    memcpy(new_sp, args_addr, args_sz);
    record_stack_limit();
    return new_sp;
}

// The amount of stack in a segment available to Rust code
inline size_t
user_stack_size(stk_seg *stk) {
    return (size_t)(stk->end
                    - (uintptr_t)&stk->data[0]
                    - RED_ZONE_SIZE);
}

struct new_stack_args {
    rust_task *task;
    size_t requested_sz;
};

void
new_stack_slow(new_stack_args *args);

// NB: This runs on the Rust stack
// This is the new stack fast path, in which we
// reuse the next cached stack segment
inline void
rust_task::new_stack_fast(size_t requested_sz) {
    // The minimum stack size, in bytes, of a Rust stack, excluding red zone
    size_t min_sz = sched_loop->min_stack_size;

    if (requested_sz > STACK_THRESHOLD) {
        if (new_big_stack())
            return;
    }

    // Try to reuse an existing stack segment
    if (stk != NULL && stk->next != NULL) {
        size_t next_sz = user_stack_size(stk->next);
        if (min_sz <= next_sz && requested_sz <= next_sz) {
            stk = stk->next;
            return;
        }
    }

    new_stack_args args = {this, requested_sz};
    call_on_c_stack(&args, (void*)new_stack_slow);
}

// NB: This runs on the Rust stack
inline void
rust_task::prev_stack() {
    // We're not going to actually delete anything now because that would
    // require switching to the C stack and be costly. Instead we'll just move
    // up the link list and clean up later, either in new_stack or after our
    // turn ends on the scheduler.
    stk = stk->prev;
    record_stack_limit();
}

// The LLVM-generated segmented-stack function prolog compares the amount of
// stack needed for each frame to the end-of-stack pointer stored in the
// TCB. As an optimization, when the frame size is less than 256 bytes, it
// will simply compare %esp to to the stack limit instead of subtracting the
// frame size. As a result we need our stack limit to account for those 256
// bytes.
const unsigned LIMIT_OFFSET = 256;

inline void
rust_task::record_stack_limit() {
    assert(stk);
    assert((uintptr_t)stk->end - RED_ZONE_SIZE
      - (uintptr_t)stk->data >= LIMIT_OFFSET
           && "Stack size must be greater than LIMIT_OFFSET");
    record_sp_limit(stk->data + LIMIT_OFFSET + RED_ZONE_SIZE);
}

inline rust_task* rust_try_get_current_task() {
    uintptr_t sp_limit = get_sp_limit();

    // FIXME (#1226) - Because of a hack in upcall_call_shim_on_c_stack this
    // value is sometimes inconveniently set to 0, so we can't use this
    // method of retreiving the task pointer and need to fall back to TLS.
    if (sp_limit == 0)
        return rust_sched_loop::try_get_task_tls();

    // The stack pointer boundary is stored in a quickly-accessible location
    // in the TCB. From that we can calculate the address of the stack segment
    // structure it belongs to, and in that structure is a pointer to the task
    // that owns it.
    uintptr_t seg_addr =
        sp_limit - RED_ZONE_SIZE - LIMIT_OFFSET - sizeof(stk_seg);
    stk_seg *stk = (stk_seg*) seg_addr;

    // Make sure we've calculated the right address
    ::check_stack_canary(stk);
    assert(stk->task != NULL && "task pointer not in stack structure");
    return stk->task;
}

inline rust_task* rust_get_current_task() {
    rust_task* task = rust_try_get_current_task();
    assert(task != NULL && "no current task");
    return task;
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

#endif /* RUST_TASK_H */
