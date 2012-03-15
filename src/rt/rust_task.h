/*
 *
 */

#ifndef RUST_TASK_H
#define RUST_TASK_H

#include <map>

#include "util/array_list.h"

#include "context.h"
#include "rust_debug.h"
#include "rust_internal.h"
#include "rust_kernel.h"
#include "boxed_region.h"
#include "rust_stack.h"
#include "rust_port_selector.h"

struct rust_box;

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

// std::lib::task::task_notification
//
// since it's currently a unary tag, we only add the fields.
struct task_notification {
    rust_task_id id;
    task_result result; // task_result
};

struct
rust_task : public kernel_owned<rust_task>, rust_cond
{
    RUST_ATOMIC_REFCOUNT();

    rust_task_id id;
    bool notify_enabled;
    rust_port_id notify_port;

    context ctx;
    stk_seg *stk;
    uintptr_t runtime_sp;      // Runtime sp while task running.
    rust_scheduler *sched;
    rust_task_thread *thread;

    // Fields known only to the runtime.
    rust_kernel *kernel;
    const char *const name;
    int32_t list_index;

    // Rendezvous pointer for receiving data when blocked on a port. If we're
    // trying to read data and no data is available on any incoming channel,
    // we block on the port, and yield control to the scheduler. Since, we
    // were not able to read anything, we remember the location where the
    // result should go in the rendezvous_ptr, and let the sender write to
    // that location before waking us up.
    uintptr_t* rendezvous_ptr;

    memory_region local_region;
    boxed_region boxed;

    // Indicates that fail() has been called and we are cleaning up.
    // We use this to suppress the "killed" flag during calls to yield.
    bool unwinding;

    bool propagate_failure;

    uint32_t cc_counter;

    debug::task_debug_info debug;

    // The amount of stack we're using, excluding red zones
    size_t total_stack_sz;

private:

    // Protects state, cond, cond_name
    lock_and_signal state_lock;
    rust_task_list *state;
    rust_cond *cond;
    const char *cond_name;

    // Protects the killed flag
    lock_and_signal kill_lock;
    // Indicates that the task was killed and needs to unwind
    bool killed;
    // Indicates that we've called back into Rust from C
    bool reentered_rust_stack;

    // The stack used for running C code, borrowed from the scheduler thread
    stk_seg *c_stack;
    uintptr_t next_c_sp;
    uintptr_t next_rust_sp;

    rust_port_selector port_selector;

    lock_and_signal supervisor_lock;
    rust_task *supervisor;     // Parent-link for failure propagation.

    // Called when the atomic refcount reaches zero
    void delete_this();

    void new_stack(size_t sz);
    void del_stack();
    void free_stack(stk_seg *stk);
    size_t get_next_stack_size(size_t min, size_t current, size_t requested);

    void return_c_stack();

    void transition(rust_task_list *src, rust_task_list *dst,
                    rust_cond *cond, const char* cond_name);

    bool must_fail_from_being_killed_unlocked();

    friend void task_start_wrapper(spawn_args *a);
    friend void cleanup_task(cleanup_args *a);
    friend void reset_stack_limit_on_c_stack(reset_args *a);

public:

    // Only a pointer to 'name' is kept, so it must live as long as this task.
    rust_task(rust_task_thread *thread,
              rust_task_list *state,
              rust_task *spawner,
              const char *name,
              size_t init_stack_sz);

    void start(spawn_fn spawnee_fn,
               rust_opaque_box *env,
               void *args);
    void start();
    bool running();
    bool blocked();
    bool blocked_on(rust_cond *cond);
    bool dead();

    void *malloc(size_t sz, const char *tag, type_desc *td=0);
    void *realloc(void *data, size_t sz);
    void free(void *p);

    void set_state(rust_task_list *state,
                   rust_cond *cond, const char* cond_name);

    bool block(rust_cond *on, const char* name);
    void wakeup(rust_cond *from);
    void die();

    // Print a backtrace, if the "bt" logging option is on.
    void backtrace();

    // Yields control to the scheduler. Called from the Rust stack
    void yield(bool *killed);

    // Fail this task (assuming caller-on-stack is different task).
    void kill();

    // Indicates that we've been killed and now is an apropriate
    // time to fail as a result
    bool must_fail_from_being_killed();

    // Fail self, assuming caller-on-stack is this task.
    void fail();
    void conclude_failure();
    void fail_parent();

    // Disconnect from our supervisor.
    void unsupervise();

    frame_glue_fns *get_frame_glue_fns(uintptr_t fp);

    void *calloc(size_t size, const char *tag);

    // Use this function sparingly. Depending on the ref count is generally
    // not at all safe.
    intptr_t get_ref_count() const { return ref_count; }

    void notify(bool success);

    void *next_stack(size_t stk_sz, void *args_addr, size_t args_sz);
    void prev_stack();
    void record_stack_limit();
    void reset_stack_limit();
    bool on_rust_stack();
    void check_stack_canary();
    void delete_all_stacks();

    void config_notify(rust_port_id port);

    void call_on_c_stack(void *args, void *fn_ptr);
    void call_on_rust_stack(void *args, void *fn_ptr);
    bool have_c_stack() { return c_stack != NULL; }

    rust_port_selector *get_port_selector() { return &port_selector; }

    rust_task_list *get_state() { return state; }
    rust_cond *get_cond() { return cond; }
    const char *get_cond_name() { return cond_name; }
};

// This stuff is on the stack-switching fast path

// Get a rough approximation of the current stack pointer
extern "C" uintptr_t get_sp();

// This is the function that switches stacks by calling another function with
// a single void* argument while changing the stack pointer. It has a funny
// name because gdb doesn't normally like to backtrace through split stacks
// (thinks it indicates a bug), but has a special case to allow functions
// named __morestack to move the stack pointer around.
extern "C" void __morestack(void *args, void *fn_ptr, uintptr_t stack_ptr);

inline static uintptr_t
sanitize_next_sp(uintptr_t next_sp) {

    // Since I'm not precisely sure where the next stack pointer sits in
    // relation to where the context switch actually happened, nor in relation
    // to the amount of stack needed for calling __morestack I've added some
    // extra bytes here.

    // FIXME: On the rust stack this potentially puts is quite far into the
    // red zone. Might want to just allocate a new rust stack every time we
    // switch back to rust.
    const uintptr_t padding = 16;

    return align_down(next_sp - padding);
}

inline void
rust_task::call_on_c_stack(void *args, void *fn_ptr) {
    // Too expensive to check
    // I(thread, on_rust_stack());

    uintptr_t prev_rust_sp = next_rust_sp;
    next_rust_sp = get_sp();

    bool borrowed_a_c_stack = false;
    uintptr_t sp;
    if (c_stack == NULL) {
        c_stack = thread->borrow_c_stack();
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
}

inline void
rust_task::call_on_rust_stack(void *args, void *fn_ptr) {
    // Too expensive to check
    // I(thread, !on_rust_stack());
    I(thread, next_rust_sp);

    bool had_reentered_rust_stack = reentered_rust_stack;
    reentered_rust_stack = true;

    uintptr_t prev_c_sp = next_c_sp;
    next_c_sp = get_sp();

    uintptr_t sp = sanitize_next_sp(next_rust_sp);

    __morestack(args, fn_ptr, sp);

    next_c_sp = prev_c_sp;
    reentered_rust_stack = had_reentered_rust_stack;
}

inline void
rust_task::return_c_stack() {
    // Too expensive to check
    // I(thread, on_rust_stack());
    I(thread, c_stack != NULL);
    thread->return_c_stack(c_stack);
    c_stack = NULL;
    next_c_sp = 0;
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
