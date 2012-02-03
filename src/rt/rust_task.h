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
#include "rust_obstack.h"
#include "boxed_region.h"

// Corresponds to the rust chan (currently _chan) type.
struct chan_handle {
    rust_task_id task;
    rust_port_id port;
};

struct rust_box;

struct stk_seg {
    stk_seg *prev;
    stk_seg *next;
    uintptr_t end;
    unsigned int valgrind_id;
#ifndef _LP64
    uint32_t pad;
#endif

    uint8_t data[];
};

struct frame_glue_fns {
    uintptr_t mark_glue_off;
    uintptr_t drop_glue_off;
    uintptr_t reloc_glue_off;
};

// portions of the task structure that are accessible from the standard
// library. This struct must agree with the std::task::rust_task record.
struct rust_task_user {
    rust_task_id id;
    intptr_t notify_enabled;   // this is way more bits than necessary, but it
                               // simplifies the alignment.
    chan_handle notify_chan;
    uintptr_t rust_sp;         // Saved sp when not running.
};

// std::lib::task::task_result
typedef unsigned long task_result;
#define tr_success 0
#define tr_failure 1

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
    rust_task_user user;

    RUST_ATOMIC_REFCOUNT();

    context ctx;
    stk_seg *stk;
    uintptr_t runtime_sp;      // Runtime sp while task running.
    rust_scheduler *sched;
    rust_task_thread *thread;
    rust_crate_cache *cache;

    // Fields known only to the runtime.
    rust_kernel *kernel;
    const char *const name;
    rust_task_list *state;
    rust_cond *cond;
    const char *cond_name;
    rust_task *supervisor;     // Parent-link for failure propagation.
    int32_t list_index;

    rust_port_id next_port_id;

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

    // Indicates that the task was killed and needs to unwind
    bool killed;
    bool propagate_failure;

    lock_and_signal lock;

    hash_map<rust_port_id, rust_port *> port_table;

    rust_obstack dynastack;

    uint32_t cc_counter;

    debug::task_debug_info debug;

    // The amount of stack we're using, excluding red zones
    size_t total_stack_sz;

    // Only a pointer to 'name' is kept, so it must live as long as this task.
    rust_task(rust_task_thread *thread,
              rust_task_list *state,
              rust_task *spawner,
              const char *name,
              size_t init_stack_sz);

    ~rust_task();

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

    void transition(rust_task_list *src, rust_task_list *dst);

    void block(rust_cond *on, const char* name);
    void wakeup(rust_cond *from);
    void die();
    void unblock();

    // Print a backtrace, if the "bt" logging option is on.
    void backtrace();

    // Yields control to the scheduler. Called from the Rust stack
    void yield(bool *killed);

    // Fail this task (assuming caller-on-stack is different task).
    void kill();

    // Fail self, assuming caller-on-stack is this task.
    void fail();
    void conclude_failure();
    void fail_parent();

    // Disconnect from our supervisor.
    void unsupervise();

    frame_glue_fns *get_frame_glue_fns(uintptr_t fp);
    rust_crate_cache * get_crate_cache();

    void *calloc(size_t size, const char *tag);

    rust_port_id register_port(rust_port *port);
    void release_port(rust_port_id id);
    rust_port *get_port_by_id(rust_port_id id);

    // Use this function sparingly. Depending on the ref count is generally
    // not at all safe.
    intptr_t get_ref_count() const { return ref_count; }

    void notify(bool success);

    void *new_stack(size_t stk_sz, void *args_addr, size_t args_sz);
    void del_stack();
    void record_stack_limit();
    void reset_stack_limit();
    bool on_rust_stack();
    void check_stack_canary();
};

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
