/*
 *
 */

#ifndef RUST_TASK_H
#define RUST_TASK_H
struct
rust_task : public rust_proxy_delegate<rust_task>,
            public dom_owned<rust_task>,
            public rust_cond
{
    // Fields known to the compiler.
    stk_seg *stk;
    uintptr_t runtime_sp;      // Runtime sp while task running.
    uintptr_t rust_sp;         // Saved sp when not running.
    gc_alloc *gc_alloc_chain;  // Linked list of GC allocations.
    rust_dom *dom;
    rust_crate_cache *cache;

    // Fields known only to the runtime.
    ptr_vec<rust_task> *state;
    rust_cond *cond;
    rust_task *supervisor;     // Parent-link for failure propagation.
    size_t idx;
    size_t gc_alloc_thresh;
    size_t gc_alloc_accum;

    // Wait queue for tasks waiting for this task.
    rust_wait_queue waiting_tasks;

    // Rendezvous pointer for receiving data when blocked on a port. If we're
    // trying to read data and no data is available on any incoming channel,
    // we block on the port, and yield control to the scheduler. Since, we
    // were not able to read anJything, we remember the location where the
    // result should go in the rendezvous_ptr, and let the sender write to
    // that location before waking us up.
    uintptr_t* rendezvous_ptr;

    rust_alarm alarm;

    rust_task(rust_dom *dom,
              rust_task *spawner);
    ~rust_task();

    void start(uintptr_t exit_task_glue,
               uintptr_t spawnee_fn,
               uintptr_t args,
               size_t callsz);
    void grow(size_t n_frame_bytes);
    bool running();
    bool blocked();
    bool blocked_on(rust_cond *cond);
    bool dead();

    void link_gc(gc_alloc *gcm);
    void unlink_gc(gc_alloc *gcm);
    void *malloc(size_t sz, type_desc *td=0);
    void *realloc(void *data, size_t sz, bool gc_mem=false);
    void free(void *p, bool gc_mem=false);

    const char *state_str();
    void transition(ptr_vec<rust_task> *svec, ptr_vec<rust_task> *dvec);

    void block(rust_cond *on);
    void wakeup(rust_cond *from);
    void die();
    void unblock();

    void check_active() { I(dom, dom->curr_task == this); }
    void check_suspended() { I(dom, dom->curr_task != this); }

    void log(uint32_t type_bits, char const *fmt, ...);

    // Swap in some glue code to run when we have returned to the
    // task's context (assuming we're the active task).
    void run_after_return(size_t nargs, uintptr_t glue);

    // Swap in some glue code to run when we're next activated
    // (assuming we're the suspended task).
    void run_on_resume(uintptr_t glue);

    // Save callee-saved registers and return to the main loop.
    void yield(size_t nargs);

    // Fail this task (assuming caller-on-stack is different task).
    void kill();

    // Fail self, assuming caller-on-stack is this task.
    void fail(size_t nargs);

    // Run the gc glue on the task stack.
    void gc(size_t nargs);

    // Disconnect from our supervisor.
    void unsupervise();

    // Notify tasks waiting for us that we are about to die.
    void notify_waiting_tasks();

    uintptr_t get_fp();
    uintptr_t get_previous_fp(uintptr_t fp);
    frame_glue_fns *get_frame_glue_fns(uintptr_t fp);
    rust_crate_cache * get_crate_cache(rust_crate const *curr_crate);
};


#endif /* RUST_TASK_H */
