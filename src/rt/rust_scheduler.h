#ifndef RUST_SCHEDULER_H
#define RUST_SCHEDULER_H

struct rust_scheduler;

class
rust_crate_cache
{
public:
    type_desc *get_type_desc(size_t size,
                             size_t align,
                             size_t n_descs,
                             type_desc const **descs);

private:

    type_desc *type_descs;

public:

    rust_scheduler *sched;
    size_t idx;

    rust_crate_cache(rust_scheduler *sched);
    ~rust_crate_cache();
    void flush();
};

struct rust_scheduler : public kernel_owned<rust_scheduler>,
                        rust_thread
{
    RUST_REFCOUNTED(rust_scheduler)

    // Fields known to the compiler:
    uintptr_t interrupt_flag;

    // Fields known only by the runtime:
    rust_log _log;
    uint32_t log_lvl;
    rust_srv *srv;
    const char *const name;

    rust_task_list newborn_tasks;
    rust_task_list running_tasks;
    rust_task_list blocked_tasks;
    rust_task_list dead_tasks;

    rust_crate_cache cache;

    randctx rctx;

    rust_kernel *kernel;
    int32_t list_index;

    const int id;

    lock_and_signal lock;
    size_t min_stack_size;

#ifndef __WIN32__
    pthread_attr_t attr;
#endif

    rust_env *env;

    // Only a pointer to 'name' is kept, so it must live as long as this
    // domain.
    rust_scheduler(rust_kernel *kernel, rust_srv *srv, int id);
    ~rust_scheduler();
    void activate(rust_task *task);
    void log(rust_task *task, uint32_t level, char const *fmt, ...);
    rust_log & get_log();
    void fail();

    rust_crate_cache *get_cache();
    size_t number_of_live_tasks();

    void reap_dead_tasks(int id);
    rust_task *schedule_task(int id);

    void start_main_loop();

    void log_state();

    void kill_all_tasks();

    rust_task *create_task(rust_task *spawner, const char *name);

    virtual void run();

#ifdef __WIN32__
    inline void win32_require(LPCTSTR fn, BOOL ok) {
        kernel->win32_require(fn, ok);
    }
#endif
};

inline rust_log &
rust_scheduler::get_log() {
    return _log;
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

#endif /* RUST_SCHEDULER_H */
