#ifndef RUST_DOM_H
#define RUST_DOM_H

struct rust_dom;

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

    rust_dom *dom;
    size_t idx;

    rust_crate_cache(rust_dom *dom);
    ~rust_crate_cache();
    void flush();
};

struct rust_dom : public kernel_owned<rust_dom>, rc_base<rust_dom>
{
    // Fields known to the compiler:
    uintptr_t interrupt_flag;

    // Fields known only by the runtime:
    rust_log _log;
    uint32_t log_lvl;
    rust_srv *srv;
    memory_region local_region;
    memory_region synchronized_region;
    const char *const name;

    rust_task_list newborn_tasks;
    rust_task_list running_tasks;
    rust_task_list blocked_tasks;
    rust_task_list dead_tasks;

    rust_crate_cache cache;

    randctx rctx;
    rust_task *root_task;
    rust_task *curr_task;
    int rval;

    rust_kernel *kernel;
    int32_t list_index;

    hash_map<rust_task *, rust_proxy<rust_task> *> _task_proxies;
    hash_map<rust_port *, rust_proxy<rust_port> *> _port_proxies;

    // Incoming messages from other domains.
    rust_message_queue *message_queue;

#ifndef __WIN32__
    pthread_attr_t attr;
#endif

    // Only a pointer to 'name' is kept, so it must live as long as this
    // domain.
    rust_dom(rust_kernel *kernel,
             rust_message_queue *message_queue, rust_srv *srv,
             const char *name);
    ~rust_dom();
    void activate(rust_task *task);
    void log(rust_task *task, uint32_t level, char const *fmt, ...);
    rust_log & get_log();
    void fail();
    void *malloc(size_t size);
    void *malloc(size_t size, memory_region::memory_region_type type);
    void *calloc(size_t size);
    void *calloc(size_t size, memory_region::memory_region_type type);
    void *realloc(void *mem, size_t size);
    void *realloc(void *mem, size_t size,
        memory_region::memory_region_type type);
    void free(void *mem);
    void free(void *mem, memory_region::memory_region_type type);

    void drain_incoming_message_queue(bool process);

#ifdef __WIN32__
    void win32_require(LPCTSTR fn, BOOL ok);
#endif

    rust_crate_cache *get_cache();
    size_t number_of_live_tasks();

    void reap_dead_tasks();
    rust_task *schedule_task();

    int start_main_loop();

    void log_state();

    rust_task *create_task(rust_task *spawner, const char *name);
};

inline rust_log &
rust_dom::get_log() {
    return _log;
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

#endif /* RUST_DOM_H */
