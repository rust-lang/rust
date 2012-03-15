// -*- c++ -*-
#ifndef RUST_KERNEL_H
#define RUST_KERNEL_H

#include <map>
#include <vector>
#include "memory_region.h"
#include "rust_log.h"

struct rust_task_thread;
class rust_scheduler;

typedef std::map<rust_sched_id, rust_scheduler*> sched_map;

/**
 * A global object shared by all thread domains. Most of the data structures
 * in this class are synchronized since they are accessed from multiple
 * threads.
 */
class rust_kernel {
    memory_region _region;
    rust_log _log;

public:
    rust_srv *srv;
private:
    // Protects max_task_id and task_table
    lock_and_signal task_lock;
    // The next task id
    rust_task_id max_task_id;
    hash_map<rust_task_id, rust_task *> task_table;

    // Protects max_port_id and port_table
    lock_and_signal port_lock;
    // The next port id
    rust_task_id max_port_id;
    hash_map<rust_port_id, rust_port *> port_table;

    lock_and_signal rval_lock;
    int rval;

    // Protects max_sched_id and sched_table, join_list
    lock_and_signal sched_lock;
    // The next scheduler id
    rust_sched_id max_sched_id;
    // A map from scheduler ids to schedulers. When this is empty
    // the kernel terminates
    sched_map sched_table;
    // A list of scheduler ids that are ready to exit
    std::vector<rust_sched_id> join_list;

public:

    struct rust_env *env;

    rust_kernel(rust_srv *srv);

    void log(uint32_t level, char const *fmt, ...);
    void fatal(char const *fmt, ...);

    void *malloc(size_t size, const char *tag);
    void *realloc(void *mem, size_t size);
    void free(void *mem);
    memory_region *region() { return &_region; }

    void fail();

    rust_sched_id create_scheduler(size_t num_threads);
    rust_scheduler* get_scheduler_by_id(rust_sched_id id);
    // Called by a scheduler to indicate that it is terminating
    void release_scheduler_id(rust_sched_id id);
    int wait_for_schedulers();

#ifdef __WIN32__
    void win32_require(LPCTSTR fn, BOOL ok);
#endif

    void register_task(rust_task *task);
    rust_task *get_task_by_id(rust_task_id id);
    void release_task_id(rust_task_id tid);

    rust_port_id register_port(rust_port *port);
    rust_port *get_port_by_id(rust_port_id id);
    void release_port_id(rust_port_id tid);

    void set_exit_status(int code);
};

#endif /* RUST_KERNEL_H */
