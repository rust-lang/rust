// -*- c++ -*-
#ifndef RUST_LOG_H
#define RUST_LOG_H

#include "rust_globals.h"

const uint32_t log_err = 1;
const uint32_t log_warn = 2;
const uint32_t log_info = 3;
const uint32_t log_debug = 4;

#define LOG(task, field, ...)                                   \
    DLOG_LVL(log_debug, task, task->sched_loop, field, __VA_ARGS__)
#define LOG_ERR(task, field, ...)                               \
    DLOG_LVL(log_err, task, task->sched_loop, field, __VA_ARGS__)
#define DLOG(sched_loop, field, ...)                                   \
    DLOG_LVL(log_debug, NULL, sched_loop, field, __VA_ARGS__)
#define DLOG_ERR(sched_loop, field, ...)                               \
    DLOG_LVL(log_err, NULL, sched_loop, field, __VA_ARGS__)
#define LOGPTR(sched_loop, msg, ptrval)                                \
    DLOG_LVL(log_debug, NULL, sched_loop, mem, "%s 0x%" PRIxPTR, msg, ptrval)
#define DLOG_LVL(lvl, task, sched_loop, field, ...)                    \
    do {                                                        \
        rust_sched_loop* _d_ = sched_loop;                      \
        if (log_rt_##field >= lvl && _d_->log_lvl >= lvl) {     \
            _d_->get_log().log(task, lvl, __VA_ARGS__);         \
        }                                                       \
    } while (0)

#define KLOG(k, field, ...) \
    KLOG_LVL(k, field, log_debug, __VA_ARGS__)
#define KLOG_LVL(k, field, lvl, ...)                          \
    do {                                                      \
        if (log_rt_##field >= lvl) {                          \
            (k)->log(lvl, __VA_ARGS__);                       \
        }                                                     \
    } while (0)

struct rust_sched_loop;
struct rust_task;

class rust_log {

public:
    rust_log(rust_sched_loop *sched_loop);
    virtual ~rust_log();

    void log(rust_task* task, uint32_t level, char const *fmt, ...);
    void trace_ln(rust_task *task, uint32_t level, char *message);
    void trace_ln(char *prefix, char *message);
    bool is_tracing(uint32_t type_bits);

private:
    rust_sched_loop *_sched_loop;
    bool _use_labels;
    void trace_ln(rust_task *task, char *message);
};

void update_log_settings(void* crate_map, char* settings);

extern uint32_t log_rt_mem;
extern uint32_t log_rt_box;
extern uint32_t log_rt_comm;
extern uint32_t log_rt_task;
extern uint32_t log_rt_dom;
extern uint32_t log_rt_trace;
extern uint32_t log_rt_cache;
extern uint32_t log_rt_upcall;
extern uint32_t log_rt_timer;
extern uint32_t log_rt_gc;
extern uint32_t log_rt_stdlib;
extern uint32_t log_rt_kern;
extern uint32_t log_rt_backtrace;
extern uint32_t log_rt_callback;

#endif /* RUST_LOG_H */
