// -*- c++ -*-
#ifndef RUST_LOG_H
#define RUST_LOG_H

const uint32_t log_err = 0;
const uint32_t log_note = 1;

#define LOG(task, field, ...)                                   \
    DLOG_LVL(log_note, task, task->sched, field, __VA_ARGS__)
#define LOG_ERR(task, field, ...)                               \
    DLOG_LVL(log_err, task, task->sched, field, __VA_ARGS__)
#define DLOG(sched, field, ...)                                   \
    DLOG_LVL(log_note, NULL, sched, field, __VA_ARGS__)
#define DLOG_ERR(sched, field, ...)                               \
    DLOG_LVL(log_err, NULL, sched, field, __VA_ARGS__)
#define LOGPTR(sched, msg, ptrval)                                \
    DLOG_LVL(log_note, NULL, sched, mem, "%s 0x%" PRIxPTR, msg, ptrval)
#define DLOG_LVL(lvl, task, sched, field, ...)                    \
    do {                                                        \
        rust_scheduler* _d_ = sched;                            \
        if (log_rt_##field >= lvl && _d_->log_lvl >= lvl) {     \
            _d_->log(task, lvl, __VA_ARGS__);                   \
        }                                                       \
    } while (0)

#define KLOG(k, field, ...) \
    KLOG_LVL(k, field, log_note, __VA_ARGS__)
#define KLOG_LVL(k, field, lvl, ...)                          \
    do {                                                      \
        if (log_rt_##field >= lvl) {                          \
            (k)->log(lvl, __VA_ARGS__);                       \
        }                                                     \
    } while (0)

struct rust_scheduler;
struct rust_task;

class rust_log {

public:
    rust_log(rust_srv *srv, rust_scheduler *sched);
    virtual ~rust_log();

    enum ansi_color {
        WHITE,
        RED,
        LIGHTRED,
        GREEN,
        LIGHTGREEN,
        YELLOW,
        LIGHTYELLOW,
        BLUE,
        LIGHTBLUE,
        MAGENTA,
        LIGHTMAGENTA,
        TEAL,
        LIGHTTEAL
    };

    void trace_ln(rust_task *task, uint32_t level, char *message);
    void trace_ln(uint32_t thread_id, char *prefix, char *message);
    bool is_tracing(uint32_t type_bits);

private:
    rust_srv *_srv;
    rust_scheduler *_sched;
    bool _use_labels;
    bool _use_colors;
    void trace_ln(rust_task *task, char *message);
};

void update_log_settings(void* crate_map, char* settings);

extern size_t log_rt_mem;
extern size_t log_rt_comm;
extern size_t log_rt_task;
extern size_t log_rt_dom;
extern size_t log_rt_trace;
extern size_t log_rt_cache;
extern size_t log_rt_upcall;
extern size_t log_rt_timer;
extern size_t log_rt_gc;
extern size_t log_rt_stdlib;
extern size_t log_rt_kern;
extern size_t log_rt_backtrace;

#endif /* RUST_LOG_H */
