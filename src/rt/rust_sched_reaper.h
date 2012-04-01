#ifndef RUST_SCHED_REAPER_H
#define RUST_SCHED_REAPER_H

#include "sync/rust_thread.h"

class rust_kernel;

/* Responsible for joining with rust_schedulers */
class rust_sched_reaper : public rust_thread {
private:
    rust_kernel *kernel;
public:
    rust_sched_reaper(rust_kernel *kernel);
    virtual void run();
};

#endif /* RUST_SCHED_REAPER_H */
