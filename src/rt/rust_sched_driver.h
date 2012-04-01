#ifndef RUST_SCHED_DRIVER_H
#define RUST_SCHED_DRIVER_H

#include "sync/lock_and_signal.h"
#include "rust_signal.h"

struct rust_sched_loop;

class rust_sched_driver : public rust_signal {
private:
    rust_sched_loop *sched_loop;
    lock_and_signal lock;
    bool signalled;

public:
    rust_sched_driver(rust_sched_loop *sched_loop);

    void start_main_loop();

    virtual void signal();
};

#endif /* RUST_SCHED_DRIVER_H */
