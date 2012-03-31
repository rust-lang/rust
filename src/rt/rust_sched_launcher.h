#ifndef RUST_SCHED_LAUNCHER_H
#define RUST_SCHED_LAUNCHER_H

#include "rust_internal.h"
#include "sync/rust_thread.h"
#include "rust_sched_driver.h"

class rust_sched_launcher
  : public kernel_owned<rust_sched_launcher>,
    public rust_thread {
public:
    rust_kernel *kernel;

private:
    rust_sched_loop sched_loop;
    rust_sched_driver driver;

public:
    rust_sched_launcher(rust_scheduler *sched, rust_srv *srv, int id);

    virtual void run();
    rust_sched_loop *get_loop() { return &sched_loop; }
};

#endif // RUST_SCHED_LAUNCHER_H
