#ifndef RUST_SCHED_LAUNCHER_H
#define RUST_SCHED_LAUNCHER_H

#include "rust_internal.h"
#include "sync/rust_thread.h"

#ifndef _WIN32
#include <pthread.h>
#else
#include <windows.h>
#endif

class rust_sched_launcher
  : public kernel_owned<rust_sched_launcher>,
    public rust_thread {
public:
    rust_kernel *kernel;

private:
    rust_task_thread thread;

public:
    rust_sched_launcher(rust_scheduler *sched, rust_srv *srv, int id);

    virtual void run();
    rust_task_thread *get_loop() { return &thread; }
};

#endif // RUST_SCHED_LAUNCHER_H
