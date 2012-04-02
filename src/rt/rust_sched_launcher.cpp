#include "rust_sched_launcher.h"
#include "rust_scheduler.h"

const size_t SCHED_STACK_SIZE = 1024*100;

rust_sched_launcher::rust_sched_launcher(rust_scheduler *sched,
                                         rust_srv *srv, int id)
    : kernel(sched->kernel),
      sched_loop(sched, srv, id),
      driver(&sched_loop) {
}

rust_thread_sched_launcher::rust_thread_sched_launcher(rust_scheduler *sched,
                                                       rust_srv *srv, int id)
    : rust_sched_launcher(sched, srv, id),
      rust_thread(SCHED_STACK_SIZE) {
}

rust_manual_sched_launcher::rust_manual_sched_launcher(rust_scheduler *sched,
                                                       rust_srv *srv, int id)
    : rust_sched_launcher(sched, srv, id) {
}

rust_sched_launcher *
rust_thread_sched_launcher_factory::create(rust_scheduler *sched, int id) {
    rust_srv *srv = sched->srv->clone();
    return new(sched->kernel, "rust_thread_sched_launcher")
        rust_thread_sched_launcher(sched, srv, id);
}
