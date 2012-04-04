
#include "rust_sched_launcher.h"
#include "rust_scheduler.h"

const size_t SCHED_STACK_SIZE = 1024*100;

rust_sched_launcher::rust_sched_launcher(rust_scheduler *sched, int id)
    : kernel(sched->kernel),
      sched_loop(sched, id),
      driver(&sched_loop) {
}

rust_thread_sched_launcher::rust_thread_sched_launcher(rust_scheduler *sched,
                                                       int id)
    : rust_sched_launcher(sched, id),
      rust_thread(SCHED_STACK_SIZE) {
}

rust_manual_sched_launcher::rust_manual_sched_launcher(rust_scheduler *sched,
                                                       int id)
    : rust_sched_launcher(sched, id) {
}

rust_sched_launcher *
rust_thread_sched_launcher_factory::create(rust_scheduler *sched, int id) {
    return new(sched->kernel, "rust_thread_sched_launcher")
        rust_thread_sched_launcher(sched, id);
}

rust_sched_launcher *
rust_manual_sched_launcher_factory::create(rust_scheduler *sched, int id) {
    assert(launcher == NULL && "I can only track one sched_launcher");
    launcher = new(sched->kernel, "rust_manual_sched_launcher")
        rust_manual_sched_launcher(sched, id);
    return launcher;
}
