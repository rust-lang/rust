
#include "rust_kernel.h"
#include "rust_sched_reaper.h"

// NB: We're using a very small stack here
const size_t STACK_SIZE = 1024*20;

rust_sched_reaper::rust_sched_reaper(rust_kernel *kernel)
    : rust_thread(STACK_SIZE), kernel(kernel) {
}

void
rust_sched_reaper::run() {
    kernel->wait_for_schedulers();
}
