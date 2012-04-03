
#include "rust_globals.h"
#include "rust_sched_driver.h"
#include "rust_sched_loop.h"

rust_sched_driver::rust_sched_driver(rust_sched_loop *sched_loop)
    : sched_loop(sched_loop),
      signalled(false) {

    assert(sched_loop != NULL);
    sched_loop->on_pump_loop(this);
}

/**
 * Starts the main scheduler loop which performs task scheduling for this
 * domain.
 *
 * Returns once no more tasks can be scheduled and all task ref_counts
 * drop to zero.
 */
void
rust_sched_driver::start_main_loop() {
    assert(sched_loop != NULL);

    rust_sched_loop_state state = sched_loop_state_keep_going;
    while (state != sched_loop_state_exit) {
        DLOG(sched_loop, dom, "pumping scheduler");
        state = sched_loop->run_single_turn();

        if (state == sched_loop_state_block) {
            scoped_lock with(lock);
            if (!signalled) {
                DLOG(sched_loop, dom, "blocking scheduler");
                lock.wait();
            }
            signalled = false;
        }
    }
}

void
rust_sched_driver::signal() {
    scoped_lock with(lock);
    signalled = true;
    lock.signal();
}
