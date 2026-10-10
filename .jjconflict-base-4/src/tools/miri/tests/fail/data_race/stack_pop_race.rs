// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

use std::thread;

#[derive(Copy, Clone)]
struct MakeSend(*const i32);
unsafe impl Send for MakeSend {}

fn main() {
    race(0); //~ERROR: Data race detected between (1) non-atomic read on thread `unnamed-1` and (2) deallocation on thread `main`
}

// Using an argument for the ptr to point to, since those do not get StorageDead.
fn race(local: i32) {
    let ptr = MakeSend(&local as *const i32);
    thread::spawn(move || {
        let ptr = ptr; // avoid field capturing
        let _val = unsafe { *ptr.0 };
    });
    // Make the other thread go first so that it does not UAF.
    thread::yield_now();
    // Deallocating the local (when `main` returns)
    // races with the read in the other thread.
}
