//@ignore-target-windows: Concurrency on Windows is not supported yet.
//@compile-flags: -Zmiri-preemption-rate=0
use std::thread;

#[derive(Copy, Clone)]
struct MakeSend(*const i32);
unsafe impl Send for MakeSend {}

fn main() {
    race(0);
}

// Using an argument for the ptr to point to, since those do not get StorageDead.
fn race(local: i32) {
    let ptr = MakeSend(&local as *const i32);
    thread::spawn(move || {
        let ptr = ptr;
        let _val = unsafe { *ptr.0 };
    });
    // Make the other thread go first so that it does not UAF.
    thread::yield_now();
    // Deallocating the local (when `main` returns)
    // races with the read in the other thread.
    // Make sure the error points at this function's end, not just the call site.
} //~ERROR: Data race detected between Deallocate on thread `main` and Read on thread `<unnamed>`
