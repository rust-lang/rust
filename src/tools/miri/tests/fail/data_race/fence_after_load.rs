// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use std::thread;
use std::time::Duration;

fn main() {
    static mut V: u32 = 0;
    let a = Arc::new(AtomicUsize::default());
    let b = a.clone();
    thread::spawn(move || {
        unsafe { V = 1 }
        b.store(1, Ordering::SeqCst);
    });
    thread::sleep(Duration::from_millis(100));
    fence(Ordering::SeqCst);
    // Imagine the other thread's actions happening here.
    assert_eq!(a.load(Ordering::Relaxed), 1);
    // The fence is useless, since it did not happen-after the `store` in the other thread.
    // Hence this is a data race.
    // Also see https://github.com/rust-lang/miri/issues/2192.
    unsafe { V = 2 } //~ERROR: Data race detected between (1) non-atomic write on thread `unnamed-1` and (2) non-atomic write on thread `main`
}
