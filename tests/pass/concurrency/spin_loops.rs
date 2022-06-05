// ignore-windows: Concurrency on Windows is not supported yet.

use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

static FLAG: AtomicUsize = AtomicUsize::new(0);

// When a thread yields, Miri's scheduler used to pick the thread with the lowest ID
// that can run. IDs are assigned in thread creation order.
// This means we could make 2 threads infinitely ping-pong with each other while
// really there is a 3rd thread that we should schedule to make progress.

fn main() {
    let waiter1 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // spin and wait
            thread::yield_now();
        }
    });
    let waiter2 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // spin and wait
            thread::yield_now();
        }
    });
    let progress = thread::spawn(|| {
        FLAG.store(1, Ordering::Release);
    });
    // The first `join` blocks the main thread and thus takes it out of the equation.
    waiter1.join().unwrap();
    waiter2.join().unwrap();
    progress.join().unwrap();
}
