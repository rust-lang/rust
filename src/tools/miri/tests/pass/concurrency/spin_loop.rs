use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

static FLAG: AtomicUsize = AtomicUsize::new(0);

fn spin() {
    let j = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // We do *not* yield, and yet this should terminate eventually.
        }
    });
    thread::yield_now(); // schedule the other thread
    FLAG.store(1, Ordering::Release);
    j.join().unwrap();
}

fn two_player_ping_pong() {
    static FLAG: AtomicUsize = AtomicUsize::new(0);

    let waiter1 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // We do *not* yield, and yet this should terminate eventually.
        }
    });
    let waiter2 = thread::spawn(|| {
        while FLAG.load(Ordering::Acquire) == 0 {
            // We do *not* yield, and yet this should terminate eventually.
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

fn main() {
    spin();
    two_player_ping_pong();
}
