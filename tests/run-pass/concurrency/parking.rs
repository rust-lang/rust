// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-disable-isolation

use std::thread;
use std::time::{Duration, Instant};

// Normally, waiting in park/park_timeout may spuriously wake up early, but we
// know Miri's timed synchronization primitives do not do that.

fn park_timeout() {
    let start = Instant::now();

    thread::park_timeout(Duration::from_millis(200));

    assert!((200..500).contains(&start.elapsed().as_millis()));
}

fn park_unpark() {
    let t1 = thread::current();
    let t2 = thread::spawn(move || {
        thread::park();
        thread::sleep(Duration::from_millis(200));
        t1.unpark();
    });

    let start = Instant::now();

    t2.thread().unpark();
    thread::park();

    assert!((200..500).contains(&start.elapsed().as_millis()));
}

fn main() {
    park_timeout();
    park_unpark();
}
