// ignore-windows: Concurrency on Windows is not supported yet.
// We are making scheduler assumptions here.
// compile-flags: -Zmiri-strict-provenance -Zmiri-preemption-rate=0

use std::sync::{Arc, Condvar, Mutex};
use std::thread;

fn check_conditional_variables_notify_all() {
    let pair = Arc::new(((Mutex::new(())), Condvar::new()));

    // Spawn threads and block them on the conditional variable.
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let pair2 = pair.clone();
            thread::spawn(move || {
                let (lock, cvar) = &*pair2;
                let guard = lock.lock().unwrap();
                // Block waiting on the conditional variable.
                let _ = cvar.wait(guard).unwrap();
            })
        })
        .inspect(|_| {
            // Ensure the other threads all run and block on the `wait`.
            thread::yield_now();
            thread::yield_now();
        })
        .collect();

    let (_, cvar) = &*pair;
    // Unblock all threads.
    cvar.notify_all();

    for handle in handles {
        handle.join().unwrap();
    }
}

fn main() {
    check_conditional_variables_notify_all();
}
