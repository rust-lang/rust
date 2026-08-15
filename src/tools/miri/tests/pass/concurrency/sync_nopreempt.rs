// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::sync::{Arc, Condvar, Mutex, RwLock};
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
                let _guard = cvar.wait(guard).unwrap();
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

fn check_rwlock_unlock_bug1() {
    // There was a bug where when un-read-locking an rwlock that still has other
    // readers waiting, we'd accidentally also let a writer in.
    // That caused an ICE.
    let l = Arc::new(RwLock::new(0));

    let r1 = l.read().unwrap();
    let r2 = l.read().unwrap();

    // Make a waiting writer.
    let l2 = l.clone();
    let t = thread::spawn(move || {
        let mut w = l2.write().unwrap();
        *w += 1;
    });
    thread::yield_now();

    drop(r1);
    assert_eq!(*r2, 0);
    thread::yield_now();
    thread::yield_now();
    thread::yield_now();
    assert_eq!(*r2, 0);
    drop(r2);
    t.join().unwrap();
}

fn check_rwlock_unlock_bug2() {
    // There was a bug where when un-read-locking an rwlock by letting the last reader leaver,
    // we'd forget to wake up a writer.
    // That meant the writer thread could never run again.
    let l = Arc::new(RwLock::new(0));

    let r = l.read().unwrap();

    // Make a waiting writer.
    let l2 = l.clone();
    let h = thread::spawn(move || {
        let _w = l2.write().unwrap();
    });
    thread::yield_now();

    drop(r);
    h.join().unwrap();
}

fn main() {
    check_conditional_variables_notify_all();
    check_rwlock_unlock_bug1();
    check_rwlock_unlock_bug2();
}
