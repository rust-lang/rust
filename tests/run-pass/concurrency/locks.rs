// ignore-windows: Concurrency on Windows is not supported yet.

use std::sync::{Arc, Mutex, RwLock};
use std::thread;

fn check_mutex() {
    let data = Arc::new(Mutex::new(0));
    let mut threads = Vec::new();

    for _ in 0..3 {
        let data = Arc::clone(&data);
        let thread = thread::spawn(move || {
            let mut data = data.lock().unwrap();
            thread::yield_now();
            *data += 1;
        });
        threads.push(thread);
    }

    for thread in threads {
        thread.join().unwrap();
    }

    assert!(data.try_lock().is_ok());

    let data = Arc::try_unwrap(data).unwrap().into_inner().unwrap();
    assert_eq!(data, 3);
}

fn check_rwlock_write() {
    let data = Arc::new(RwLock::new(0));
    let mut threads = Vec::new();

    for _ in 0..3 {
        let data = Arc::clone(&data);
        let thread = thread::spawn(move || {
            let mut data = data.write().unwrap();
            thread::yield_now();
            *data += 1;
        });
        threads.push(thread);
    }

    for thread in threads {
        thread.join().unwrap();
    }

    assert!(data.try_write().is_ok());

    let data = Arc::try_unwrap(data).unwrap().into_inner().unwrap();
    assert_eq!(data, 3);
}

fn check_rwlock_read_no_deadlock() {
    let l1 = Arc::new(RwLock::new(0));
    let l2 = Arc::new(RwLock::new(0));

    let l1_copy = Arc::clone(&l1);
    let l2_copy = Arc::clone(&l2);
    let _guard1 = l1.read().unwrap();
    let handle = thread::spawn(move || {
        let _guard2 = l2_copy.read().unwrap();
        thread::yield_now();
        let _guard1 = l1_copy.read().unwrap();
    });
    thread::yield_now();
    let _guard2 = l2.read().unwrap();
    handle.join().unwrap();
}

fn main() {
    check_mutex();
    check_rwlock_write();
    check_rwlock_read_no_deadlock();
}
