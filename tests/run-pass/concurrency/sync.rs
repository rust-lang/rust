// ignore-windows: Concurrency on Windows is not supported yet.

use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Barrier, Condvar, Mutex, Once, RwLock};
use std::thread;

// Check if Rust barriers are working.

/// This test is taken from the Rust documentation.
fn check_barriers() {
    let mut handles = Vec::with_capacity(10);
    let barrier = Arc::new(Barrier::new(10));
    for _ in 0..10 {
        let c = barrier.clone();
        // The same messages will be printed together.
        // You will NOT see any interleaving.
        handles.push(thread::spawn(move || {
            println!("before wait");
            c.wait();
            println!("after wait");
        }));
    }
    // Wait for other threads to finish.
    for handle in handles {
        handle.join().unwrap();
    }
}

// Check if Rust conditional variables are working.

/// The test taken from the Rust documentation.
fn check_conditional_variables() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = pair.clone();

    // Inside of our lock, spawn a new thread, and then wait for it to start.
    thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        // We notify the condvar that the value has changed.
        cvar.notify_one();
    });

    // Wait for the thread to start up.
    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cvar.wait(started).unwrap();
    }
}

// Check if locks are working.

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

// Check if channels are working.

/// The test taken from the Rust documentation.
fn simple_send() {
    let (tx, rx) = channel();
    thread::spawn(move || {
        tx.send(10).unwrap();
    });
    assert_eq!(rx.recv().unwrap(), 10);
}

/// The test taken from the Rust documentation.
fn multiple_send() {
    let (tx, rx) = channel();
    for i in 0..10 {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(i).unwrap();
        });
    }

    let mut sum = 0;
    for _ in 0..10 {
        let j = rx.recv().unwrap();
        assert!(0 <= j && j < 10);
        sum += j;
    }
    assert_eq!(sum, 45);
}

/// The test taken from the Rust documentation.
fn send_on_sync() {
    let (sender, receiver) = sync_channel(1);

    // this returns immediately
    sender.send(1).unwrap();

    thread::spawn(move || {
        // this will block until the previous message has been received
        sender.send(2).unwrap();
    });

    assert_eq!(receiver.recv().unwrap(), 1);
    assert_eq!(receiver.recv().unwrap(), 2);
}

// Check if Rust once statics are working.

static mut VAL: usize = 0;
static INIT: Once = Once::new();

fn get_cached_val() -> usize {
    unsafe {
        INIT.call_once(|| {
            VAL = expensive_computation();
        });
        VAL
    }
}

fn expensive_computation() -> usize {
    let mut i = 1;
    let mut c = 1;
    while i < 10000 {
        i *= c;
        c += 1;
    }
    i
}

/// The test taken from the Rust documentation.
fn check_once() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                thread::yield_now();
                let val = get_cached_val();
                assert_eq!(val, 40320);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

fn main() {
    check_barriers();
    check_conditional_variables();
    check_mutex();
    check_rwlock_write();
    check_rwlock_read_no_deadlock();
    simple_send();
    multiple_send();
    send_on_sync();
    check_once();
}
