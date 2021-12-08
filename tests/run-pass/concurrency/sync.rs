// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-disable-isolation -Zmiri-check-number-validity

use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Barrier, Condvar, Mutex, Once, RwLock};
use std::thread;
use std::time::{Duration, Instant};

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
fn check_conditional_variables_notify_one() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = pair.clone();

    // Spawn a new thread.
    thread::spawn(move || {
        thread::yield_now();
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        // We notify the condvar that the value has changed.
        cvar.notify_one();
    });

    // Wait for the thread to fully start up.
    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cvar.wait(started).unwrap();
    }
}

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

/// Test that waiting on a conditional variable with a timeout does not
/// deadlock.
fn check_conditional_variables_timed_wait_timeout() {
    let lock = Mutex::new(());
    let cvar = Condvar::new();
    let guard = lock.lock().unwrap();
    let now = Instant::now();
    let (_guard, timeout) = cvar.wait_timeout(guard, Duration::from_millis(100)).unwrap();
    assert!(timeout.timed_out());
    let elapsed_time = now.elapsed().as_millis();
    assert!(100 <= elapsed_time && elapsed_time <= 500);
}

/// Test that signaling a conditional variable when waiting with a timeout works
/// as expected.
fn check_conditional_variables_timed_wait_notimeout() {
    let pair = Arc::new((Mutex::new(()), Condvar::new()));
    let pair2 = pair.clone();

    let (lock, cvar) = &*pair;
    let guard = lock.lock().unwrap();

    let handle = thread::spawn(move || {
        let (_lock, cvar) = &*pair2;
        cvar.notify_one();
    });

    let (_guard, timeout) = cvar.wait_timeout(guard, Duration::from_millis(500)).unwrap();
    assert!(!timeout.timed_out());
    handle.join().unwrap();
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
    while i < 1000 {
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
                assert_eq!(val, 5040);
            })
        })
        .collect();
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
    thread::spawn(move || {
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

fn park_timeout() {
    let start = Instant::now();

    thread::park_timeout(Duration::from_millis(200));
    // Normally, waiting in park/park_timeout may spuriously wake up early, but we
    // know Miri's timed synchronization primitives do not do that.

    assert!((200..1000).contains(&start.elapsed().as_millis()));
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
    // Normally, waiting in park/park_timeout may spuriously wake up early, but we
    // know Miri's timed synchronization primitives do not do that.

    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn check_condvar() {
    let _ = std::sync::Condvar::new();
}

fn main() {
    check_barriers();
    check_conditional_variables_notify_one();
    check_conditional_variables_notify_all();
    check_conditional_variables_timed_wait_timeout();
    check_conditional_variables_timed_wait_notimeout();
    check_mutex();
    check_rwlock_write();
    check_rwlock_read_no_deadlock();
    simple_send();
    multiple_send();
    send_on_sync();
    check_once();
    check_rwlock_unlock_bug1();
    check_rwlock_unlock_bug2();
    park_timeout();
    park_unpark();
    check_condvar();
}
