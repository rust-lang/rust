// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

//! Cause a panic in one thread while another thread is unwinding. This checks
//! that separate threads have their own panicking state.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::{JoinHandle, spawn};

struct BlockOnDrop(Option<JoinHandle<()>>);

impl BlockOnDrop {
    fn new(handle: JoinHandle<()>) -> BlockOnDrop {
        BlockOnDrop(Some(handle))
    }
}

impl Drop for BlockOnDrop {
    fn drop(&mut self) {
        eprintln!("Thread 2 blocking on thread 1");
        let _ = self.0.take().unwrap().join();
        eprintln!("Thread 1 has exited");
    }
}

fn main() {
    let t1_started_pair = Arc::new((Mutex::new(false), Condvar::new()));
    let t2_started_pair = Arc::new((Mutex::new(false), Condvar::new()));

    let t1_continue_mutex = Arc::new(Mutex::new(()));
    let t1_continue_guard = t1_continue_mutex.lock();

    let t1 = {
        let t1_started_pair = t1_started_pair.clone();
        let t1_continue_mutex = t1_continue_mutex.clone();
        spawn(move || {
            eprintln!("Thread 1 starting, will block on mutex");
            let (mutex, condvar) = &*t1_started_pair;
            *mutex.lock().unwrap() = true;
            condvar.notify_one();

            drop(t1_continue_mutex.lock());
            panic!("panic in thread 1");
        })
    };

    // Wait for thread 1 to signal it has started.
    let (t1_started_mutex, t1_started_condvar) = &*t1_started_pair;
    let mut t1_started_guard = t1_started_mutex.lock().unwrap();
    while !*t1_started_guard {
        t1_started_guard = t1_started_condvar.wait(t1_started_guard).unwrap();
    }
    eprintln!("Thread 1 reported it has started");
    // Thread 1 should now be blocked waiting on t1_continue_mutex.

    let t2 = {
        let t2_started_pair = t2_started_pair.clone();
        let block_on_drop = BlockOnDrop::new(t1);
        spawn(move || {
            let _capture = block_on_drop;

            let (mutex, condvar) = &*t2_started_pair;
            *mutex.lock().unwrap() = true;
            condvar.notify_one();

            panic!("panic in thread 2");
        })
    };

    // Wait for thread 2 to signal it has started.
    let (t2_started_mutex, t2_started_condvar) = &*t2_started_pair;
    let mut t2_started_guard = t2_started_mutex.lock().unwrap();
    while !*t2_started_guard {
        t2_started_guard = t2_started_condvar.wait(t2_started_guard).unwrap();
    }
    eprintln!("Thread 2 reported it has started");
    // Thread 2 should now have already panicked and be in the middle of
    // unwinding. It should now be blocked on joining thread 1.

    // Unlock t1_continue_mutex, and allow thread 1 to proceed.
    eprintln!("Unlocking mutex");
    drop(t1_continue_guard);
    // Thread 1 will panic the next time it is scheduled. This will test the
    // behavior of interest to this test, whether Miri properly handles
    // concurrent panics in two different threads.

    // Block the main thread on waiting to join thread 2. Thread 2 should
    // already be blocked on joining thread 1, so thread 1 will be scheduled
    // to run next, as it is the only ready thread.
    assert!(t2.join().is_err());
    eprintln!("Thread 2 has exited");
}
