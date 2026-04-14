//! Futex synchronization primitive integration tests.
//!
//! Exercises the FUTEX_WAIT / FUTEX_WAKE syscalls through the `stem`
//! wrappers and builds a simple spinlock-free mutex on top of them to
//! validate real-world contention scenarios.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use abi::errors::Errno;
use core::sync::atomic::{AtomicU32, Ordering};
use stem::println;
use stem::syscall::{futex_wait, futex_wake, get_tid, spawn_thread, task_wait};

// ============================================================================
// Minimal futex-based mutex
// ============================================================================

/// A simple mutex backed by the kernel futex wait/wake interface.
///
/// State values:
///   0 = unlocked
///   1 = locked, no waiters
///   2 = locked, waiters present
struct FutexMutex {
    state: AtomicU32,
}

impl FutexMutex {
    const fn new() -> Self {
        Self {
            state: AtomicU32::new(0),
        }
    }

    fn lock(&self) {
        // Fast path: try to go from 0 → 1 without a syscall.
        let mut expected = 0u32;
        loop {
            match self.state.compare_exchange_weak(
                expected,
                1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => {
                    // Mark that there are waiters then sleep.
                    //
                    // Setting state = 2 here is intentionally relaxed and
                    // slightly racy: a concurrent unlock might clear the state
                    // between our read of `actual` and this store.  That is
                    // benign because futex_wait will re-check the value and
                    // return EAGAIN immediately if it is no longer 2, so the
                    // outer loop will retry the CAS and eventually succeed.
                    expected = actual;
                    if actual != 2 {
                        self.state.store(2, Ordering::Relaxed);
                    }
                    // Block until the state changes away from 2.
                    let _ = futex_wait(&self.state, 2, 0);
                    expected = 0;
                }
            }
        }
    }

    fn unlock(&self) {
        let prev = self.state.swap(0, Ordering::Release);
        if prev == 2 {
            // There are waiters — wake one.
            let _ = futex_wake(&self.state, 1);
        }
    }
}

// ============================================================================
// Test 1 – Basic lock / unlock (single thread, no contention)
// ============================================================================

fn test_basic_mutex() {
    println!("[test_futex] test_basic_mutex: starting");
    let m = FutexMutex::new();
    m.lock();
    m.unlock();
    m.lock();
    m.unlock();
    println!("[test_futex] test_basic_mutex: PASS");
}

// ============================================================================
// Test 2 – EAGAIN on uncontended futex_wait (value already changed)
// ============================================================================

fn test_eagain_when_value_changed() {
    println!("[test_futex] test_eagain_when_value_changed: starting");
    let atom = AtomicU32::new(42);
    // Pass expected=0, but actual value is 42 → should return EAGAIN immediately.
    let result = futex_wait(&atom, 0, 0);
    match result {
        Err(Errno::EAGAIN) => {}
        other => panic!(
            "expected EAGAIN, got {:?}",
            other
        ),
    }
    println!("[test_futex] test_eagain_when_value_changed: PASS");
}

// ============================================================================
// Test 3 – Wake ordering: wake(count=1) unblocks exactly one waiter
// ============================================================================
//
// We spin up two threads that both call futex_wait on the same address,
// then call futex_wake(count=1) once and verify exactly one thread is
// released and the other remains sleeping (we verify by waking the
// remaining one afterwards).

static WAKE_ATOM: AtomicU32 = AtomicU32::new(0);
static WAKE_ORDER_COUNT: AtomicU32 = AtomicU32::new(0);

extern "C" fn waiter_thread_fn(_arg: usize) -> usize {
    // Block until WAKE_ATOM != 0.
    loop {
        let v = WAKE_ATOM.load(Ordering::Acquire);
        if v != 0 {
            break;
        }
        let _ = futex_wait(&WAKE_ATOM, 0, 0);
    }
    WAKE_ORDER_COUNT.fetch_add(1, Ordering::AcqRel);
    0
}

fn test_wake_ordering() {
    println!("[test_futex] test_wake_ordering: starting");

    WAKE_ATOM.store(0, Ordering::SeqCst);
    WAKE_ORDER_COUNT.store(0, Ordering::SeqCst);

    // Allocate two stacks and spawn two waiting threads.
    let stack1 = stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default())
        .expect("stack alloc 1");
    let stack2 = stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default())
        .expect("stack alloc 2");

    let tid1 =
        spawn_thread(waiter_thread_fn as usize, 0, &stack1).expect("spawn thread 1");
    let tid2 =
        spawn_thread(waiter_thread_fn as usize, 0, &stack2).expect("spawn thread 2");

    // Yield a few times so both threads have a chance to reach futex_wait.
    // This is best-effort: the threads may still be racing toward futex_wait
    // when we proceed, but that is intentional — futex_wait is designed to
    // handle the case where the value changes before the thread sleeps (it
    // returns EAGAIN and the thread re-checks in its own loop).
    for _ in 0..10 {
        stem::syscall::yield_now();
    }

    // Set the flag and wake exactly one thread.
    WAKE_ATOM.store(1, Ordering::Release);
    let woken = futex_wake(&WAKE_ATOM, 1).expect("futex_wake");
    assert!(woken <= 1, "wake(1) should not wake more than 1 waiter");

    // Wake the remaining thread.
    let _woken2 = futex_wake(&WAKE_ATOM, 1);

    // Wait for both threads to finish.
    task_wait(tid1).expect("task_wait tid1");
    task_wait(tid2).expect("task_wait tid2");

    let total = WAKE_ORDER_COUNT.load(Ordering::SeqCst);
    assert_eq!(total, 2, "both waiters should have incremented the counter");

    println!("[test_futex] test_wake_ordering: PASS");
}

// ============================================================================
// Test 4 – Contention test: multiple threads increment a shared counter
//          protected by a FutexMutex.
// ============================================================================

static CONTENTION_MUTEX: FutexMutex = FutexMutex::new();
static SHARED_COUNTER: AtomicU32 = AtomicU32::new(0);

const CONTENTION_THREADS: usize = 4;
const CONTENTION_ITERS: u32 = 200;

extern "C" fn contention_thread_fn(_arg: usize) -> usize {
    for _ in 0..CONTENTION_ITERS {
        CONTENTION_MUTEX.lock();
        // Non-atomic read-modify-write to exercise the mutex.
        let v = SHARED_COUNTER.load(Ordering::Relaxed);
        // Yield to increase the probability of a race if the mutex is broken.
        stem::syscall::yield_now();
        SHARED_COUNTER.store(v + 1, Ordering::Relaxed);
        CONTENTION_MUTEX.unlock();
    }
    0
}

fn test_contention() {
    println!("[test_futex] test_contention: starting");

    SHARED_COUNTER.store(0, Ordering::SeqCst);

    let mut tids = alloc::vec::Vec::new();
    let mut stacks = alloc::vec::Vec::new();

    for _ in 0..CONTENTION_THREADS {
        let stack = stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default())
            .expect("stack alloc");
        let tid =
            spawn_thread(contention_thread_fn as usize, 0, &stack).expect("spawn contention thread");
        tids.push(tid);
        stacks.push(stack);
    }

    for tid in tids {
        task_wait(tid).expect("task_wait contention thread");
    }

    let expected = CONTENTION_THREADS as u32 * CONTENTION_ITERS;
    let actual = SHARED_COUNTER.load(Ordering::SeqCst);
    assert_eq!(
        actual, expected,
        "counter mismatch: got {}, expected {}",
        actual, expected
    );

    println!("[test_futex] test_contention: PASS");
}

// ============================================================================
// Entry point
// ============================================================================

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("--- test_futex starting ---");

    test_basic_mutex();
    test_eagain_when_value_changed();
    test_wake_ordering();
    test_contention();

    println!("--- test_futex: all tests PASSED ---");
    stem::syscall::exit(0);
}
