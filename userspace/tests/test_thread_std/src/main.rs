//! `std::thread` smoke tests for ThingOS.
//!
//! These tests exercise the `std::thread` PAL layer implemented in
//! `library/std/src/sys/thread/thingos.rs`.  They run inside the ThingOS
//! userspace and use `std::thread::spawn` / `JoinHandle::join` backed by
//! `SYS_SPAWN_THREAD` + `SYS_TASK_WAIT`.
//!
//! # Test inventory
//!
//! | Test                       | What it exercises                                   |
//! |----------------------------|-----------------------------------------------------|
//! | `thread_spawn_join`        | Basic spawn + return value through join             |
//! | `thread_shared_mutex`      | `Arc<Mutex<T>>` shared across threads               |
//! | `thread_yield`             | Two threads making progress via `yield_now`         |
//! | `thread_many_increments`   | Mutex-protected counter across N threads            |
//! | `thread_local_basic`       | `thread_local!` value differs per thread            |
//! | `thread_panic_join`        | Panicking thread aborts (panic = "abort" model)     |
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


extern crate std;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// ── Test 1: thread_spawn_join ─────────────────────────────────────────────────
//
// Spawn a thread that returns a known value; verify join delivers it.

fn test_spawn_join() {
    std::println!("[test_thread_std] test_spawn_join: starting");
    let h = std::thread::spawn(|| 42u32);
    let result = h.join().expect("join failed");
    assert_eq!(result, 42, "expected 42, got {}", result);
    std::println!("[test_thread_std] test_spawn_join: PASS");
}

// ── Test 2: thread_shared_mutex ───────────────────────────────────────────────
//
// Arc<Mutex<T>> shared between two threads.

fn test_shared_mutex() {
    std::println!("[test_thread_std] test_shared_mutex: starting");

    let x = Arc::new(Mutex::new(0u32));
    let y = Arc::clone(&x);

    let h = std::thread::spawn(move || {
        *y.lock().unwrap() = 7;
    });

    h.join().expect("join failed");
    assert_eq!(*x.lock().unwrap(), 7, "mutex value not written by child thread");

    std::println!("[test_thread_std] test_shared_mutex: PASS");
}

// ── Test 3: thread_yield ──────────────────────────────────────────────────────
//
// Two threads both increment a counter; yield_now lets each make progress.
// We just assert the final value is correct — the point is that both threads ran.

fn test_yield() {
    std::println!("[test_thread_std] test_yield: starting");

    static COUNTER: AtomicU32 = AtomicU32::new(0);
    COUNTER.store(0, Ordering::SeqCst);

    const ITERS: u32 = 500;

    let h1 = std::thread::spawn(|| {
        for _ in 0..ITERS {
            COUNTER.fetch_add(1, Ordering::AcqRel);
            std::thread::yield_now();
        }
    });
    let h2 = std::thread::spawn(|| {
        for _ in 0..ITERS {
            COUNTER.fetch_add(1, Ordering::AcqRel);
            std::thread::yield_now();
        }
    });

    h1.join().expect("thread 1 join failed");
    h2.join().expect("thread 2 join failed");

    let total = COUNTER.load(Ordering::SeqCst);
    assert_eq!(total, ITERS * 2, "yield: counter {} != expected {}", total, ITERS * 2);

    std::println!("[test_thread_std] test_yield: PASS");
}

// ── Test 4: thread_many_increments ────────────────────────────────────────────
//
// N threads each increment a Mutex-protected counter M times; verify total.

fn test_many_increments() {
    std::println!("[test_thread_std] test_many_increments: starting");

    const N_THREADS: usize = 4;
    const M_ITERS: u32 = 250;

    let counter = Arc::new(Mutex::new(0u32));
    let mut handles = alloc::vec::Vec::new();

    for _ in 0..N_THREADS {
        let c = Arc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            for _ in 0..M_ITERS {
                *c.lock().unwrap() += 1;
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join failed");
    }

    let total = *counter.lock().unwrap();
    let expected = N_THREADS as u32 * M_ITERS;
    assert_eq!(total, expected, "many_increments: {} != {}", total, expected);

    std::println!("[test_thread_std] test_many_increments: PASS");
}

// ── Test 5: thread_local_basic ────────────────────────────────────────────────
//
// Verify that thread_local! values are per-thread.
//
// This test is only meaningful with native ELF TLS (`target_thread_local`).
// With `"has-thread-local": true` in the ThingOS target JSON, rustc generates
// direct FS-relative accesses for #[thread_local] statics.

fn test_thread_local_basic() {
    std::println!("[test_thread_std] test_thread_local_basic: starting");

    thread_local! {
        static VAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    // Set the main thread's TLS value to 1.
    VAL.with(|v| v.set(1));

    // Spawn a child that sets its own copy to 2 and returns it.
    let h = std::thread::spawn(|| {
        // Initial value should be 0 (fresh TLS block, initialized from template).
        let initial = VAL.with(|v| v.get());
        VAL.with(|v| v.set(2));
        let after_set = VAL.with(|v| v.get());
        (initial, after_set)
    });

    let (initial, after_set) = h.join().expect("thread join failed");

    // Main thread's copy must still be 1.
    let main_val = VAL.with(|v| v.get());

    assert_eq!(initial, 0, "child initial TLS value should be 0, got {}", initial);
    assert_eq!(after_set, 2, "child after set should be 2, got {}", after_set);
    assert_eq!(main_val, 1, "main thread TLS value should still be 1, got {}", main_val);

    std::println!("[test_thread_std] test_thread_local_basic: PASS");
}

// ── Test 6: thread_panic_join ─────────────────────────────────────────────────
//
// ThingOS targets use `panic = "abort"`.  A panicking child thread therefore
// terminates the whole process before join() can return.  The ThingOS port
// documents this as the current panic model.
//
// This test verifies that a non-panicking thread joins successfully, which
// confirms the join path is sound.  A separate manual test (not included here
// to avoid process abort) would be needed to observe panic→abort behaviour.

fn test_panic_join_no_panic() {
    std::println!("[test_thread_std] test_panic_join_no_panic: starting");

    // Successful thread (no panic) should join with Ok(value).
    let h = std::thread::spawn(|| "success");
    let result = h.join();
    assert!(result.is_ok(), "expected Ok from non-panicking thread");
    assert_eq!(result.unwrap(), "success");

    std::println!("[test_thread_std] test_panic_join_no_panic: PASS (panic=abort model: child panics abort the process)");
}

// ── Entry ─────────────────────────────────────────────────────────────────────

#[stem::main]
fn main(_arg: usize) -> ! {
    std::println!("--- test_thread_std starting ---");

    test_spawn_join();
    test_shared_mutex();
    test_yield();
    test_many_increments();
    test_thread_local_basic();
    test_panic_join_no_panic();

    std::println!("--- test_thread_std: all tests PASSED ---");
    std::process::exit(0);
}
