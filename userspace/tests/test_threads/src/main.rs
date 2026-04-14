//! Thread integration tests.
//!
//! Validates the kernel threading substrate:
//!   1. Multiple threads can be created in the same address space.
//!   2. Shared-memory writes made by one thread are visible to all others.
//!   3. `task_wait` (join) correctly blocks until the target thread exits.
//!   4. `get_tid` returns a unique TID per thread while `getpid` returns the
//!      same thread-group ID (TGID) for every thread in the process.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use stem::println;
use stem::syscall::{exit, get_tid, getpid, spawn_thread, task_wait, yield_now};

// ============================================================================
// Test 1 – Basic spawn and join
// ============================================================================
//
// Spawn a single thread, wait for it to exit, and verify task_wait returns Ok.

extern "C" fn thread_simple(_arg: usize) -> ! {
    exit(0);
}

fn test_spawn_and_join() {
    println!("[test_threads] test_spawn_and_join: starting");

    let stack =
        stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default()).expect("stack");
    let tid = spawn_thread(thread_simple as usize, 0, &stack).expect("spawn");

    task_wait(tid).expect("task_wait");

    println!("[test_threads] test_spawn_and_join: PASS");
}

// ============================================================================
// Test 2 – Shared memory write visibility
// ============================================================================
//
// The main thread writes a value into a shared atomic, then spawns a worker
// that reads it back and stores the observed value in a second atomic.
// After joining, the main thread asserts the worker saw the written value.
//
// Because all threads share the same address space the write must be visible
// without any cross-process IPC.

static WRITE_SRC: AtomicU32 = AtomicU32::new(0);
static OBSERVED_BY_WORKER: AtomicU32 = AtomicU32::new(0);

extern "C" fn shared_mem_worker(_arg: usize) -> ! {
    let val = WRITE_SRC.load(Ordering::Acquire);
    OBSERVED_BY_WORKER.store(val, Ordering::Release);
    exit(0);
}

fn test_shared_memory_visibility() {
    println!("[test_threads] test_shared_memory_visibility: starting");

    const MAGIC: u32 = 0xDEAD_BEEF;
    WRITE_SRC.store(MAGIC, Ordering::Release);
    OBSERVED_BY_WORKER.store(0, Ordering::SeqCst);

    let stack =
        stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default()).expect("stack");
    let tid = spawn_thread(shared_mem_worker as usize, 0, &stack).expect("spawn");
    task_wait(tid).expect("task_wait");

    let observed = OBSERVED_BY_WORKER.load(Ordering::Acquire);
    assert_eq!(
        observed, MAGIC,
        "shared memory not visible: expected 0x{:x}, got 0x{:x}",
        MAGIC, observed
    );

    println!("[test_threads] test_shared_memory_visibility: PASS");
}

// ============================================================================
// Test 3 – Multiple threads, shared counter
// ============================================================================
//
// Spin up N threads each of which increments a shared counter M times.
// After all threads exit the counter must equal N * M.

const N_THREADS: usize = 4;
const M_ITERS: u32 = 100;

static COUNTER: AtomicU32 = AtomicU32::new(0);

extern "C" fn counter_thread(_arg: usize) -> ! {
    for _ in 0..M_ITERS {
        COUNTER.fetch_add(1, Ordering::AcqRel);
    }
    exit(0);
}

fn test_multi_thread_counter() {
    println!("[test_threads] test_multi_thread_counter: starting");

    COUNTER.store(0, Ordering::SeqCst);

    let mut tids = alloc::vec::Vec::new();
    let mut stacks = alloc::vec::Vec::new();

    for _ in 0..N_THREADS {
        let stack =
            stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default())
                .expect("stack alloc");
        let tid = spawn_thread(counter_thread as usize, 0, &stack).expect("spawn");
        tids.push(tid);
        stacks.push(stack);
    }

    for tid in tids {
        task_wait(tid).expect("task_wait");
    }

    let expected = N_THREADS as u32 * M_ITERS;
    let actual = COUNTER.load(Ordering::SeqCst);
    assert_eq!(
        actual, expected,
        "counter mismatch: got {}, expected {}",
        actual, expected
    );

    println!("[test_threads] test_multi_thread_counter: PASS");
}

// ============================================================================
// Test 4 – TID uniqueness and TGID identity
// ============================================================================
//
// Each thread records its own TID.  After all threads exit, verify that:
//   • every TID is unique (no two threads share a TID), and
//   • every thread sees the same TGID (getpid() == main thread's getpid()).

const TID_THREADS: usize = 3;
static RECORDED_TIDS: [AtomicU64; TID_THREADS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static RECORDED_TGIDS: [AtomicU64; TID_THREADS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static TID_SLOT: AtomicU32 = AtomicU32::new(0);

extern "C" fn record_tid_thread(_arg: usize) -> ! {
    let slot = TID_SLOT.fetch_add(1, Ordering::AcqRel) as usize;
    if slot < TID_THREADS {
        RECORDED_TIDS[slot].store(get_tid().unwrap_or(0), Ordering::Release);
        RECORDED_TGIDS[slot].store(getpid() as u64, Ordering::Release);
    }
    exit(0);
}

fn test_tid_uniqueness_and_tgid_identity() {
    println!("[test_threads] test_tid_uniqueness_and_tgid_identity: starting");

    TID_SLOT.store(0, Ordering::SeqCst);
    for r in &RECORDED_TIDS {
        r.store(0, Ordering::SeqCst);
    }
    for r in &RECORDED_TGIDS {
        r.store(0, Ordering::SeqCst);
    }

    let main_tgid = getpid() as u64;

    let mut tids = alloc::vec::Vec::new();
    let mut stacks = alloc::vec::Vec::new();

    for _ in 0..TID_THREADS {
        let stack =
            stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default())
                .expect("stack alloc");
        let tid = spawn_thread(record_tid_thread as usize, 0, &stack).expect("spawn");
        tids.push(tid);
        stacks.push(stack);
    }

    // Yield to let threads run before joining.
    for _ in 0..20 {
        yield_now();
    }

    for tid in tids {
        task_wait(tid).expect("task_wait");
    }

    // Verify all TIDs are non-zero and unique.
    let mut seen = alloc::vec::Vec::new();
    for r in &RECORDED_TIDS {
        let t = r.load(Ordering::Acquire);
        assert!(t != 0, "thread TID was not recorded");
        assert!(!seen.contains(&t), "duplicate TID {}", t);
        seen.push(t);
    }

    // Verify all threads saw the same TGID as the main thread.
    for (i, r) in RECORDED_TGIDS.iter().enumerate() {
        let tgid = r.load(Ordering::Acquire);
        assert_eq!(
            tgid, main_tgid,
            "thread {} reported TGID {} instead of {}",
            i, tgid, main_tgid
        );
    }

    println!("[test_threads] test_tid_uniqueness_and_tgid_identity: PASS");
}

// ============================================================================
// Test 5 – Argument passing to spawned thread
// ============================================================================
//
// Passes a known value to a thread via the `arg` parameter of `spawn_thread`
// and verifies it was received by storing it in a shared atomic.

static RECEIVED_ARG: AtomicU64 = AtomicU64::new(0);

extern "C" fn record_arg_thread(arg: usize) -> ! {
    RECEIVED_ARG.store(arg as u64, Ordering::Release);
    exit(0);
}

fn test_thread_arg_passing() {
    println!("[test_threads] test_thread_arg_passing: starting");

    const ARG_VAL: usize = 0x1234_5678;
    RECEIVED_ARG.store(0, Ordering::SeqCst);

    let stack =
        stem::stack::Stack::alloc_growing_stack(stem::stack::StackSpec::default()).expect("stack");
    let tid = spawn_thread(record_arg_thread as usize, ARG_VAL, &stack).expect("spawn");
    task_wait(tid).expect("task_wait");

    let received = RECEIVED_ARG.load(Ordering::Acquire);
    assert_eq!(
        received, ARG_VAL as u64,
        "arg not received: got {} expected {}",
        received, ARG_VAL
    );

    println!("[test_threads] test_thread_arg_passing: PASS");
}

// ============================================================================
// Entry point
// ============================================================================

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("--- test_threads starting ---");

    test_spawn_and_join();
    test_shared_memory_visibility();
    test_multi_thread_counter();
    test_tid_uniqueness_and_tgid_identity();
    test_thread_arg_passing();

    println!("--- test_threads: all tests PASSED ---");
    exit(0);
}
