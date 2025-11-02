//@ revisions: default_R1W1 default_R1W2 spinloop_assume_R1W1 spinloop_assume_R1W2
//@compile-flags: -Zmiri-ignore-leaks -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-verbose
//@normalize-stderr-test: "Verification took .*s" -> "Verification took [TIME]s"

// This test is a translations of the GenMC test `ms-queue-dynamic`, but with all code related to GenMC's hazard pointer API removed.
// The test leaks memory, so leak checks are disabled.
//
// Test variant naming convention: "[VARIANT_NAME]_R[#reader_threads]_W[#writer_threads]".
// We test different numbers of writer threads to see the scaling.
// Implementing optimizations such as automatic spinloop-assume transformation or symmetry reduction should reduce the number of explored executions.
// We also test variants using manual spinloop replacement, which should yield fewer executions in total compared to the unmodified code.
//
// FIXME(genmc): Add revisions `default_R1W3` and `spinloop_assume_R1W3` once Miri-GenMC performance is improved. These currently slow down the test suite too much.
//
// The test uses verbose output to see the difference between blocked and explored executions.

#![no_main]
#![allow(static_mut_refs)]

#[path = "../../../utils/genmc.rs"]
mod genmc;
#[allow(unused)]
#[path = "../../../utils/mod.rs"]
mod utils;

use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use genmc::*;
use libc::pthread_t;

const MAX_THREADS: usize = 32;

static mut QUEUE: MyStack = MyStack::new();
static mut INPUT: [u64; MAX_THREADS] = [0; MAX_THREADS];
static mut OUTPUT: [Option<u64>; MAX_THREADS] = [None; MAX_THREADS];

#[repr(C)]
struct Node {
    value: u64,
    next: AtomicPtr<Node>,
}

struct MyStack {
    head: AtomicPtr<Node>,
    tail: AtomicPtr<Node>,
}

impl Node {
    pub unsafe fn alloc() -> *mut Self {
        alloc(Layout::new::<Self>()) as *mut Self
    }

    pub unsafe fn free(node: *mut Self) {
        dealloc(node as *mut u8, Layout::new::<Self>())
    }
}

impl MyStack {
    pub const fn new() -> Self {
        let head = AtomicPtr::new(std::ptr::null_mut());
        let tail = AtomicPtr::new(std::ptr::null_mut());
        Self { head, tail }
    }

    pub unsafe fn init_queue(&mut self, _num_threads: usize) {
        let dummy = Node::alloc();

        (*dummy).next = AtomicPtr::new(std::ptr::null_mut());
        self.head = AtomicPtr::new(dummy);
        self.tail = AtomicPtr::new(dummy);
    }

    pub unsafe fn clear_queue(&mut self, _num_threads: usize) {
        let mut next;
        let mut head = *self.head.get_mut();
        while !head.is_null() {
            next = *(*head).next.get_mut();
            Node::free(head);
            head = next;
        }
    }

    pub unsafe fn enqueue(&self, value: u64) {
        let mut tail;
        let node = Node::alloc();
        (*node).value = value;
        (*node).next = AtomicPtr::new(std::ptr::null_mut());

        loop {
            tail = self.tail.load(Acquire);
            let next = (*tail).next.load(Acquire);
            if tail != self.tail.load(Acquire) {
                // Looping here has no side effects, so we prevent exploring any executions where this branch happens.
                #[cfg(any(spinloop_assume_R1W1, spinloop_assume_R1W2, spinloop_assume_R1W3))]
                utils::miri_genmc_assume(false); // GenMC will stop any execution that reaches this.
                continue;
            }

            if next.is_null() {
                if (*tail).next.compare_exchange(next, node, Release, Relaxed).is_ok() {
                    break;
                }
            } else {
                let _ = self.tail.compare_exchange(tail, next, Release, Relaxed);
            }
        }

        let _ = self.tail.compare_exchange(tail, node, Release, Relaxed);
    }

    pub unsafe fn dequeue(&self) -> Option<u64> {
        loop {
            let head = self.head.load(Acquire);
            let tail = self.tail.load(Acquire);

            let next = (*head).next.load(Acquire);
            if self.head.load(Acquire) != head {
                // Looping here has no side effects, so we prevent exploring any executions where this branch happens.
                #[cfg(any(spinloop_assume_R1W1, spinloop_assume_R1W2, spinloop_assume_R1W3))]
                utils::miri_genmc_assume(false); // GenMC will stop any execution that reaches this.
                continue;
            }
            if head == tail {
                if next.is_null() {
                    return None;
                }
                let _ = self.tail.compare_exchange(tail, next, Release, Relaxed);
            } else {
                let ret_val = (*next).value;
                if self.head.compare_exchange(head, next, Release, Relaxed).is_ok() {
                    // NOTE: The popped `Node` is leaked.
                    return Some(ret_val);
                }
                // Looping here has no side effects, so we prevent exploring any executions where this branch happens.
                // All operations in the loop leading to here are either loads, or failed compare-exchange operations.
                #[cfg(any(spinloop_assume_R1W1, spinloop_assume_R1W2, spinloop_assume_R1W3))]
                utils::miri_genmc_assume(false); // GenMC will stop any execution that reaches this.
            }
        }
    }
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // We try multiple different parameters for the number and types of threads:
    let (readers, writers) = if cfg!(any(default_R1W3, spinloop_assume_R1W3)) {
        (1, 3)
    } else if cfg!(any(default_R1W2, spinloop_assume_R1W2)) {
        (1, 2)
    } else {
        // default_R1W1, spinloop_assume_R1W1
        (1, 1)
    };

    let num_threads = readers + writers;
    if num_threads > MAX_THREADS {
        std::process::abort();
    }

    let mut i = 0;
    unsafe {
        MyStack::init_queue(&mut QUEUE, num_threads);

        /* Spawn threads */
        let mut thread_ids: [pthread_t; MAX_THREADS] = [0; MAX_THREADS];
        for _ in 0..readers {
            let pid = i as u64;
            thread_ids[i] = spawn_pthread_closure(move || {
                OUTPUT[pid as usize] = QUEUE.dequeue();
            });
            i += 1;
        }
        for _ in 0..writers {
            let pid = i as u64;
            thread_ids[i] = spawn_pthread_closure(move || {
                INPUT[pid as usize] = pid * 10;
                QUEUE.enqueue(INPUT[pid as usize]);
            });
            i += 1;
        }

        for i in 0..num_threads {
            join_pthread(thread_ids[i]);
        }

        MyStack::clear_queue(&mut QUEUE, num_threads);
    }

    0
}
