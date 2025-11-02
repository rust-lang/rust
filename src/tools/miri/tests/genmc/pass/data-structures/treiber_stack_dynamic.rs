//@ revisions: default_R1W1 default_R1W2 default_R1W3 spinloop_assume_R1W1 spinloop_assume_R1W2 spinloop_assume_R1W3
//@compile-flags: -Zmiri-ignore-leaks -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-verbose
//@normalize-stderr-test: "Verification took .*s" -> "Verification took [TIME]s"

// This test is a translations of the GenMC test `treiber-stack-dynamic`, but with all code related to GenMC's hazard pointer API removed.
// The test leaks memory, so leak checks are disabled.
//
// Test variant naming convention: "[VARIANT_NAME]_R[#reader_threads]_W[#writer_threads]".
// We test different numbers of writer threads to see the scaling.
// Implementing optimizations such as automatic spinloop-assume transformation or symmetry reduction should reduce the number of explored executions.
// We also test variants using manual spinloop replacement, which should yield fewer executions in total compared to the unmodified code.
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

static mut STACK: MyStack = MyStack::new();

#[repr(C)]
struct Node {
    value: u64,
    next: AtomicPtr<Node>,
}

struct MyStack {
    top: AtomicPtr<Node>,
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
        Self { top: AtomicPtr::new(std::ptr::null_mut()) }
    }

    pub unsafe fn clear_stack(&mut self, _num_threads: usize) {
        let mut next;
        let mut top = *self.top.get_mut();
        while !top.is_null() {
            next = *(*top).next.get_mut();
            Node::free(top);
            top = next;
        }
    }

    pub unsafe fn push(&self, value: u64) {
        let node = Node::alloc();
        (*node).value = value;

        loop {
            let top = self.top.load(Acquire);
            (*node).next.store(top, Relaxed);
            if self.top.compare_exchange(top, node, Release, Relaxed).is_ok() {
                break;
            }
            // We manually limit the number of iterations of this spinloop to 1.
            #[cfg(any(spinloop_assume_R1W1, spinloop_assume_R1W2, spinloop_assume_R1W3))]
            utils::miri_genmc_assume(false); // GenMC will stop any execution that reaches this.
        }
    }

    pub unsafe fn pop(&self) -> u64 {
        loop {
            let top = self.top.load(Acquire);
            if top.is_null() {
                return 0;
            }

            let next = (*top).next.load(Relaxed);
            if self.top.compare_exchange(top, next, Release, Relaxed).is_ok() {
                // NOTE: The popped `Node` is leaked.
                return (*top).value;
            }
            // We manually limit the number of iterations of this spinloop to 1.
            #[cfg(any(spinloop_assume_R1W1, spinloop_assume_R1W2, spinloop_assume_R1W3))]
            utils::miri_genmc_assume(false); // GenMC will stop any execution that reaches this.
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
        let mut thread_ids: [pthread_t; MAX_THREADS] = [0; MAX_THREADS];
        for _ in 0..readers {
            thread_ids[i] = spawn_pthread_closure(move || {
                let _idx = STACK.pop();
            });
            i += 1;
        }
        for _ in 0..writers {
            let pid = i as u64;
            thread_ids[i] = spawn_pthread_closure(move || {
                STACK.push(pid);
            });
            i += 1;
        }

        for i in 0..num_threads {
            join_pthread(thread_ids[i]);
        }

        MyStack::clear_stack(&mut STACK, num_threads);
    }

    0
}
