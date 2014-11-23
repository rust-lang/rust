// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Useful synchronization primitives
//!
//! This module contains useful safe and unsafe synchronization primitives.
//! Most of the primitives in this module do not provide any sort of locking
//! and/or blocking at all, but rather provide the necessary tools to build
//! other types of concurrent primitives.

#![experimental]

pub use self::one::{Once, ONCE_INIT};

pub use alloc::arc::{Arc, Weak};
pub use self::lock::{Mutex, MutexGuard, Condvar, Barrier,
                     RWLock, RWLockReadGuard, RWLockWriteGuard};

// The mutex/rwlock in this module are not meant for reexport
pub use self::raw::{Semaphore, SemaphoreGuard};

pub use self::future::Future;
pub use self::task_pool::TaskPool;

// Core building blocks for all primitives in this crate

#[stable]
pub mod atomic;

// Concurrent data structures

pub mod spsc_queue;
pub mod mpsc_queue;
pub mod mpmc_bounded_queue;
pub mod deque;

// Low-level concurrency primitives

mod raw;
mod mutex;
mod one;

// Higher level primitives based on those above

mod lock;

// Task management

mod future;
mod task_pool;
