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
//! This modules contains useful safe and unsafe synchronization primitives.
//! Most of the primitives in this module do not provide any sort of locking
//! and/or blocking at all, but rather provide the necessary tools to build
//! other types of concurrent primitives.

pub use core_sync::{atomics, deque, mpmc_bounded_queue, mpsc_queue, spsc_queue};
pub use core_sync::{Arc, Weak, Mutex, MutexGuard, Condvar, Barrier};
pub use core_sync::{RWLock, RWLockReadGuard, RWLockWriteGuard};
pub use core_sync::{Semaphore, SemaphoreGuard};
pub use core_sync::one::{Once, ONCE_INIT};

pub use self::future::Future;
pub use self::task_pool::TaskPool;

mod future;
mod task_pool;
