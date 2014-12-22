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

pub use alloc::arc::{Arc, Weak};

pub use self::mutex::{Mutex, MutexGuard, StaticMutex, StaticMutexGuard, MUTEX_INIT};
pub use self::rwlock::{RWLock, StaticRWLock, RWLOCK_INIT};
pub use self::rwlock::{RWLockReadGuard, RWLockWriteGuard};
pub use self::rwlock::{StaticRWLockReadGuard, StaticRWLockWriteGuard};
pub use self::condvar::{Condvar, StaticCondvar, CONDVAR_INIT, AsMutexGuard};
pub use self::once::{Once, ONCE_INIT};
pub use self::semaphore::{Semaphore, SemaphoreGuard};
pub use self::barrier::Barrier;

pub use self::future::Future;
pub use self::task_pool::TaskPool;

pub mod atomic;
mod barrier;
mod condvar;
mod future;
mod mutex;
mod once;
mod poison;
mod rwlock;
mod semaphore;
mod task_pool;
