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

#![stable(feature = "rust1", since = "1.0.0")]

pub use alloc::arc::{Arc, Weak};
pub use core::sync::atomic;
pub use self::barrier::{Barrier, BarrierWaitResult};
pub use self::condvar::{Condvar, StaticCondvar, WaitTimeoutResult, CONDVAR_INIT};
pub use self::mutex::MUTEX_INIT;
pub use self::mutex::{Mutex, MutexGuard, StaticMutex};
pub use self::remutex::{ReentrantMutex, ReentrantMutexGuard};
pub use self::once::{Once, ONCE_INIT};
pub use self::poison::{PoisonError, TryLockError, TryLockResult, LockResult};
pub use self::rwlock::{RwLockReadGuard, RwLockWriteGuard};
pub use self::rwlock::{RwLock, StaticRwLock, RW_LOCK_INIT};
pub use self::semaphore::{Semaphore, SemaphoreGuard};

pub mod mpsc;

mod barrier;
mod condvar;
mod mutex;
mod remutex;
mod once;
mod poison;
mod rwlock;
mod semaphore;
