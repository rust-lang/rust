// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Useful synchronization primitives.
//!
//! This module contains useful safe and unsafe synchronization primitives.
//! Most of the primitives in this module do not provide any sort of locking
//! and/or blocking at all, but rather provide the necessary tools to build
//! other types of concurrent primitives.

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::arc::{Arc, Weak};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::sync::atomic;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::barrier::{Barrier, BarrierWaitResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::condvar::{Condvar, WaitTimeoutResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::mutex::{Mutex, MutexGuard};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::once::{Once, OnceState, ONCE_INIT};
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys_common::poison::{PoisonError, TryLockError, TryLockResult, LockResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub mod mpsc;

mod barrier;
mod condvar;
mod mutex;
mod once;
mod rwlock;
