// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use cell::UnsafeCell;
use error::FromError;
use fmt;
use thread::Thread;

pub struct Flag { failed: UnsafeCell<bool> }
pub const FLAG_INIT: Flag = Flag { failed: UnsafeCell { value: false } };

impl Flag {
    #[inline]
    pub fn borrow(&self) -> LockResult<Guard> {
        let ret = Guard { panicking: Thread::panicking() };
        if unsafe { *self.failed.get() } {
            Err(new_poison_error(ret))
        } else {
            Ok(ret)
        }
    }

    #[inline]
    pub fn done(&self, guard: &Guard) {
        if !guard.panicking && Thread::panicking() {
            unsafe { *self.failed.get() = true; }
        }
    }

    #[inline]
    pub fn get(&self) -> bool {
        unsafe { *self.failed.get() }
    }
}

#[allow(missing_copy_implementations)]
pub struct Guard {
    panicking: bool,
}

/// A type of error which can be returned whenever a lock is acquired.
///
/// Both Mutexes and RwLocks are poisoned whenever a task fails while the lock
/// is held. The precise semantics for when a lock is poisoned is documented on
/// each lock, but once a lock is poisoned then all future acquisitions will
/// return this error.
#[stable]
pub struct PoisonError<T> {
    guard: T,
}

/// An enumeration of possible errors which can occur while calling the
/// `try_lock` method.
#[stable]
pub enum TryLockError<T> {
    /// The lock could not be acquired because another task failed while holding
    /// the lock.
    #[stable]
    Poisoned(PoisonError<T>),
    /// The lock could not be acquired at this time because the operation would
    /// otherwise block.
    #[stable]
    WouldBlock,
}

/// A type alias for the result of a lock method which can be poisoned.
///
/// The `Ok` variant of this result indicates that the primitive was not
/// poisoned, and the `Guard` is contained within. The `Err` variant indicates
/// that the primitive was poisoned. Note that the `Err` variant *also* carries
/// the associated guard, and it can be acquired through the `into_inner`
/// method.
#[stable]
pub type LockResult<Guard> = Result<Guard, PoisonError<Guard>>;

/// A type alias for the result of a nonblocking locking method.
///
/// For more information, see `LockResult`. A `TryLockResult` doesn't
/// necessarily hold the associated guard in the `Err` type as the lock may not
/// have been acquired for other reasons.
#[stable]
pub type TryLockResult<Guard> = Result<Guard, TryLockError<Guard>>;

impl<T> fmt::Show for PoisonError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "poisoned lock: another task failed inside".fmt(f)
    }
}

impl<T> PoisonError<T> {
    /// Consumes this error indicating that a lock is poisoned, returning the
    /// underlying guard to allow access regardless.
    #[stable]
    pub fn into_guard(self) -> T { self.guard }
}

impl<T> FromError<PoisonError<T>> for TryLockError<T> {
    fn from_error(err: PoisonError<T>) -> TryLockError<T> {
        TryLockError::Poisoned(err)
    }
}

impl<T> fmt::Show for TryLockError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TryLockError::Poisoned(ref p) => p.fmt(f),
            TryLockError::WouldBlock => {
                "try_lock failed because the operation would block".fmt(f)
            }
        }
    }
}

pub fn new_poison_error<T>(guard: T) -> PoisonError<T> {
    PoisonError { guard: guard }
}

pub fn map_result<T, U, F>(result: LockResult<T>, f: F)
                           -> LockResult<U>
                           where F: FnOnce(T) -> U {
    match result {
        Ok(t) => Ok(f(t)),
        Err(PoisonError { guard }) => Err(new_poison_error(f(guard)))
    }
}
