// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A "once initialization" primitive
//!
//! This primitive is meant to be used to run one-time initialization. An
//! example use case would be for initializing an FFI library.

use isize;
use marker::Sync;
use mem::drop;
use ops::FnOnce;
use sync::atomic::{AtomicIsize, Ordering, ATOMIC_ISIZE_INIT};
use sync::{StaticMutex, MUTEX_INIT};

/// A synchronization primitive which can be used to run a one-time global
/// initialization. Useful for one-time initialization for FFI or related
/// functionality. This type can only be constructed with the `ONCE_INIT`
/// value.
///
/// # Example
///
/// ```rust
/// use std::sync::{Once, ONCE_INIT};
///
/// static START: Once = ONCE_INIT;
///
/// START.call_once(|| {
///     // run initialization here
/// });
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Once {
    mutex: StaticMutex,
    cnt: AtomicIsize,
    lock_cnt: AtomicIsize,
}

unsafe impl Sync for Once {}

/// Initialization value for static `Once` values.
#[stable(feature = "rust1", since = "1.0.0")]
pub const ONCE_INIT: Once = Once {
    mutex: MUTEX_INIT,
    cnt: ATOMIC_ISIZE_INIT,
    lock_cnt: ATOMIC_ISIZE_INIT,
};

impl Once {
    /// Perform an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling task if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn call_once<F>(&'static self, f: F) where F: FnOnce() {
        // Optimize common path: load is much cheaper than fetch_add.
        if self.cnt.load(Ordering::SeqCst) < 0 {
            return
        }

        // Implementation-wise, this would seem like a fairly trivial primitive.
        // The stickler part is where our mutexes currently require an
        // allocation, and usage of a `Once` shouldn't leak this allocation.
        //
        // This means that there must be a deterministic destroyer of the mutex
        // contained within (because it's not needed after the initialization
        // has run).
        //
        // The general scheme here is to gate all future threads once
        // initialization has completed with a "very negative" count, and to
        // allow through threads to lock the mutex if they see a non negative
        // count. For all threads grabbing the mutex, exactly one of them should
        // be responsible for unlocking the mutex, and this should only be done
        // once everyone else is done with the mutex.
        //
        // This atomicity is achieved by swapping a very negative value into the
        // shared count when the initialization routine has completed. This will
        // read the number of threads which will at some point attempt to
        // acquire the mutex. This count is then squirreled away in a separate
        // variable, and the last person on the way out of the mutex is then
        // responsible for destroying the mutex.
        //
        // It is crucial that the negative value is swapped in *after* the
        // initialization routine has completed because otherwise new threads
        // calling `call_once` will return immediately before the initialization
        // has completed.

        let prev = self.cnt.fetch_add(1, Ordering::SeqCst);
        if prev < 0 {
            // Make sure we never overflow, we'll never have isize::MIN
            // simultaneous calls to `call_once` to make this value go back to 0
            self.cnt.store(isize::MIN, Ordering::SeqCst);
            return
        }

        // If the count is negative, then someone else finished the job,
        // otherwise we run the job and record how many people will try to grab
        // this lock
        let guard = self.mutex.lock();
        if self.cnt.load(Ordering::SeqCst) > 0 {
            f();
            let prev = self.cnt.swap(isize::MIN, Ordering::SeqCst);
            self.lock_cnt.store(prev, Ordering::SeqCst);
        }
        drop(guard);

        // Last one out cleans up after everyone else, no leaks!
        if self.lock_cnt.fetch_add(-1, Ordering::SeqCst) == 1 {
            unsafe { self.mutex.destroy() }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::v1::*;

    use thread;
    use super::{ONCE_INIT, Once};
    use sync::mpsc::channel;

    #[test]
    fn smoke_once() {
        static O: Once = ONCE_INIT;
        let mut a = 0;
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static O: Once = ONCE_INIT;
        static mut run: bool = false;

        let (tx, rx) = channel();
        for _ in 0..10 {
            let tx = tx.clone();
            thread::spawn(move|| {
                for _ in 0..4 { thread::yield_now() }
                unsafe {
                    O.call_once(|| {
                        assert!(!run);
                        run = true;
                    });
                    assert!(run);
                }
                tx.send(()).unwrap();
            });
        }

        unsafe {
            O.call_once(|| {
                assert!(!run);
                run = true;
            });
            assert!(run);
        }

        for _ in 0..10 {
            rx.recv().unwrap();
        }
    }
}
