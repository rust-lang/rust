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

use std::int;
use std::sync::atomics;

use sync::mutex::{StaticMutex, MUTEX_INIT};

/// A type which can be used to run a one-time global initialization. This type
/// is *unsafe* to use because it is built on top of the `Mutex` in this module.
/// It does not know whether the currently running task is in a green or native
/// context, and a blocking mutex should *not* be used under normal
/// circumstances on a green task.
///
/// Despite its unsafety, it is often useful to have a one-time initialization
/// routine run for FFI bindings or related external functionality. This type
/// can only be statically constructed with the `ONCE_INIT` value.
///
/// # Example
///
/// ```rust
/// use sync::one::{Once, ONCE_INIT};
///
/// static mut START: Once = ONCE_INIT;
/// unsafe {
///     START.doit(|| {
///         // run initialization here
///     });
/// }
/// ```
pub struct Once {
    mutex: StaticMutex,
    state: atomics::AtomicInt,
    lock_cnt: atomics::AtomicInt,
}

/// Initialization value for static `Once` values.
pub static ONCE_INIT: Once = Once {
    mutex: MUTEX_INIT,
    state: atomics::INIT_ATOMIC_INT,
    lock_cnt: atomics::INIT_ATOMIC_INT,
};

impl Once {
    /// Perform an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `doit` has been called, and
    /// otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling *os thread* if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified).
    #[inline]
    pub fn doit(&self, f: ||) {
        // Implementation-wise, this would seem like a fairly trivial primitive.
        // The stickler part is where our mutexes currently require an
        // allocation, and usage of a `Once` shouldn't leak this allocation.
        //
        // This means that there must be a deterministic destroyer of the mutex
        // contained within (because it's not needed after the initialization
        // has run).
        //
        // The general algorithm here is we keep a state value that indicates whether
        // we've finished the work. This lets us avoid the expensive atomic
        // read-modify-writes and mutex if there's no need. If we haven't finished
        // the work, then we use a second value to keep track of how many outstanding
        // threads are trying to use the mutex. When the last thread releases the
        // mutex it drops the lock count to a "very negative" value to indicate to
        // other threads that the mutex is gone.

        let state = self.state.load(atomics::Acquire);
        if state != 2 {
            self.doit_slow(f)
        }
    }

    #[inline(never)]
    fn doit_slow(&self, f: ||) {
        // If the count is negative, then someone else finished the job,
        // otherwise we run the job and record how many people will try to grab
        // this lock
        if self.lock_cnt.fetch_add(1, atomics::Acquire) < 0 {
            // Make sure we never overflow.
            self.lock_cnt.store(int::MIN, atomics::Relaxed);
            return
        }

        let guard = self.mutex.lock();
        if self.state.compare_and_swap(0, 1, atomics::Relaxed) == 0 {
            // we're the first one here
            f();
            self.state.store(2, atomics::Release);
        }
        drop(guard);

        // Last one out cleans up after everyone else, no leaks!
        if self.lock_cnt.fetch_add(-1, atomics::Release) == 1 {
            // we just decremented it to 0, now make sure we can drop it to int::MIN.
            // If this fails, someone else just waltzed in and took the mutex
            if self.lock_cnt.compare_and_swap(0, int::MIN, atomics::AcqRel) == 0 {
                // good, we really were the last ones out
                unsafe { self.mutex.destroy() }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{ONCE_INIT, Once};
    use std::task;

    #[test]
    fn smoke_once() {
        static mut o: Once = ONCE_INIT;
        let mut a = 0;
        unsafe { o.doit(|| a += 1); }
        assert_eq!(a, 1);
        unsafe { o.doit(|| a += 1); }
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static mut o: Once = ONCE_INIT;
        static mut run: bool = false;

        let (tx, rx) = channel();
        for _ in range(0, 10) {
            let tx = tx.clone();
            spawn(proc() {
                for _ in range(0, 4) { task::deschedule() }
                unsafe {
                    o.doit(|| {
                        assert!(!run);
                        run = true;
                    });
                    assert!(run);
                }
                tx.send(());
            });
        }

        unsafe {
            o.doit(|| {
                assert!(!run);
                run = true;
            });
            assert!(run);
        }

        for _ in range(0, 10) {
            rx.recv();
        }
    }
}
