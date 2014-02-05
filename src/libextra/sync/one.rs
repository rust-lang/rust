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
/// use extra::sync::one::{Once, ONCE_INIT};
///
/// static mut START: Once = ONCE_INIT;
/// unsafe {
///     START.doit(|| {
///         // run initialization here
///     });
/// }
/// ```
pub struct Once {
    priv mutex: StaticMutex,
    priv cnt: atomics::AtomicInt,
    priv lock_cnt: atomics::AtomicInt,
}

/// Initialization value for static `Once` values.
pub static ONCE_INIT: Once = Once {
    mutex: MUTEX_INIT,
    cnt: atomics::INIT_ATOMIC_INT,
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
    pub fn doit(&mut self, f: ||) {
        // Implementation-wise, this would seem like a fairly trivial primitive.
        // The stickler part is where our mutexes currently require an
        // allocation, and usage of a `Once` should't leak this allocation.
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
        // calling `doit` will return immediately before the initialization has
        // completed.

        let prev = self.cnt.fetch_add(1, atomics::SeqCst);
        if prev < 0 {
            // Make sure we never overflow, we'll never have int::MIN
            // simultaneous calls to `doit` to make this value go back to 0
            self.cnt.store(int::MIN, atomics::SeqCst);
            return
        }

        // If the count is negative, then someone else finished the job,
        // otherwise we run the job and record how many people will try to grab
        // this lock
        {
            let _guard = self.mutex.lock();
            if self.cnt.load(atomics::SeqCst) > 0 {
                f();
                let prev = self.cnt.swap(int::MIN, atomics::SeqCst);
                self.lock_cnt.store(prev, atomics::SeqCst);
            }
        }

        // Last one out cleans up after everyone else, no leaks!
        if self.lock_cnt.fetch_add(-1, atomics::SeqCst) == 1 {
            unsafe { self.mutex.destroy() }
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

        let (p, c) = SharedChan::new();
        for _ in range(0, 10) {
            let c = c.clone();
            spawn(proc() {
                for _ in range(0, 4) { task::deschedule() }
                unsafe {
                    o.doit(|| {
                        assert!(!run);
                        run = true;
                    });
                    assert!(run);
                }
                c.send(());
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
            p.recv();
        }
    }
}
