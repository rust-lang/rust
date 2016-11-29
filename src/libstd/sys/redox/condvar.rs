// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;
use intrinsics::{atomic_cxchg, atomic_xadd, atomic_xchg};
use ptr;
use time::Duration;

use sys::mutex::{mutex_lock, mutex_unlock, Mutex};
use sys::syscall::{futex, FUTEX_WAIT, FUTEX_WAKE, FUTEX_REQUEUE};

pub struct Condvar {
    lock: UnsafeCell<*mut i32>,
    seq: UnsafeCell<i32>
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar {
            lock: UnsafeCell::new(ptr::null_mut()),
            seq: UnsafeCell::new(0)
        }
    }

    #[inline]
    pub unsafe fn init(&self) {
        *self.lock.get() = ptr::null_mut();
        *self.seq.get() = 0;
    }

    #[inline]
    pub fn notify_one(&self) {
        unsafe {
            let seq = self.seq.get();

            atomic_xadd(seq, 1);

            let _ = futex(seq, FUTEX_WAKE, 1, 0, ptr::null_mut());
        }
    }

    #[inline]
    pub fn notify_all(&self) {
        unsafe {
            let lock = self.lock.get();
            let seq = self.seq.get();

            if *lock == ptr::null_mut() {
                return;
            }

            atomic_xadd(seq, 1);

            let _ = futex(seq, FUTEX_REQUEUE, 1, ::usize::MAX, *lock);
        }
    }

    #[inline]
    pub fn wait(&self, mutex: &Mutex) {
        unsafe {
            let lock = self.lock.get();
            let seq = self.seq.get();

            if *lock != mutex.lock.get() {
                if *lock != ptr::null_mut() {
                    panic!("Condvar used with more than one Mutex");
                }

                atomic_cxchg(lock as *mut usize, 0, mutex.lock.get() as usize);
            }

            mutex_unlock(*lock);

            let _ = futex(seq, FUTEX_WAIT, *seq, 0, ptr::null_mut());

            while atomic_xchg(*lock, 2) != 0 {
                let _ = futex(*lock, FUTEX_WAIT, 2, 0, ptr::null_mut());
            }

            mutex_lock(*lock);
        }
    }

    #[inline]
    pub fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        ::sys_common::util::dumb_print(format_args!("condvar wait_timeout\n"));
        unimplemented!();
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        *self.lock.get() = ptr::null_mut();
        *self.seq.get() = 0;
    }
}

unsafe impl Send for Condvar {}

unsafe impl Sync for Condvar {}
