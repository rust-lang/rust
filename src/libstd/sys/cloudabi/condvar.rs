// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;
use mem;
use sync::atomic::{AtomicU32, Ordering};
use sys::cloudabi::abi;
use sys::mutex::{self, Mutex};
use sys::time::dur2intervals;
use time::Duration;

extern "C" {
    #[thread_local]
    static __pthread_thread_id: abi::tid;
}

pub struct Condvar {
    condvar: UnsafeCell<AtomicU32>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar {
            condvar: UnsafeCell::new(AtomicU32::new(abi::CONDVAR_HAS_NO_WAITERS.0)),
        }
    }

    pub unsafe fn init(&mut self) {}

    pub unsafe fn notify_one(&self) {
        let condvar = self.condvar.get();
        if (*condvar).load(Ordering::Relaxed) != abi::CONDVAR_HAS_NO_WAITERS.0 {
            let ret = abi::condvar_signal(condvar as *mut abi::condvar, abi::scope::PRIVATE, 1);
            assert_eq!(
                ret,
                abi::errno::SUCCESS,
                "Failed to signal on condition variable"
            );
        }
    }

    pub unsafe fn notify_all(&self) {
        let condvar = self.condvar.get();
        if (*condvar).load(Ordering::Relaxed) != abi::CONDVAR_HAS_NO_WAITERS.0 {
            let ret = abi::condvar_signal(
                condvar as *mut abi::condvar,
                abi::scope::PRIVATE,
                abi::nthreads::max_value(),
            );
            assert_eq!(
                ret,
                abi::errno::SUCCESS,
                "Failed to broadcast on condition variable"
            );
        }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let mutex = mutex::raw(mutex);
        assert_eq!(
            (*mutex).load(Ordering::Relaxed) & !abi::LOCK_KERNEL_MANAGED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            "This lock is not write-locked by this thread"
        );

        // Call into the kernel to wait on the condition variable.
        let condvar = self.condvar.get();
        let subscription = abi::subscription {
            type_: abi::eventtype::CONDVAR,
            union: abi::subscription_union {
                condvar: abi::subscription_condvar {
                    condvar: condvar as *mut abi::condvar,
                    condvar_scope: abi::scope::PRIVATE,
                    lock: mutex as *mut abi::lock,
                    lock_scope: abi::scope::PRIVATE,
                },
            },
            ..mem::zeroed()
        };
        let mut event: abi::event = mem::uninitialized();
        let mut nevents: usize = mem::uninitialized();
        let ret = abi::poll(&subscription, &mut event, 1, &mut nevents);
        assert_eq!(
            ret,
            abi::errno::SUCCESS,
            "Failed to wait on condition variable"
        );
        assert_eq!(
            event.error,
            abi::errno::SUCCESS,
            "Failed to wait on condition variable"
        );
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let mutex = mutex::raw(mutex);
        assert_eq!(
            (*mutex).load(Ordering::Relaxed) & !abi::LOCK_KERNEL_MANAGED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            "This lock is not write-locked by this thread"
        );

        // Call into the kernel to wait on the condition variable.
        let condvar = self.condvar.get();
        let subscriptions = [
            abi::subscription {
                type_: abi::eventtype::CONDVAR,
                union: abi::subscription_union {
                    condvar: abi::subscription_condvar {
                        condvar: condvar as *mut abi::condvar,
                        condvar_scope: abi::scope::PRIVATE,
                        lock: mutex as *mut abi::lock,
                        lock_scope: abi::scope::PRIVATE,
                    },
                },
                ..mem::zeroed()
            },
            abi::subscription {
                type_: abi::eventtype::CLOCK,
                union: abi::subscription_union {
                    clock: abi::subscription_clock {
                        clock_id: abi::clockid::MONOTONIC,
                        timeout: dur2intervals(&dur),
                        ..mem::zeroed()
                    },
                },
                ..mem::zeroed()
            },
        ];
        let mut events: [abi::event; 2] = mem::uninitialized();
        let mut nevents: usize = mem::uninitialized();
        let ret = abi::poll(subscriptions.as_ptr(), events.as_mut_ptr(), 2, &mut nevents);
        assert_eq!(
            ret,
            abi::errno::SUCCESS,
            "Failed to wait on condition variable"
        );
        for i in 0..nevents {
            assert_eq!(
                events[i].error,
                abi::errno::SUCCESS,
                "Failed to wait on condition variable"
            );
            if events[i].type_ == abi::eventtype::CONDVAR {
                return true;
            }
        }
        false
    }

    pub unsafe fn destroy(&self) {
        let condvar = self.condvar.get();
        assert_eq!(
            (*condvar).load(Ordering::Relaxed),
            abi::CONDVAR_HAS_NO_WAITERS.0,
            "Attempted to destroy a condition variable with blocked threads"
        );
    }
}
