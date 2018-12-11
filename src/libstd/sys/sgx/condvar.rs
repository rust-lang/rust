// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::mutex::Mutex;
use time::Duration;

use super::waitqueue::{WaitVariable, WaitQueue, SpinMutex};

pub struct Condvar {
    inner: SpinMutex<WaitVariable<()>>,
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: SpinMutex::new(WaitVariable::new(())) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn notify_one(&self) {
        let _ = WaitQueue::notify_one(self.inner.lock());
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let _ = WaitQueue::notify_all(self.inner.lock());
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let guard = self.inner.lock();
        mutex.unlock();
        WaitQueue::wait(guard);
        mutex.lock()
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        panic!("timeout not supported in SGX");
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}
