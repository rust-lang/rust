// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rt::sched::Scheduler;
use rt::local_ptr;

pub trait Local {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn exists() -> bool;
    fn borrow(f: &fn(&mut Self));
    unsafe fn unsafe_borrow() -> *mut Self;
}

impl Local for Scheduler {
    fn put(value: ~Scheduler) { unsafe { local_ptr::put(value) }}
    fn take() -> ~Scheduler { unsafe { local_ptr::take() } }
    fn exists() -> bool { local_ptr::exists() }
    fn borrow(f: &fn(&mut Scheduler)) { unsafe { local_ptr::borrow(f) } }
    unsafe fn unsafe_borrow() -> *mut Scheduler { local_ptr::unsafe_borrow() }
}

#[cfg(test)]
mod test {
    use rt::sched::Scheduler;
    use rt::uv::uvio::UvEventLoop;
    use super::*;

    #[test]
    fn thread_local_scheduler_smoke_test() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
    }

    #[test]
    fn thread_local_scheduler_two_instances() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
    }

    #[test]
    fn borrow_smoke_test() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        unsafe {
            let _scheduler: *mut Scheduler = Local::unsafe_borrow();
        }
        let _scheduler: ~Scheduler = Local::take();
    }
}