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
    fn put_local(value: ~Self);
    fn take_local() -> ~Self;
    fn exists_local() -> bool;
    fn borrow_local(f: &fn(&mut Self));
    unsafe fn unsafe_borrow_local() -> *mut Self;
}

impl Local for Scheduler {
    fn put_local(value: ~Scheduler) { unsafe { local_ptr::put(value) }}
    fn take_local() -> ~Scheduler { unsafe { local_ptr::take() } }
    fn exists_local() -> bool { local_ptr::exists() }
    fn borrow_local(f: &fn(&mut Scheduler)) { unsafe { local_ptr::borrow(f) } }
    unsafe fn unsafe_borrow_local() -> *mut Scheduler { local_ptr::unsafe_borrow() }
}