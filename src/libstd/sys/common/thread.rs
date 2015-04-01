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

use usize;
use libc;
use thunk::Thunk;
use sys_common::stack;
use sys::stack_overflow;

// This is the starting point of rust os threads. The first thing we do
// is make sure that we don't trigger __morestack (also why this has a
// no_stack_check annotation), and then we extract the main function
// and invoke it.
#[no_stack_check]
pub fn start_thread(main: *mut libc::c_void) {
    unsafe {
        stack::record_os_managed_stack_bounds(0, usize::MAX);
        let _handler = stack_overflow::Handler::new();
        let main: Box<Thunk> = Box::from_raw(main as *mut Thunk);
        main();
    }
}
