// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::boxed::FnBox;
use libc;
use sys::stack_overflow;

pub unsafe fn start_thread(main: *mut libc::c_void) {
    // Next, set up our stack overflow handler which may get triggered if we run
    // out of stack.
    let _handler = stack_overflow::Handler::new();

    // Finally, let's run some code.
    Box::from_raw(main as *mut Box<FnBox()>)()
}
