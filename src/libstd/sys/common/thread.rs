// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use boxed::Box;
use mem;
use uint;
use libc;
use thunk::Thunk;
use sys_common::stack;
use sys::{thread, stack_overflow};

// This is the starting point of rust os threads. The first thing we do
// is make sure that we don't trigger __morestack (also why this has a
// no_stack_check annotation), and then we extract the main function
// and invoke it.
#[no_stack_check]
pub fn start_thread(main: *mut libc::c_void) -> thread::rust_thread_return {
    unsafe {
        stack::record_os_managed_stack_bounds(0, uint::MAX);
        let handler = stack_overflow::Handler::new();
        let f: Box<Thunk> = mem::transmute(main);
        f.invoke(());
        drop(handler);
        mem::transmute(0 as thread::rust_thread_return)
    }
}
