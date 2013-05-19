// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Access to the thread-local Scheduler

use prelude::*;
use ptr::mut_null;
use libc::c_void;
use cast;
use cell::Cell;

use rt::sched::Scheduler;
use rt::rtio::{EventLoop, IoFactoryObject};
use unstable::finally::Finally;
use rt::local_ptr;
use tls = rt::thread_local_storage;
use rt::local::Local;

#[cfg(test)] use rt::uv::uvio::UvEventLoop;

pub unsafe fn unsafe_borrow_io() -> *mut IoFactoryObject {
    let sched = Local::unsafe_borrow::<Scheduler>();
    let io: *mut IoFactoryObject = (*sched).event_loop.io().unwrap();
    return io;
}
