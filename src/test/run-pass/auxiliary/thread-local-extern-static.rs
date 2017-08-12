// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(cfg_target_thread_local, const_fn, thread_local)]
#![crate_type = "lib"]

#[cfg(target_thread_local)]
use std::cell::Cell;

#[no_mangle]
#[cfg(target_thread_local)]
#[thread_local]
pub static FOO: Cell<u32> = Cell::new(3);
