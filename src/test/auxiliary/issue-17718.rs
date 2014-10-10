// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sync::atomic;

pub const C1: uint = 1;
pub const C2: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;
pub const C3: fn() = foo;
pub const C4: uint = C1 * C1 + C1 / C1;
pub const C5: &'static uint = &C4;

pub static S1: uint = 3;
pub static S2: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;

fn foo() {}
