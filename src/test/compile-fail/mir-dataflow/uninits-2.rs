// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// General test of maybe_uninits state computed by MIR dataflow.

#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;
use std::mem::{drop, replace};

struct S(i32);

#[rustc_mir_borrowck]
#[rustc_mir(rustc_peek_maybe_uninit,stop_after_dataflow)]
fn foo(x: &mut S) {
    // `x` is initialized here, so maybe-uninit bit is 0.

    unsafe { rustc_peek(&x) }; //~ ERROR rustc_peek: bit not set

    ::std::mem::drop(x);

    // `x` definitely uninitialized here, so maybe-uninit bit is 1.
    unsafe { rustc_peek(&x) };
}
fn main() {
    foo(&mut S(13));
    foo(&mut S(13));
}
