// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10381)

// compile-flags:-g
// debugger:run

// Test whether compiling a recursive enum definition crashes debug info generation. The test case
// is taken from issue #11083.

#[allow(unused_variable)];

pub struct Window<'a> {
    callbacks: WindowCallbacks<'a>
}

struct WindowCallbacks<'a> {
    pos_callback: Option<WindowPosCallback<'a>>,
}

pub type WindowPosCallback<'a> = 'a |&Window, i32, i32|;

fn main() {
    let x = WindowCallbacks { pos_callback: None };
}
