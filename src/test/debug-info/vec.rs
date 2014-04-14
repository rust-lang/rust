// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:set print pretty off
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:print a
// check:$1 = {1, 2, 3}
// debugger:print vec::VECT
// check:$2 = {4, 5, 6}

#![allow(unused_variable)]

static mut VECT: [i32, ..3] = [1, 2, 3];

fn main() {
    let a = [1, 2, 3];

    unsafe {
        VECT[0] = 4;
        VECT[1] = 5;
        VECT[2] = 6;
    }

    zzz();
}

fn zzz() {()}
