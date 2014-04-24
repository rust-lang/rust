// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print *the_a_ref
// gdb-check:$1 = TheA

// gdb-command:print *the_b_ref
// gdb-check:$2 = TheB

// gdb-command:print *the_c_ref
// gdb-check:$3 = TheC

#![allow(unused_variable)]

enum ABC { TheA, TheB, TheC }

fn main() {
    let the_a = TheA;
    let the_a_ref: &ABC = &the_a;

    let the_b = TheB;
    let the_b_ref: &ABC = &the_b;

    let the_c = TheC;
    let the_c_ref: &ABC = &the_c;

    zzz();
}

fn zzz() {()}
