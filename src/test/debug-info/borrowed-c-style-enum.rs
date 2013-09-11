// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print *the_a_ref
// check:$1 = TheA

// debugger:print *the_b_ref
// check:$2 = TheB

// debugger:print *the_c_ref
// check:$3 = TheC

#[allow(unused_variable)];

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
