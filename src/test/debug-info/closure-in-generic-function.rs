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

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run

// debugger:finish
// debugger:print x
// check:$1 = 0.5
// debugger:print y
// check:$2 = 10
// debugger:continue

// debugger:finish
// debugger:print *x
// check:$3 = 29
// debugger:print *y
// check:$4 = 110
// debugger:continue

fn some_generic_fun<T1, T2>(a: T1, b: T2) -> (T2, T1) {

    let closure = |x, y| {
        zzz();
        (y, x)
    };

    closure(a, b)
}

fn main() {
    some_generic_fun(0.5, 10);
    some_generic_fun(&29, ~110);
}

fn zzz() {()}
