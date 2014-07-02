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
// gdb-command:print x
// gdb-check:$1 = 0.5
// gdb-command:print y
// gdb-check:$2 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print *x
// gdb-check:$3 = 29
// gdb-command:print *y
// gdb-check:$4 = 110
// gdb-command:continue

fn some_generic_fun<T1, T2>(a: T1, b: T2) -> (T2, T1) {

    let closure = |x, y| {
        zzz();
        (y, x)
    };

    closure(a, b)
}

fn main() {
    some_generic_fun(0.5f64, 10i);
    some_generic_fun(&29i, box 110i);
}

fn zzz() {()}
