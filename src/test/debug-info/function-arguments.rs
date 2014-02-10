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
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print x
// check:$1 = 111102
// debugger:print y
// check:$2 = true

// debugger:continue
// debugger:finish

// debugger:print a
// check:$3 = 2000
// debugger:print b
// check:$4 = 3000

fn main() {

    fun(111102, true);
    nested(2000, 3000);

    fn nested(a: i32, b: i64) -> (i32, i64) {
        zzz();
        (a, b)
    }
}

fn fun(x: int, y: bool) -> (int, bool) {
    zzz();

    (x, y)
}

fn zzz() {()}
