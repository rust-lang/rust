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
// check:$1 = -1
// debugger:print y
// check:$2 = 1
// debugger:continue

// debugger:finish
// debugger:print x
// check:$3 = -1
// debugger:print y
// check:$4 = 2.5
// debugger:continue

// debugger:finish
// debugger:print x
// check:$5 = -2.5
// debugger:print y
// check:$6 = 1
// debugger:continue

// debugger:finish
// debugger:print x
// check:$7 = -2.5
// debugger:print y
// check:$8 = 2.5
// debugger:continue

fn outer<TA: Clone>(a: TA) {
    inner(a.clone(), 1);
    inner(a.clone(), 2.5);

    fn inner<TX, TY>(x: TX, y: TY) {
        zzz();
    }
}

fn main() {
    outer(-1);
    outer(-2.5);
}

fn zzz() {()}
