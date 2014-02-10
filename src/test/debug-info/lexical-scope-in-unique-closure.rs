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
// check:$1 = false
// debugger:continue

// debugger:finish
// debugger:print x
// check:$2 = false
// debugger:continue

// debugger:finish
// debugger:print x
// check:$3 = 1000
// debugger:continue

// debugger:finish
// debugger:print x
// check:$4 = 2.5
// debugger:continue

// debugger:finish
// debugger:print x
// check:$5 = true
// debugger:continue

// debugger:finish
// debugger:print x
// check:$6 = false
// debugger:continue

fn main() {

    let x = false;

    zzz();
    sentinel();

    let unique_closure: proc(int) = proc(x) {
        zzz();
        sentinel();

        let x = 2.5;

        zzz();
        sentinel();

        let x = true;

        zzz();
        sentinel();
    };

    zzz();
    sentinel();

    unique_closure(1000);

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
