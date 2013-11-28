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

// debugger:print constant
// check:$1 = 1
// debugger:print a_struct
// check:$2 = {a = -2, b = 3.5, c = 4}
// debugger:print *owned
// check:$3 = 5

#[allow(unused_variable)];

struct Struct {
    a: int,
    b: f64,
    c: uint
}

fn main() {
    let constant = 1;

    let a_struct = Struct {
        a: -2,
        b: 3.5,
        c: 4
    };

    let owned = ~5;

    let closure: proc() = proc() {
        zzz();
        do_something(&constant, &a_struct.a, owned);
    };

    closure();
}

fn do_something(_: &int, _:&int, _:&int) {

}

fn zzz() {()}
