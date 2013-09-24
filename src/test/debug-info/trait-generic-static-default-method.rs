// xfail-test

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
// debugger:print arg1
// check:$1 = 1000
// debugger:print arg2
// check:$2 = 0.5
// debugger:continue

// debugger:finish
// debugger:print arg1
// check:$3 = 2000
// debugger:print *arg2
// check:$4 = {1, 2, 3}
// debugger:continue


struct Struct {
    x: int
}

trait Trait {
    fn generic_static_default_method<T>(arg1: int, arg2: T) -> int {
        zzz();
        arg1
    }
}

impl Trait for Struct {}

fn main() {

    // Is this really how to use these?
    Trait::generic_static_default_method::<Struct, float>(1000, 0.5);
    Trait::generic_static_default_method::<Struct, &(int, int, int)>(2000, &(1, 2, 3));

}

fn zzz() {()}
