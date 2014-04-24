// ignore-test

// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print arg1
// gdb-check:$1 = 1000
// gdb-command:print *arg2
// gdb-check:$2 = {1, 2.5}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print arg1
// gdb-check:$3 = 2000
// gdb-command:print *arg2
// gdb-check:$4 = {3.5, {4, 5, 6}}
// gdb-command:continue


struct Struct {
    x: int
}

trait Trait<T1> {
    fn generic_static_default_method<T2>(arg1: int, arg2: &(T1, T2)) -> int {
        zzz();
        arg1
    }
}

impl<T> Trait<T> for Struct {}

fn main() {

    // Is this really how to use these?
    Trait::generic_static_default_method::<int, Struct, float>(1000, &(1, 2.5));
    Trait::generic_static_default_method::<float, Struct, (int, int, int)>(2000, &(3.5, (4, 5, 6)));

}

fn zzz() {()}
