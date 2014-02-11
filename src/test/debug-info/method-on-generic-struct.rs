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
// debugger:rbreak zzz
// debugger:run

// STACK BY REF
// debugger:finish
// debugger:print *self
// check:$1 = {x = {8888, -8888}}
// debugger:print arg1
// check:$2 = -1
// debugger:print arg2
// check:$3 = -2
// debugger:continue

// STACK BY VAL
// debugger:finish
// debugger:print self
// check:$4 = {x = {8888, -8888}}
// debugger:print arg1
// check:$5 = -3
// debugger:print arg2
// check:$6 = -4
// debugger:continue

// OWNED BY REF
// debugger:finish
// debugger:print *self
// check:$7 = {x = 1234.5}
// debugger:print arg1
// check:$8 = -5
// debugger:print arg2
// check:$9 = -6
// debugger:continue

// OWNED BY VAL
// debugger:finish
// debugger:print self
// check:$10 = {x = 1234.5}
// debugger:print arg1
// check:$11 = -7
// debugger:print arg2
// check:$12 = -8
// debugger:continue

// OWNED MOVED
// debugger:finish
// debugger:print *self
// check:$13 = {x = 1234.5}
// debugger:print arg1
// check:$14 = -9
// debugger:print arg2
// check:$15 = -10
// debugger:continue

struct Struct<T> {
    x: T
}

impl<T> Struct<T> {

    fn self_by_ref(&self, arg1: int, arg2: int) -> int {
        zzz();
        arg1 + arg2
    }

    fn self_by_val(self, arg1: int, arg2: int) -> int {
        zzz();
        arg1 + arg2
    }

    fn self_owned(~self, arg1: int, arg2: int) -> int {
        zzz();
        arg1 + arg2
    }
}

fn main() {
    let stack = Struct { x: (8888_u32, -8888_i32) };
    let _ = stack.self_by_ref(-1, -2);
    let _ = stack.self_by_val(-3, -4);

    let owned = ~Struct { x: 1234.5 };
    let _ = owned.self_by_ref(-5, -6);
    let _ = owned.self_by_val(-7, -8);
    let _ = owned.self_owned(-9, -10);
}

fn zzz() {()}
