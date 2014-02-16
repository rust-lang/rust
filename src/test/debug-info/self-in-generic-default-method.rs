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
// check:$1 = {x = 987}
// debugger:print arg1
// check:$2 = -1
// debugger:print/d arg2
// check:$3 = -2
// debugger:continue

// STACK BY VAL
// debugger:finish
// debugger:print self
// check:$4 = {x = 987}
// debugger:print arg1
// check:$5 = -3
// debugger:print arg2
// check:$6 = -4
// debugger:continue

// OWNED BY REF
// debugger:finish
// debugger:print *self
// check:$7 = {x = 879}
// debugger:print arg1
// check:$8 = -5
// debugger:print arg2
// check:$9 = -6
// debugger:continue

// OWNED BY VAL
// debugger:finish
// debugger:print self
// check:$10 = {x = 879}
// debugger:print arg1
// check:$11 = -7
// debugger:print arg2
// check:$12 = -8
// debugger:continue

// OWNED MOVED
// debugger:finish
// debugger:print *self
// check:$13 = {x = 879}
// debugger:print arg1
// check:$14 = -9
// debugger:print arg2
// check:$15 = -10.5
// debugger:continue

struct Struct {
    x: int
}

trait Trait {

    fn self_by_ref<T>(&self, arg1: int, arg2: T) -> int {
        zzz();
        arg1
    }

    fn self_by_val<T>(self, arg1: int, arg2: T) -> int {
        zzz();
        arg1
    }

    fn self_owned<T>(~self, arg1: int, arg2: T) -> int {
        zzz();
        arg1
    }
}

impl Trait for Struct {}

fn main() {
    let stack = Struct { x: 987 };
    let _ = stack.self_by_ref(-1, -2_i8);
    let _ = stack.self_by_val(-3, -4_i16);

    let owned = ~Struct { x: 879 };
    let _ = owned.self_by_ref(-5, -6_i32);
    let _ = owned.self_by_val(-7, -8_i64);
    let _ = owned.self_owned(-9, -10.5_f32);
}

fn zzz() {()}
