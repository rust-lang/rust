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

// STACK BY REF
// gdb-command:finish
// gdb-command:print *self
// gdb-check:$1 = {x = {8888, -8888}}
// gdb-command:print arg1
// gdb-check:$2 = -1
// gdb-command:print/d arg2
// gdb-check:$3 = -2
// gdb-command:continue

// STACK BY VAL
// gdb-command:finish
// gdb-command:print self
// gdb-check:$4 = {x = {8888, -8888}}
// gdb-command:print arg1
// gdb-check:$5 = -3
// gdb-command:print arg2
// gdb-check:$6 = -4
// gdb-command:continue

// OWNED BY REF
// gdb-command:finish
// gdb-command:print *self
// gdb-check:$7 = {x = 1234.5}
// gdb-command:print arg1
// gdb-check:$8 = -5
// gdb-command:print arg2
// gdb-check:$9 = -6
// gdb-command:continue

// OWNED BY VAL
// gdb-command:finish
// gdb-command:print self
// gdb-check:$10 = {x = 1234.5}
// gdb-command:print arg1
// gdb-check:$11 = -7
// gdb-command:print arg2
// gdb-check:$12 = -8
// gdb-command:continue

// OWNED MOVED
// gdb-command:finish
// gdb-command:print *self
// gdb-check:$13 = {x = 1234.5}
// gdb-command:print arg1
// gdb-check:$14 = -9
// gdb-command:print arg2
// gdb-check:$15 = -10.5
// gdb-command:continue

struct Struct<T> {
    x: T
}

impl<T1> Struct<T1> {

    fn self_by_ref<T2>(&self, arg1: int, arg2: T2) -> int {
        zzz();
        arg1
    }

    fn self_by_val<T2>(self, arg1: int, arg2: T2) -> int {
        zzz();
        arg1
    }

    fn self_owned<T2>(~self, arg1: int, arg2: T2) -> int {
        zzz();
        arg1
    }
}

fn main() {
    let stack = Struct { x: (8888_u32, -8888_i32) };
    let _ = stack.self_by_ref(-1, -2_i8);
    let _ = stack.self_by_val(-3, -4_i16);

    let owned = box Struct { x: 1234.5 };
    let _ = owned.self_by_ref(-5, -6_i32);
    let _ = owned.self_by_val(-7, -8_i64);
    let _ = owned.self_owned(-9, -10.5_f32);
}

fn zzz() {()}
