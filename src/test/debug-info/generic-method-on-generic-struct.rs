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

// STACK BY REF
// debugger:finish
// debugger:print *self
// check:$1 = {x = {8888, -8888}}
// debugger:print arg1
// check:$2 = -1
// debugger:print/d arg2
// check:$3 = -2
// debugger:continue

// STACK BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {x = {8888, -8888}}
// debugger:print arg1
// check:$4 = -3
// debugger:print arg2
// check:$5 = -4
// debugger:continue

// OWNED BY REF
// debugger:finish
// debugger:print *self
// check:$6 = {x = 1234.5}
// debugger:print arg1
// check:$7 = -5
// debugger:print arg2
// check:$8 = -6
// debugger:continue

// OWNED BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {x = 1234.5}
// debugger:print arg1
// check:$9 = -7
// debugger:print arg2
// check:$10 = -8
// debugger:continue

// OWNED MOVED
// debugger:finish
// debugger:print *self
// check:$11 = {x = 1234.5}
// debugger:print arg1
// check:$12 = -9
// debugger:print arg2
// check:$13 = -10.5
// debugger:continue

// MANAGED BY REF
// debugger:finish
// debugger:print *self
// check:$14 = {x = -1}
// debugger:print arg1
// check:$15 = -11
// debugger:print arg2
// check:$16 = -12.5
// debugger:continue

// MANAGED BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {x = -1}
// debugger:print arg1
// check:$17 = -13
// debugger:print *arg2
// check:$18 = {-14, 14}
// debugger:continue

// MANAGED SELF
// debugger:finish
// debugger:print self->val
// check:$19 = {x = -1}
// debugger:print arg1
// check:$20 = -15
// debugger:print *arg2
// check:$21 = {-16, 16.5}
// debugger:continue

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

    fn self_managed<T2>(@self, arg1: int, arg2: T2) -> int {
        zzz();
        arg1
    }
}

fn main() {
    let stack = Struct { x: (8888_u32, -8888_i32) };
    let _ = stack.self_by_ref(-1, -2_i8);
    let _ = stack.self_by_val(-3, -4_i16);

    let owned = ~Struct { x: 1234.5 };
    let _ = owned.self_by_ref(-5, -6_i32);
    let _ = owned.self_by_val(-7, -8_i64);
    let _ = owned.self_owned(-9, -10.5_f32);

    let managed = @Struct { x: -1_i16 };
    let _ = managed.self_by_ref(-11, -12.5_f64);
    let _ = managed.self_by_val(-13, &(-14, 14));
    let _ = managed.self_managed(-15, &(-16, 16.5));
}

fn zzz() {()}
