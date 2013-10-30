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
// check:$1 = {{Variant2, [...]}, {Variant2, 117901063}}
// debugger:print arg1
// check:$2 = -1
// debugger:print arg2
// check:$3 = -2
// debugger:continue

// STACK BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {{Variant2, [...]}, {Variant2, 117901063}}
// debugger:print arg1
// check:$4 = -3
// debugger:print arg2
// check:$5 = -4
// debugger:continue

// OWNED BY REF
// debugger:finish
// debugger:print *self
// check:$6 = {{Variant1, x = 1799, y = 1799}, {Variant1, [...]}}
// debugger:print arg1
// check:$7 = -5
// debugger:print arg2
// check:$8 = -6
// debugger:continue

// OWNED BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {{Variant1, x = 1799, y = 1799}, {Variant1, [...]}}
// debugger:print arg1
// check:$9 = -7
// debugger:print arg2
// check:$10 = -8
// debugger:continue

// OWNED MOVED
// debugger:finish
// debugger:print *self
// check:$11 = {{Variant1, x = 1799, y = 1799}, {Variant1, [...]}}
// debugger:print arg1
// check:$12 = -9
// debugger:print arg2
// check:$13 = -10
// debugger:continue

// MANAGED BY REF
// debugger:finish
// debugger:print *self
// check:$14 = {{Variant2, [...]}, {Variant2, 117901063}}
// debugger:print arg1
// check:$15 = -11
// debugger:print arg2
// check:$16 = -12
// debugger:continue

// MANAGED BY VAL
// debugger:finish
// d ebugger:print self -- ignored for now because of issue #8512
// c heck:$X = {{Variant2, [...]}, {Variant2, 117901063}}
// debugger:print arg1
// check:$17 = -13
// debugger:print arg2
// check:$18 = -14
// debugger:continue

// MANAGED SELF
// debugger:finish
// debugger:print self->val
// check:$19 = {{Variant2, [...]}, {Variant2, 117901063}}
// debugger:print arg1
// check:$20 = -15
// debugger:print arg2
// check:$21 = -16
// debugger:continue

#[feature(struct_variant)];

enum Enum {
    Variant1 { x: u16, y: u16 },
    Variant2 (u32)
}

impl Enum {

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

    fn self_managed(@self, arg1: int, arg2: int) -> int {
        zzz();
        arg1 + arg2
    }
}

fn main() {
    let stack = Variant2(117901063);
    let _ = stack.self_by_ref(-1, -2);
    let _ = stack.self_by_val(-3, -4);

    let owned = ~Variant1{ x: 1799, y: 1799 };
    let _ = owned.self_by_ref(-5, -6);
    let _ = owned.self_by_val(-7, -8);
    let _ = owned.self_owned(-9, -10);

    let managed = @Variant2(117901063);
    let _ = managed.self_by_ref(-11, -12);
    let _ = managed.self_by_val(-13, -14);
    let _ = managed.self_managed(-15, -16);
}

fn zzz() {()}
