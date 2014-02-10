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

// STRUCT
// debugger:finish
// debugger:print arg1
// check:$1 = 1
// debugger:print arg2
// check:$2 = 2
// debugger:continue

// ENUM
// debugger:finish
// debugger:print arg1
// check:$3 = -3
// debugger:print arg2
// check:$4 = 4.5
// debugger:print arg3
// check:$5 = 5
// debugger:continue

#[feature(struct_variant)];

struct Struct {
    x: int
}

impl Struct {

    fn static_method<T1, T2>(arg1: T1, arg2: T2) -> int {
        zzz();
        return 0;
    }
}

enum Enum {
    Variant1 { x: int },
    Variant2,
    Variant3(f64, int, char),
}

impl Enum {

    fn static_method<T1, T2, T3>(arg1: T1, arg2: T2, arg3: T3) -> int {
        zzz();
        return 1;
    }
}

fn main() {
    Struct::static_method(1, 2);
    Enum::static_method(-3, 4.5, 5);
}

fn zzz() {()}
