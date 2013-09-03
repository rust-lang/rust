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
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print variable
// check:$1 = 1
// debugger:print constant
// check:$2 = 2
// debugger:print a_struct
// check:$3 = {a = -3, b = 4.5, c = 5}
// debugger:print *struct_ref
// check:$4 = {a = -3, b = 4.5, c = 5}
// debugger:print *owned
// check:$5 = 6
// debugger:print managed->val
// check:$6 = 7
// debugger:print closure_local
// check:$7 = 8
// debugger:continue

// debugger:finish
// debugger:print variable
// check:$8 = 1
// debugger:print constant
// check:$9 = 2
// debugger:print a_struct
// check:$10 = {a = -3, b = 4.5, c = 5}
// debugger:print *struct_ref
// check:$11 = {a = -3, b = 4.5, c = 5}
// debugger:print *owned
// check:$12 = 6
// debugger:print managed->val
// check:$13 = 7
// debugger:print closure_local
// check:$14 = 8
// debugger:continue

#[allow(unused_variable)];

struct Struct {
    a: int,
    b: float,
    c: uint
}

fn main() {
    let mut variable = 1;
    let constant = 2;

    let a_struct = Struct {
        a: -3,
        b: 4.5,
        c: 5
    };

    let struct_ref = &a_struct;
    let owned = ~6;
    let managed = @7;

    let closure = || {
        let closure_local = 8;

        let nested_closure = || {
            zzz();
            variable = constant + a_struct.a + struct_ref.a + *owned + *managed + closure_local;
        };

        zzz();

        nested_closure();
    };

    closure();
}

fn zzz() {()}
