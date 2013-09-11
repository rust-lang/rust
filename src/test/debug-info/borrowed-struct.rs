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

// debugger:print *stack_val_ref
// check:$1 = {x = 10, y = 23.5}

// debugger:print *stack_val_interior_ref_1
// check:$2 = 10

// debugger:print *stack_val_interior_ref_2
// check:$3 = 23.5

// debugger:print *ref_to_unnamed
// check:$4 = {x = 11, y = 24.5}

// debugger:print *managed_val_ref
// check:$5 = {x = 12, y = 25.5}

// debugger:print *managed_val_interior_ref_1
// check:$6 = 12

// debugger:print *managed_val_interior_ref_2
// check:$7 = 25.5

// debugger:print *unique_val_ref
// check:$8 = {x = 13, y = 26.5}

// debugger:print *unique_val_interior_ref_1
// check:$9 = 13

// debugger:print *unique_val_interior_ref_2
// check:$10 = 26.5

#[allow(unused_variable)];

struct SomeStruct {
    x: int,
    y: f64
}

fn main() {
    let stack_val: SomeStruct = SomeStruct { x: 10, y: 23.5 };
    let stack_val_ref: &SomeStruct = &stack_val;
    let stack_val_interior_ref_1: &int = &stack_val.x;
    let stack_val_interior_ref_2: &f64 = &stack_val.y;
    let ref_to_unnamed: &SomeStruct = &SomeStruct { x: 11, y: 24.5 };

    let managed_val = @SomeStruct { x: 12, y: 25.5 };
    let managed_val_ref: &SomeStruct = managed_val;
    let managed_val_interior_ref_1: &int = &managed_val.x;
    let managed_val_interior_ref_2: &f64 = &managed_val.y;

    let unique_val = ~SomeStruct { x: 13, y: 26.5 };
    let unique_val_ref: &SomeStruct = unique_val;
    let unique_val_interior_ref_1: &int = &unique_val.x;
    let unique_val_interior_ref_2: &f64 = &unique_val.y;

    zzz();
}

fn zzz() {()}
