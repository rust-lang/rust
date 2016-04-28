// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

struct StructWithDrop<T1, T2> {
    x: T1,
    y: T2,
}

impl<T1, T2> Drop for StructWithDrop<T1, T2> {
    fn drop(&mut self) {}
}

struct StructNoDrop<T1, T2> {
    x: T1,
    y: T2,
}

enum EnumWithDrop<T1, T2> {
    A(T1),
    B(T2)
}

impl<T1, T2> Drop for EnumWithDrop<T1, T2> {
    fn drop(&mut self) {}
}

enum EnumNoDrop<T1, T2> {
    A(T1),
    B(T2)
}


struct NonGenericNoDrop(i32);

struct NonGenericWithDrop(i32);
//~ TRANS_ITEM drop-glue generic_drop_glue::NonGenericWithDrop[0]
//~ TRANS_ITEM drop-glue-contents generic_drop_glue::NonGenericWithDrop[0]

impl Drop for NonGenericWithDrop {
    //~ TRANS_ITEM fn generic_drop_glue::{{impl}}[2]::drop[0]
    fn drop(&mut self) {}
}

//~ TRANS_ITEM fn generic_drop_glue::main[0]
fn main() {
    //~ TRANS_ITEM drop-glue generic_drop_glue::StructWithDrop[0]<i8, char>
    //~ TRANS_ITEM drop-glue-contents generic_drop_glue::StructWithDrop[0]<i8, char>
    //~ TRANS_ITEM fn generic_drop_glue::{{impl}}[0]::drop[0]<i8, char>
    let _ = StructWithDrop { x: 0i8, y: 'a' }.x;

    //~ TRANS_ITEM drop-glue generic_drop_glue::StructWithDrop[0]<&str, generic_drop_glue::NonGenericNoDrop[0]>
    //~ TRANS_ITEM drop-glue-contents generic_drop_glue::StructWithDrop[0]<&str, generic_drop_glue::NonGenericNoDrop[0]>
    //~ TRANS_ITEM fn generic_drop_glue::{{impl}}[0]::drop[0]<&str, generic_drop_glue::NonGenericNoDrop[0]>
    let _ = StructWithDrop { x: "&str", y: NonGenericNoDrop(0) }.y;

    // Should produce no drop glue
    let _ = StructNoDrop { x: 'a', y: 0u32 }.x;

    // This is supposed to generate drop-glue because it contains a field that
    // needs to be dropped.
    //~ TRANS_ITEM drop-glue generic_drop_glue::StructNoDrop[0]<generic_drop_glue::NonGenericWithDrop[0], f64>
    let _ = StructNoDrop { x: NonGenericWithDrop(0), y: 0f64 }.y;

    //~ TRANS_ITEM drop-glue generic_drop_glue::EnumWithDrop[0]<i32, i64>
    //~ TRANS_ITEM drop-glue-contents generic_drop_glue::EnumWithDrop[0]<i32, i64>
    //~ TRANS_ITEM fn generic_drop_glue::{{impl}}[1]::drop[0]<i32, i64>
    let _ = match EnumWithDrop::A::<i32, i64>(0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as i32
    };

    //~ TRANS_ITEM drop-glue generic_drop_glue::EnumWithDrop[0]<f64, f32>
    //~ TRANS_ITEM drop-glue-contents generic_drop_glue::EnumWithDrop[0]<f64, f32>
    //~ TRANS_ITEM fn generic_drop_glue::{{impl}}[1]::drop[0]<f64, f32>
    let _ = match EnumWithDrop::B::<f64, f32>(1.0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as f64
    };

    let _ = match EnumNoDrop::A::<i32, i64>(0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as i32
    };

    let _ = match EnumNoDrop::B::<f64, f32>(1.0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as f64
    };
}

//~ TRANS_ITEM drop-glue i8
