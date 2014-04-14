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
// debugger:finish

// debugger:print *the_a_ref
// check:$1 = {{TheA, x = 0, y = 8970181431921507452}, {TheA, 0, 2088533116, 2088533116}}

// debugger:print *the_b_ref
// check:$2 = {{TheB, x = 0, y = 1229782938247303441}, {TheB, 0, 286331153, 286331153}}

// debugger:print *univariant_ref
// check:$3 = {4820353753753434}

#![allow(unused_variable)]
#![feature(struct_variant)]

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum ABC {
    TheA { x: i64, y: i64 },
    TheB (i64, i32, i32),
}

// This is a special case since it does not have the implicit discriminant field.
enum Univariant {
    TheOnlyCase(i64)
}

fn main() {

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let the_a = TheA { x: 0, y: 8970181431921507452 };
    let the_a_ref: &ABC = &the_a;

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let the_b = TheB (0, 286331153, 286331153);
    let the_b_ref: &ABC = &the_b;

    let univariant = TheOnlyCase(4820353753753434);
    let univariant_ref: &Univariant = &univariant;

    zzz();
}

fn zzz() {()}
