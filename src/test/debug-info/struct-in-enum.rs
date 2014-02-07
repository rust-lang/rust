// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-tidy-linelength
// xfail-android: FIXME(#10381)

// compile-flags:-Z extra-debug-info
// debugger:set print union on
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print case1
// check:$1 = {{Case1, 0, {x = 2088533116, y = 2088533116, z = 31868}}, {Case1, 0, 8970181431921507452, 31868}}

// debugger:print case2
// check:$2 = {{Case2, 0, {x = 286331153, y = 286331153, z = 4369}}, {Case2, 0, 1229782938247303441, 4369}}

// debugger:print univariant
// check:$3 = {{x = 123, y = 456, z = 789}}

#[allow(unused_variable)];

struct Struct {
    x: u32,
    y: i32,
    z: i16
}

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular {
    Case1(u64, Struct),
    Case2(u64, u64, i16)
}

enum Univariant {
    TheOnlyCase(Struct)
}

fn main() {

    // In order to avoid endianess trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1 = Case1(0, Struct { x: 2088533116, y: 2088533116, z: 31868 });

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2 = Case2(0, 1229782938247303441, 4369);

    let univariant = TheOnlyCase(Struct { x: 123, y: 456, z: 789 });

    zzz();
}

fn zzz() {()}
