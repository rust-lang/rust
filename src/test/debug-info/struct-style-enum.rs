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
// debugger:set print union on
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print case1
// check:$1 = {{Case1, a = 0, b = 31868, c = 31868, d = 31868, e = 31868}, {Case1, a = 0, b = 2088533116, c = 2088533116}, {Case1, a = 0, b = 8970181431921507452}}

// debugger:print case2
// check:$2 = {{Case2, a = 0, b = 4369, c = 4369, d = 4369, e = 4369}, {Case2, a = 0, b = 286331153, c = 286331153}, {Case2, a = 0, b = 1229782938247303441}}

// debugger:print case3
// check:$3 = {{Case3, a = 0, b = 22873, c = 22873, d = 22873, e = 22873}, {Case3, a = 0, b = 1499027801, c = 1499027801}, {Case3, a = 0, b = 6438275382588823897}}

// debugger:print univariant
// check:$4 = {a = -1}

#[allow(unused_variable)];

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular {
    Case1 { a: u64, b: u16, c: u16, d: u16, e: u16},
    Case2 { a: u64, b: u32, c: u32},
    Case3 { a: u64, b: u64 }
}

enum Univariant {
    TheOnlyCase { a: i64 }
}

fn main() {

    // In order to avoid endianess trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1 = Case1 { a: 0, b: 31868, c: 31868, d: 31868, e: 31868 };

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2 = Case2 { a: 0, b: 286331153, c: 286331153 };

    // 0b0101100101011001010110010101100101011001010110010101100101011001 = 6438275382588823897
    // 0b01011001010110010101100101011001 = 1499027801
    // 0b0101100101011001 = 22873
    // 0b01011001 = 89
    let case3 = Case3 { a: 0, b: 6438275382588823897 };

    let univariant = TheOnlyCase { a: -1 };

    zzz();
}

fn zzz() {()}
