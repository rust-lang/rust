// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android

// compile-flags:-Z extra-debug-info
// debugger:set print pretty off
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print no_padding1
// check:$1 = {x = {0, 1, 2}, y = -3, z = {4.5, 5.5}}
// debugger:print no_padding2
// check:$2 = {x = {6, 7, 8}, y = {{9, 10}, {11, 12}}}

// debugger:print struct_internal_padding
// check:$3 = {x = {13, 14}, y = {15, 16}}

// debugger:print single_vec
// check:$4 = {x = {17, 18, 19, 20, 21}}

// debugger:print struct_padded_at_end
// check:$5 = {x = {22, 23}, y = {24, 25}}

#[allow(unused_variable)];

struct NoPadding1 {
    x: [u32, ..3],
    y: i32,
    z: [f32, ..2]
}

struct NoPadding2 {
    x: [u32, ..3],
    y: [[u32, ..2], ..2]
}

struct StructInternalPadding {
    x: [i16, ..2],
    y: [i64, ..2]
}

struct SingleVec {
    x: [i16, ..5]
}

struct StructPaddedAtEnd {
    x: [i64, ..2],
    y: [i16, ..2]
}

fn main() {

    let no_padding1 = NoPadding1 {
        x: [0, 1, 2],
        y: -3,
        z: [4.5, 5.5]
    };

    let no_padding2 = NoPadding2 {
        x: [6, 7, 8],
        y: [[9, 10], [11, 12]]
    };

    let struct_internal_padding = StructInternalPadding {
        x: [13, 14],
        y: [15, 16]
    };

    let single_vec = SingleVec {
        x: [17, 18, 19, 20, 21]
    };

    let struct_padded_at_end = StructPaddedAtEnd {
        x: [22, 23],
        y: [24, 25]
    };

    zzz();
}

fn zzz() {()}
