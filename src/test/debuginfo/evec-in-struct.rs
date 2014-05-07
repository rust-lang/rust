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
// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print no_padding1
// gdb-check:$1 = {x = {0, 1, 2}, y = -3, z = {4.5, 5.5}}
// gdb-command:print no_padding2
// gdb-check:$2 = {x = {6, 7, 8}, y = {{9, 10}, {11, 12}}}

// gdb-command:print struct_internal_padding
// gdb-check:$3 = {x = {13, 14}, y = {15, 16}}

// gdb-command:print single_vec
// gdb-check:$4 = {x = {17, 18, 19, 20, 21}}

// gdb-command:print struct_padded_at_end
// gdb-check:$5 = {x = {22, 23}, y = {24, 25}}

#![allow(unused_variable)]

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
