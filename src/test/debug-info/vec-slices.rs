// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:set print pretty off
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:print empty.length
// check:$1 = 0

// debugger:print singleton.length
// check:$2 = 1
// debugger:print *((int64_t[1]*)(singleton.data_ptr))
// check:$3 = {1}

// debugger:print multiple.length
// check:$4 = 4
// debugger:print *((int64_t[4]*)(multiple.data_ptr))
// check:$5 = {2, 3, 4, 5}

// debugger:print slice_of_slice.length
// check:$6 = 2
// debugger:print *((int64_t[2]*)(slice_of_slice.data_ptr))
// check:$7 = {3, 4}

// debugger:print padded_tuple.length
// check:$8 = 2
// debugger:print padded_tuple.data_ptr[0]
// check:$9 = {6, 7}
// debugger:print padded_tuple.data_ptr[1]
// check:$10 = {8, 9}

// debugger:print padded_struct.length
// check:$11 = 2
// debugger:print padded_struct.data_ptr[0]
// check:$12 = {x = 10, y = 11, z = 12}
// debugger:print padded_struct.data_ptr[1]
// check:$13 = {x = 13, y = 14, z = 15}

// debugger:print 'vec-slices::MUT_VECT_SLICE'.length
// check:$14 = 2
// debugger:print *((int64_t[2]*)('vec-slices::MUT_VECT_SLICE'.data_ptr))
// check:$15 = {64, 65}

#![allow(unused_variable)]

struct AStruct {
    x: i16,
    y: i32,
    z: i16
}

static VECT_SLICE: &'static [i64] = &[64, 65];
static mut MUT_VECT_SLICE: &'static [i64] = &[32];

fn main() {
    let empty: &[i64] = &[];
    let singleton: &[i64] = &[1];
    let multiple: &[i64] = &[2, 3, 4, 5];
    let slice_of_slice = multiple.slice(1,3);

    let padded_tuple: &[(i32, i16)] = &[(6, 7), (8, 9)];

    let padded_struct: &[AStruct] = &[
        AStruct { x: 10, y: 11, z: 12 },
        AStruct { x: 13, y: 14, z: 15 }
    ];

    unsafe {
        MUT_VECT_SLICE = VECT_SLICE;
    }

    zzz();
}

fn zzz() {()}
