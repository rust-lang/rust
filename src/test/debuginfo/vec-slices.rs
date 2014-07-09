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

// === GDB TESTS ===================================================================================

// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print empty.length
// gdb-check:$1 = 0

// gdb-command:print singleton.length
// gdb-check:$2 = 1
// gdb-command:print *((int64_t[1]*)(singleton.data_ptr))
// gdb-check:$3 = {1}

// gdb-command:print multiple.length
// gdb-check:$4 = 4
// gdb-command:print *((int64_t[4]*)(multiple.data_ptr))
// gdb-check:$5 = {2, 3, 4, 5}

// gdb-command:print slice_of_slice.length
// gdb-check:$6 = 2
// gdb-command:print *((int64_t[2]*)(slice_of_slice.data_ptr))
// gdb-check:$7 = {3, 4}

// gdb-command:print padded_tuple.length
// gdb-check:$8 = 2
// gdb-command:print padded_tuple.data_ptr[0]
// gdb-check:$9 = {6, 7}
// gdb-command:print padded_tuple.data_ptr[1]
// gdb-check:$10 = {8, 9}

// gdb-command:print padded_struct.length
// gdb-check:$11 = 2
// gdb-command:print padded_struct.data_ptr[0]
// gdb-check:$12 = {x = 10, y = 11, z = 12}
// gdb-command:print padded_struct.data_ptr[1]
// gdb-check:$13 = {x = 13, y = 14, z = 15}

// gdb-command:print 'vec-slices::MUT_VECT_SLICE'.length
// gdb-check:$14 = 2
// gdb-command:print *((int64_t[2]*)('vec-slices::MUT_VECT_SLICE'.data_ptr))
// gdb-check:$15 = {64, 65}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print empty
// lldb-check:[...]$0 = &[]

// lldb-command:print singleton
// lldb-check:[...]$1 = &[1]

// lldb-command:print multiple
// lldb-check:[...]$2 = &[2, 3, 4, 5]

// lldb-command:print slice_of_slice
// lldb-check:[...]$3 = &[3, 4]

// lldb-command:print padded_tuple
// lldb-check:[...]$4 = &[(6, 7), (8, 9)]

// lldb-command:print padded_struct
// lldb-check:[...]$5 = &[AStruct { x: 10, y: 11, z: 12 }, AStruct { x: 13, y: 14, z: 15 }]

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

    zzz(); // #break
}

fn zzz() {()}
