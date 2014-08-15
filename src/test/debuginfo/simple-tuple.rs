// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:set print pretty off
// gdb-command:rbreak zzz

// gdb-command:print/d 'simple-tuple::NO_PADDING_8'
// gdb-check:$1 = {-50, 50}
// gdb-command:print 'simple-tuple::NO_PADDING_16'
// gdb-check:$2 = {-1, 2, 3}
// gdb-command:print 'simple-tuple::NO_PADDING_32'
// gdb-check:$3 = {4, 5, 6}
// gdb-command:print 'simple-tuple::NO_PADDING_64'
// gdb-check:$4 = {7, 8, 9}

// gdb-command:print 'simple-tuple::INTERNAL_PADDING_1'
// gdb-check:$5 = {10, 11}
// gdb-command:print 'simple-tuple::INTERNAL_PADDING_2'
// gdb-check:$6 = {12, 13, 14, 15}

// gdb-command:print 'simple-tuple::PADDING_AT_END'
// gdb-check:$7 = {16, 17}

// gdb-command:run
// gdb-command:finish

// gdb-command:print/d noPadding8
// gdb-check:$8 = {-100, 100}
// gdb-command:print noPadding16
// gdb-check:$9 = {0, 1, 2}
// gdb-command:print noPadding32
// gdb-check:$10 = {3, 4.5, 5}
// gdb-command:print noPadding64
// gdb-check:$11 = {6, 7.5, 8}

// gdb-command:print internalPadding1
// gdb-check:$12 = {9, 10}
// gdb-command:print internalPadding2
// gdb-check:$13 = {11, 12, 13, 14}

// gdb-command:print paddingAtEnd
// gdb-check:$14 = {15, 16}

// gdb-command:print/d 'simple-tuple::NO_PADDING_8'
// gdb-check:$15 = {-127, 127}
// gdb-command:print 'simple-tuple::NO_PADDING_16'
// gdb-check:$16 = {-10, 10, 9}
// gdb-command:print 'simple-tuple::NO_PADDING_32'
// gdb-check:$17 = {14, 15, 16}
// gdb-command:print 'simple-tuple::NO_PADDING_64'
// gdb-check:$18 = {17, 18, 19}

// gdb-command:print 'simple-tuple::INTERNAL_PADDING_1'
// gdb-check:$19 = {110, 111}
// gdb-command:print 'simple-tuple::INTERNAL_PADDING_2'
// gdb-check:$20 = {112, 113, 114, 115}

// gdb-command:print 'simple-tuple::PADDING_AT_END'
// gdb-check:$21 = {116, 117}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print/d noPadding8
// lldb-check:[...]$0 = (-100, 100)
// lldb-command:print noPadding16
// lldb-check:[...]$1 = (0, 1, 2)
// lldb-command:print noPadding32
// lldb-check:[...]$2 = (3, 4.5, 5)
// lldb-command:print noPadding64
// lldb-check:[...]$3 = (6, 7.5, 8)

// lldb-command:print internalPadding1
// lldb-check:[...]$4 = (9, 10)
// lldb-command:print internalPadding2
// lldb-check:[...]$5 = (11, 12, 13, 14)

// lldb-command:print paddingAtEnd
// lldb-check:[...]$6 = (15, 16)

#![allow(unused_variable)]
#![allow(dead_code)]

static mut NO_PADDING_8: (i8, u8) = (-50, 50);
static mut NO_PADDING_16: (i16, i16, u16) = (-1, 2, 3);

static mut NO_PADDING_32: (i32, f32, u32) = (4, 5.0, 6);
static mut NO_PADDING_64: (i64, f64, u64) = (7, 8.0, 9);

static mut INTERNAL_PADDING_1: (i16, i32) = (10, 11);
static mut INTERNAL_PADDING_2: (i16, i32, u32, u64) = (12, 13, 14, 15);

static mut PADDING_AT_END: (i32, i16) = (16, 17);

fn main() {
    let noPadding8: (i8, u8) = (-100, 100);
    let noPadding16: (i16, i16, u16) = (0, 1, 2);
    let noPadding32: (i32, f32, u32) = (3, 4.5, 5);
    let noPadding64: (i64, f64, u64) = (6, 7.5, 8);

    let internalPadding1: (i16, i32) = (9, 10);
    let internalPadding2: (i16, i32, u32, u64) = (11, 12, 13, 14);

    let paddingAtEnd: (i32, i16) = (15, 16);

    unsafe {
        NO_PADDING_8 = (-127, 127);
        NO_PADDING_16 = (-10, 10, 9);

        NO_PADDING_32 = (14, 15.0, 16);
        NO_PADDING_64 = (17, 18.0, 19);

        INTERNAL_PADDING_1 = (110, 111);
        INTERNAL_PADDING_2 = (112, 113, 114, 115);

        PADDING_AT_END = (116, 117);
    }

    zzz(); // #break
}

fn zzz() {()}
