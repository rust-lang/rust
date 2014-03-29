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
// debugger:set print pretty off
// debugger:rbreak zzz

// debugger:print/d 'simple-tuple::NO_PADDING_8'
// check:$1 = {-50, 50}
// debugger:print 'simple-tuple::NO_PADDING_16'
// check:$2 = {-1, 2, 3}
// debugger:print 'simple-tuple::NO_PADDING_32'
// check:$3 = {4, 5, 6}
// debugger:print 'simple-tuple::NO_PADDING_64'
// check:$4 = {7, 8, 9}

// debugger:print 'simple-tuple::INTERNAL_PADDING_1'
// check:$5 = {10, 11}
// debugger:print 'simple-tuple::INTERNAL_PADDING_2'
// check:$6 = {12, 13, 14, 15}

// debugger:print 'simple-tuple::PADDING_AT_END'
// check:$7 = {16, 17}

// debugger:run
// debugger:finish

// debugger:print/d noPadding8
// check:$8 = {-100, 100}
// debugger:print noPadding16
// check:$9 = {0, 1, 2}
// debugger:print noPadding32
// check:$10 = {3, 4.5, 5}
// debugger:print noPadding64
// check:$11 = {6, 7.5, 8}

// debugger:print internalPadding1
// check:$12 = {9, 10}
// debugger:print internalPadding2
// check:$13 = {11, 12, 13, 14}

// debugger:print paddingAtEnd
// check:$14 = {15, 16}

// debugger:print/d 'simple-tuple::NO_PADDING_8'
// check:$15 = {-127, 127}
// debugger:print 'simple-tuple::NO_PADDING_16'
// check:$16 = {-10, 10, 9}
// debugger:print 'simple-tuple::NO_PADDING_32'
// check:$17 = {14, 15, 16}
// debugger:print 'simple-tuple::NO_PADDING_64'
// check:$18 = {17, 18, 19}

// debugger:print 'simple-tuple::INTERNAL_PADDING_1'
// check:$19 = {110, 111}
// debugger:print 'simple-tuple::INTERNAL_PADDING_2'
// check:$20 = {112, 113, 114, 115}

// debugger:print 'simple-tuple::PADDING_AT_END'
// check:$21 = {116, 117}

#[allow(unused_variable)];
#[allow(dead_code)];

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

    zzz();
}

fn zzz() {()}
