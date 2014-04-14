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
// debugger:print s
// check:$1 = {a = 1, b = 2.5}
// debugger:continue

// debugger:finish
// debugger:print x
// check:$2 = {a = 3, b = 4.5}
// debugger:print y
// check:$3 = 5
// debugger:print z
// check:$4 = 6.5
// debugger:continue

// debugger:finish
// debugger:print a
// check:$5 = {7, 8, 9.5, 10.5}
// debugger:continue

// debugger:finish
// debugger:print a
// check:$6 = {11.5, 12.5, 13, 14}
// debugger:continue

// debugger:finish
// debugger:print x
// check:$7 = {{Case1, x = 0, y = 8970181431921507452}, {Case1, 0, 2088533116, 2088533116}}
// debugger:continue

#![feature(struct_variant)]

#[deriving(Clone)]
struct Struct {
    a: int,
    b: f64
}

#[deriving(Clone)]
struct StructStruct {
    a: Struct,
    b: Struct
}

fn fun(s: Struct) {
    zzz();
}

fn fun_fun(StructStruct { a: x, b: Struct { a: y, b: z } }: StructStruct) {
    zzz();
}

fn tup(a: (int, uint, f64, f64)) {
    zzz();
}

struct Newtype(f64, f64, int, uint);

fn new_type(a: Newtype) {
    zzz();
}

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Enum {
    Case1 { x: i64, y: i64 },
    Case2 (i64, i32, i32),
}

fn by_val_enum(x: Enum) {
    zzz();
}

fn main() {
    fun(Struct { a: 1, b: 2.5 });
    fun_fun(StructStruct { a: Struct { a: 3, b: 4.5 }, b: Struct { a: 5, b: 6.5 } });
    tup((7, 8, 9.5, 10.5));
    new_type(Newtype(11.5, 12.5, 13, 14));

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    by_val_enum(Case1 { x: 0, y: 8970181431921507452 });
}

fn zzz() {()}
