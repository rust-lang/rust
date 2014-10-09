// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// ignore-android: FIXME(#10381)

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print s
// gdb-check:$1 = {a = 1, b = 2.5}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$2 = {a = 3, b = 4.5}
// gdb-command:print y
// gdb-check:$3 = 5
// gdb-command:print z
// gdb-check:$4 = 6.5
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$5 = {7, 8, 9.5, 10.5}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$6 = {11.5, 12.5, 13, 14}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = {{RUST$ENUM$DISR = Case1, x = 0, y = 8970181431921507452}, {RUST$ENUM$DISR = Case1, 0, 2088533116, 2088533116}}
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print s
// lldb-check:[...]$0 = Struct { a: 1, b: 2.5 }
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$1 = Struct { a: 3, b: 4.5 }
// lldb-command:print y
// lldb-check:[...]$2 = 5
// lldb-command:print z
// lldb-check:[...]$3 = 6.5
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$4 = (7, 8, 9.5, 10.5)
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$5 = Newtype(11.5, 12.5, 13, 14)
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$6 = Case1 { x: 0, y: 8970181431921507452 }
// lldb-command:continue

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
    zzz(); // #break
}

fn fun_fun(StructStruct { a: x, b: Struct { a: y, b: z } }: StructStruct) {
    zzz(); // #break
}

fn tup(a: (int, uint, f64, f64)) {
    zzz(); // #break
}

struct Newtype(f64, f64, int, uint);

fn new_type(a: Newtype) {
    zzz(); // #break
}

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Enum {
    Case1 { x: i64, y: i64 },
    Case2 (i64, i32, i32),
}

fn by_val_enum(x: Enum) {
    zzz(); // #break
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

fn zzz() { () }
