// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test makes sure that the LLDB pretty printer does not throw an exception
// when trying to handle a Vec<> or anything else that contains zero-sized
// fields.

// min-lldb-version: 310
// ignore-gdb
// ignore-tidy-linelength

// compile-flags:-g

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print v
// lldb-check:[...]$0 = vec![1, 2, 3]
// lldb-command:print zs
// lldb-check:[...]$1 = StructWithZeroSizedField { x: ZeroSizedStruct, y: 123, z: ZeroSizedStruct, w: 456 }
// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct ZeroSizedStruct;

struct StructWithZeroSizedField {
    x: ZeroSizedStruct,
    y: u32,
    z: ZeroSizedStruct,
    w: u64
}

fn main() {
    let v = vec![1,2,3];

    let zs = StructWithZeroSizedField {
        x: ZeroSizedStruct,
        y: 123,
        z: ZeroSizedStruct,
        w: 456
    };

    zzz(); // #break
}

fn zzz() { () }
