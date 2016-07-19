// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-gdb
// compile-flags:-g
// min-lldb-version: 310

// Check that structs get placed in the correct namespace

// lldb-command:run
// lldb-command:p struct1
// lldb-check:(struct_namespace::Struct1) $0 = [...]
// lldb-command:p struct2
// lldb-check:(struct_namespace::Struct2) $1 = [...]

// lldb-command:p mod1_struct1
// lldb-check:(struct_namespace::mod1::Struct1) $2 = [...]
// lldb-command:p mod1_struct2
// lldb-check:(struct_namespace::mod1::Struct2) $3 = [...]

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct1 {
    a: u32,
    b: u64,
}

struct Struct2(u32);

mod mod1 {

    pub struct Struct1 {
        pub a: u32,
        pub b: u64,
    }

    pub struct Struct2(pub u32);
}


fn main() {
    let struct1 = Struct1 {
        a: 0,
        b: 1,
    };

    let struct2 = Struct2(2);

    let mod1_struct1 = mod1::Struct1 {
        a: 3,
        b: 4,
    };

    let mod1_struct2 = mod1::Struct2(5);

    zzz(); // #break
}

#[inline(never)]
fn zzz() {()}
