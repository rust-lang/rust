// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-bitrig
// ignore-solaris
// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-tidy-linelength
// ignore-lldb
// ignore-android: FIXME(#10381)
// compile-flags:-g

// gdb-command: run

// gdb-command: print regular_struct
// gdb-check:$1 = RegularStruct = {the_first_field = 101, the_second_field = 102.5, the_third_field = false}

// gdb-command: print empty_struct
// gdb-check:$2 = EmptyStruct

// gdb-command: print c_style_enum1
// gdbg-check:$3 = CStyleEnumVar1
// gdbr-check:$3 = gdb_pretty_struct_and_enums_pre_gdb_7_7::CStyleEnum::CStyleEnumVar1

// gdb-command: print c_style_enum2
// gdbg-check:$4 = CStyleEnumVar2
// gdbr-check:$4 = gdb_pretty_struct_and_enums_pre_gdb_7_7::CStyleEnum::CStyleEnumVar2

// gdb-command: print c_style_enum3
// gdbg-check:$5 = CStyleEnumVar3
// gdbr-check:$5 = gdb_pretty_struct_and_enums_pre_gdb_7_7::CStyleEnum::CStyleEnumVar3

#![allow(dead_code, unused_variables)]

struct RegularStruct {
    the_first_field: isize,
    the_second_field: f64,
    the_third_field: bool,
}

struct EmptyStruct;

enum CStyleEnum {
    CStyleEnumVar1,
    CStyleEnumVar2,
    CStyleEnumVar3,
}

fn main() {

    let regular_struct = RegularStruct {
        the_first_field: 101,
        the_second_field: 102.5,
        the_third_field: false
    };

    let empty_struct = EmptyStruct;

    let c_style_enum1 = CStyleEnum::CStyleEnumVar1;
    let c_style_enum2 = CStyleEnum::CStyleEnumVar2;
    let c_style_enum3 = CStyleEnum::CStyleEnumVar3;

    zzz(); // #break
}

fn zzz() { () }
