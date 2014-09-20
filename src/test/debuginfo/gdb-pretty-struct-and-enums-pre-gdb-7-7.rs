// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test uses only GDB Python API features which should be available in
// older versions of GDB too. A more extensive test can be found in
// gdb-pretty-struct-and-enums.rs

// ignore-windows failing on win32 bot
// ignore-tidy-linelength
// ignore-lldb
// ignore-android: FIXME(#10381)
// compile-flags:-g
// gdb-use-pretty-printer

// The following line actually doesn't have to do anything with pretty printing,
// it just tells GDB to print values on one line:
// gdb-command: set print pretty off

// gdb-command: rbreak zzz
// gdb-command: run
// gdb-command: finish

// gdb-command: print regular_struct
// gdb-check:$1 = RegularStruct = {the_first_field = 101, the_second_field = 102.5, the_third_field = false}

// gdb-command: print empty_struct
// gdb-check:$2 = EmptyStruct

// gdb-command: print c_style_enum1
// gdb-check:$3 = CStyleEnumVar1

// gdb-command: print c_style_enum2
// gdb-check:$4 = CStyleEnumVar2

// gdb-command: print c_style_enum3
// gdb-check:$5 = CStyleEnumVar3

struct RegularStruct {
    the_first_field: int,
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

    let c_style_enum1 = CStyleEnumVar1;
    let c_style_enum2 = CStyleEnumVar2;
    let c_style_enum3 = CStyleEnumVar3;

    zzz();
}

fn zzz() { () }
