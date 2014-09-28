// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows failing on win32 bot
// ignore-tidy-linelength
// ignore-lldb
// ignore-android: FIXME(#10381)
// compile-flags:-g
// gdb-use-pretty-printer

// This test uses some GDB Python API features (e.g. accessing anonymous fields)
// which are only available in newer GDB version. The following directive will
// case the test runner to ignore this test if an older GDB version is used:
// min-gdb-version 7.7

// The following line actually doesn't have to do anything with pretty printing,
// it just tells GDB to print values on one line:
// gdb-command: set print pretty off

// gdb-command: rbreak zzz
// gdb-command: run
// gdb-command: finish

// gdb-command: print regular_struct
// gdb-check:$1 = RegularStruct = {the_first_field = 101, the_second_field = 102.5, the_third_field = false, the_fourth_field = "I'm so pretty, oh so pretty..."}

// gdb-command: print tuple
// gdb-check:$2 = {true, 103, "blub"}

// gdb-command: print tuple_struct
// gdb-check:$3 = TupleStruct = {-104.5, 105}

// gdb-command: print empty_struct
// gdb-check:$4 = EmptyStruct

// gdb-command: print c_style_enum1
// gdb-check:$5 = CStyleEnumVar1

// gdb-command: print c_style_enum2
// gdb-check:$6 = CStyleEnumVar2

// gdb-command: print c_style_enum3
// gdb-check:$7 = CStyleEnumVar3

// gdb-command: print mixed_enum_c_style_var
// gdb-check:$8 = MixedEnumCStyleVar

// gdb-command: print mixed_enum_tuple_var
// gdb-check:$9 = MixedEnumTupleVar = {106, 107, false}

// gdb-command: print mixed_enum_struct_var
// gdb-check:$10 = MixedEnumStructVar = {field1 = 108.5, field2 = 109}

// gdb-command: print some
// gdb-check:$11 = Some = {110}

// gdb-command: print none
// gdb-check:$12 = None

// gdb-command: print nested_variant1
// gdb-check:$13 = NestedVariant1 = {NestedStruct = {regular_struct = RegularStruct = {the_first_field = 111, the_second_field = 112.5, the_third_field = true, the_fourth_field = "NestedStructString1"}, tuple_struct = TupleStruct = {113.5, 114}, empty_struct = EmptyStruct, c_style_enum = CStyleEnumVar2, mixed_enum = MixedEnumTupleVar = {115, 116, false}}}

// gdb-command: print nested_variant2
// gdb-check:$14 = NestedVariant2 = {abc = NestedStruct = {regular_struct = RegularStruct = {the_first_field = 117, the_second_field = 118.5, the_third_field = false, the_fourth_field = "NestedStructString10"}, tuple_struct = TupleStruct = {119.5, 120}, empty_struct = EmptyStruct, c_style_enum = CStyleEnumVar3, mixed_enum = MixedEnumStructVar = {field1 = 121.5, field2 = -122}}}

#![feature(struct_variant)]

struct RegularStruct {
    the_first_field: int,
    the_second_field: f64,
    the_third_field: bool,
    the_fourth_field: &'static str,
}

struct TupleStruct(f64, i16);

struct EmptyStruct;

enum CStyleEnum {
    CStyleEnumVar1,
    CStyleEnumVar2,
    CStyleEnumVar3,
}

enum MixedEnum {
    MixedEnumCStyleVar,
    MixedEnumTupleVar(u32, u16, bool),
    MixedEnumStructVar { field1: f64, field2: i32 }
}

struct NestedStruct {
    regular_struct: RegularStruct,
    tuple_struct: TupleStruct,
    empty_struct: EmptyStruct,
    c_style_enum: CStyleEnum,
    mixed_enum: MixedEnum,
}

enum NestedEnum {
    NestedVariant1(NestedStruct),
    NestedVariant2 { abc: NestedStruct }
}

fn main() {

    let regular_struct = RegularStruct {
        the_first_field: 101,
        the_second_field: 102.5,
        the_third_field: false,
        the_fourth_field: "I'm so pretty, oh so pretty..."
    };

    let tuple = ( true, 103u32, "blub" );

    let tuple_struct = TupleStruct(-104.5, 105);

    let empty_struct = EmptyStruct;

    let c_style_enum1 = CStyleEnumVar1;
    let c_style_enum2 = CStyleEnumVar2;
    let c_style_enum3 = CStyleEnumVar3;

    let mixed_enum_c_style_var = MixedEnumCStyleVar;
    let mixed_enum_tuple_var = MixedEnumTupleVar(106, 107, false);
    let mixed_enum_struct_var = MixedEnumStructVar { field1: 108.5, field2: 109 };

    let some = Some(110u);
    let none: Option<int> = None;

    let nested_variant1 = NestedVariant1(
        NestedStruct {
            regular_struct: RegularStruct {
                the_first_field: 111,
                the_second_field: 112.5,
                the_third_field: true,
                the_fourth_field: "NestedStructString1",
            },
            tuple_struct: TupleStruct(113.5, 114),
            empty_struct: EmptyStruct,
            c_style_enum: CStyleEnumVar2,
            mixed_enum: MixedEnumTupleVar(115, 116, false)
        }
    );

    let nested_variant2 = NestedVariant2 {
        abc: NestedStruct {
            regular_struct: RegularStruct {
                the_first_field: 117,
                the_second_field: 118.5,
                the_third_field: false,
                the_fourth_field: "NestedStructString10",
            },
            tuple_struct: TupleStruct(119.5, 120),
            empty_struct: EmptyStruct,
            c_style_enum: CStyleEnumVar3,
            mixed_enum: MixedEnumStructVar {
                field1: 121.5,
                field2: -122
            }
        }
    };

    zzz();
}

fn zzz() { () }
