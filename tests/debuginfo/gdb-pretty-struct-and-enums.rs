//@ ignore-lldb
//@ ignore-android: FIXME(#10381)
//@ min-gdb-version: 8.1

//@ compile-flags:-g

// gdb-command: run

// gdb-command: print regular_struct
// gdbr-check:$1 = gdb_pretty_struct_and_enums::RegularStruct {the_first_field: 101, the_second_field: 102.5, the_third_field: false}

// gdb-command: print empty_struct
// gdbr-check:$2 = gdb_pretty_struct_and_enums::EmptyStruct

// gdb-command: print c_style_enum1
// gdbr-check:$3 = gdb_pretty_struct_and_enums::CStyleEnum::CStyleEnumVar1

// gdb-command: print c_style_enum2
// gdbr-check:$4 = gdb_pretty_struct_and_enums::CStyleEnum::CStyleEnumVar2

// gdb-command: print c_style_enum3
// gdbr-check:$5 = gdb_pretty_struct_and_enums::CStyleEnum::CStyleEnumVar3

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
