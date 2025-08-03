//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print tuple_interior_padding
// gdb-check:$1 = (0, c_style_enum_in_composite::AnEnum::OneHundred)

// gdb-command:print tuple_padding_at_end
// gdb-check:$2 = ((1, c_style_enum_in_composite::AnEnum::OneThousand), 2)

// gdb-command:print tuple_different_enums
// gdb-check:$3 = (c_style_enum_in_composite::AnEnum::OneThousand, c_style_enum_in_composite::AnotherEnum::MountainView, c_style_enum_in_composite::AnEnum::OneMillion, c_style_enum_in_composite::AnotherEnum::Vienna)

// gdb-command:print padded_struct
// gdb-check:$4 = c_style_enum_in_composite::PaddedStruct {a: 3, b: c_style_enum_in_composite::AnEnum::OneMillion, c: 4, d: c_style_enum_in_composite::AnotherEnum::Toronto, e: 5}

// gdb-command:print packed_struct
// gdb-check:$5 = c_style_enum_in_composite::PackedStruct {a: 6, b: c_style_enum_in_composite::AnEnum::OneHundred, c: 7, d: c_style_enum_in_composite::AnotherEnum::Vienna, e: 8}

// gdb-command:print non_padded_struct
// gdb-check:$6 = c_style_enum_in_composite::NonPaddedStruct {a: c_style_enum_in_composite::AnEnum::OneMillion, b: c_style_enum_in_composite::AnotherEnum::MountainView, c: c_style_enum_in_composite::AnEnum::OneThousand, d: c_style_enum_in_composite::AnotherEnum::Toronto}

// gdb-command:print struct_with_drop
// gdb-check:$7 = (c_style_enum_in_composite::StructWithDrop {a: c_style_enum_in_composite::AnEnum::OneHundred, b: c_style_enum_in_composite::AnotherEnum::Vienna}, 9)

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v tuple_interior_padding
// lldb-check:[...] { 0 = 0 1 = OneHundred }

// lldb-command:v tuple_padding_at_end
// lldb-check:[...] { 0 = { 0 = 1 1 = OneThousand } 1 = 2 }

// lldb-command:v tuple_different_enums
// lldb-check:[...] { 0 = OneThousand 1 = MountainView 2 = OneMillion 3 = Vienna }

// lldb-command:v padded_struct
// lldb-check:[...] { a = 3 b = OneMillion c = 4 d = Toronto e = 5 }

// lldb-command:v packed_struct
// lldb-check:[...] { a = 6 b = OneHundred c = 7 d = Vienna e = 8 }

// lldb-command:v non_padded_struct
// lldb-check:[...] { a = OneMillion b = MountainView c = OneThousand d = Toronto }

// lldb-command:v struct_with_drop
// lldb-check:[...] { 0 = { a = OneHundred b = Vienna } 1 = 9 }

#![allow(unused_variables)]

use self::AnEnum::{OneHundred, OneThousand, OneMillion};
use self::AnotherEnum::{MountainView, Toronto, Vienna};

enum AnEnum {
    OneHundred = 100,
    OneThousand = 1000,
    OneMillion = 1000000
}

enum AnotherEnum {
    MountainView,
    Toronto,
    Vienna
}

struct PaddedStruct {
    a: i16,
    b: AnEnum,
    c: i16,
    d: AnotherEnum,
    e: i16
}

#[repr(packed)]
struct PackedStruct {
    a: i16,
    b: AnEnum,
    c: i16,
    d: AnotherEnum,
    e: i16
}

struct NonPaddedStruct {
    a: AnEnum,
    b: AnotherEnum,
    c: AnEnum,
    d: AnotherEnum
}

struct StructWithDrop {
    a: AnEnum,
    b: AnotherEnum
}

impl Drop for StructWithDrop {
    fn drop(&mut self) {()}
}

fn main() {

    let tuple_interior_padding = (0_i16, OneHundred);
    // It will depend on the machine architecture if any padding is actually involved here
    let tuple_padding_at_end = ((1_u64, OneThousand), 2_u64);
    let tuple_different_enums = (OneThousand, MountainView, OneMillion, Vienna);

    let padded_struct = PaddedStruct {
        a: 3,
        b: OneMillion,
        c: 4,
        d: Toronto,
        e: 5
    };

    let packed_struct = PackedStruct {
        a: 6,
        b: OneHundred,
        c: 7,
        d: Vienna,
        e: 8
    };

    let non_padded_struct = NonPaddedStruct {
        a: OneMillion,
        b: MountainView,
        c: OneThousand,
        d: Toronto
    };

    let struct_with_drop = (StructWithDrop { a: OneHundred, b: Vienna }, 9_i64);

    zzz(); // #break
}

fn zzz() { () }
