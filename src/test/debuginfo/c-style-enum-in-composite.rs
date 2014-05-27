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

// gdb-command:print tuple_interior_padding
// gdb-check:$1 = {0, OneHundred}

// gdb-command:print tuple_padding_at_end
// gdb-check:$2 = {{1, OneThousand}, 2}

// gdb-command:print tuple_different_enums
// gdb-check:$3 = {OneThousand, MountainView, OneMillion, Vienna}

// gdb-command:print padded_struct
// gdb-check:$4 = {a = 3, b = OneMillion, c = 4, d = Toronto, e = 5}

// gdb-command:print packed_struct
// gdb-check:$5 = {a = 6, b = OneHundred, c = 7, d = Vienna, e = 8}

// gdb-command:print non_padded_struct
// gdb-check:$6 = {a = OneMillion, b = MountainView, c = OneThousand, d = Toronto}

// gdb-command:print struct_with_drop
// gdb-check:$7 = {{a = OneHundred, b = Vienna}, 9}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print tuple_interior_padding
// lldb-check:[...]$0 = (0, OneHundred)

// lldb-command:print tuple_padding_at_end
// lldb-check:[...]$1 = ((1, OneThousand), 2)
// lldb-command:print tuple_different_enums
// lldb-check:[...]$2 = (OneThousand, MountainView, OneMillion, Vienna)

// lldb-command:print padded_struct
// lldb-check:[...]$3 = PaddedStruct { a: 3, b: OneMillion, c: 4, d: Toronto, e: 5 }

// lldb-command:print packed_struct
// lldb-check:[...]$4 = PackedStruct { a: 6, b: OneHundred, c: 7, d: Vienna, e: 8 }

// lldb-command:print non_padded_struct
// lldb-check:[...]$5 = NonPaddedStruct { a: OneMillion, b: MountainView, c: OneThousand, d: Toronto }

// lldb-command:print struct_with_drop
// lldb-check:[...]$6 = (StructWithDrop { a: OneHundred, b: Vienna }, 9)

#![allow(unused_variable)]

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
