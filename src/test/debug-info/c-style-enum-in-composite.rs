// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print tuple_interior_padding
// check:$1 = {0, OneHundred}

// debugger:print tuple_padding_at_end
// check:$2 = {{1, OneThousand}, 2}

// debugger:print tuple_different_enums
// check:$3 = {OneThousand, MountainView, OneMillion, Vienna}

// debugger:print padded_struct
// check:$4 = {a = 3, b = OneMillion, c = 4, d = Toronto, e = 5}

// debugger:print packed_struct
// check:$5 = {a = 6, b = OneHundred, c = 7, d = Vienna, e = 8}

// debugger:print non_padded_struct
// check:$6 = {a = OneMillion, b = MountainView, c = OneThousand, d = Toronto}

// debugger:print struct_with_drop
// check:$7 = {{a = OneHundred, b = Vienna}, 9}

#[allow(unused_variable)];

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

#[packed]
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

    zzz();
}

fn zzz() {()}
