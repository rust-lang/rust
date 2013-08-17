// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 Broken because of LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=16249

// compile-flags:-Z extra-debug-info
// debugger:set print pretty off
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print no_padding1
// check:$1 = {x = {0, 1}, y = 2, z = {3, 4, 5}}
// debugger:print no_padding2
// check:$2 = {x = {6, 7}, y = {{8, 9}, 10}}

// debugger:print tuple_internal_padding
// check:$3 = {x = {11, 12}, y = {13, 14}}
// debugger:print struct_internal_padding
// check:$4 = {x = {15, 16}, y = {17, 18}}
// debugger:print both_internally_padded
// check:$5 = {x = {19, 20, 21}, y = {22, 23}}

// debugger:print single_tuple
// check:$6 = {x = {24, 25, 26}}

// debugger:print tuple_padded_at_end
// check:$7 = {x = {27, 28}, y = {29, 30}}
// debugger:print struct_padded_at_end
// check:$8 = {x = {31, 32}, y = {33, 34}}
// debugger:print both_padded_at_end
// check:$9 = {x = {35, 36, 37}, y = {38, 39}}

// debugger:print mixed_padding
// check:$10 = {x = {{40, 41, 42}, {43, 44}}, y = {45, 46, 47, 48}}

#[allow(unused_variable)];

struct NoPadding1 {
    x: (i32, i32),
    y: i32,
    z: (i32, i32, i32)
}

struct NoPadding2 {
    x: (i32, i32),
    y: ((i32, i32), i32)
}

struct TupleInternalPadding {
    x: (i16, i32),
    y: (i32, i64)
}

struct StructInternalPadding {
    x: (i16, i16),
    y: (i64, i64)
}

struct BothInternallyPadded {
    x: (i16, i32, i32),
    y: (i32, i64)
}

struct SingleTuple {
    x: (i16, i32, i64)
}

struct TuplePaddedAtEnd {
    x: (i32, i16),
    y: (i64, i32)
}

struct StructPaddedAtEnd {
    x: (i64, i64),
    y: (i16, i16)
}

struct BothPaddedAtEnd {
    x: (i32, i32, i16),
    y: (i64, i32)
}

// Data-layout (padding signified by dots, one column = 2 bytes):
// [a.bbc...ddddee..ffffg.hhi...]
struct MixedPadding {
    x: ((i16, i32, i16), (i64, i32)),
    y: (i64, i16, i32, i16)
}


fn main() {
    let no_padding1 = NoPadding1 {
        x: (0, 1),
        y: 2,
        z: (3, 4, 5)
    };

    let no_padding2 = NoPadding2 {
        x: (6, 7),
        y: ((8, 9), 10)
    };

    let tuple_internal_padding = TupleInternalPadding {
        x: (11, 12),
        y: (13, 14)
    };

    let struct_internal_padding = StructInternalPadding {
        x: (15, 16),
        y: (17, 18)
    };

    let both_internally_padded = BothInternallyPadded {
        x: (19, 20, 21),
        y: (22, 23)
    };

    let single_tuple = SingleTuple {
        x: (24, 25, 26)
    };

    let tuple_padded_at_end = TuplePaddedAtEnd {
        x: (27, 28),
        y: (29, 30)
    };

    let struct_padded_at_end = StructPaddedAtEnd {
        x: (31, 32),
        y: (33, 34)
    };

    let both_padded_at_end = BothPaddedAtEnd {
        x: (35, 36, 37),
        y: (38, 39)
    };

    let mixed_padding = MixedPadding {
        x: ((40, 41, 42), (43, 44)),
        y: (45, 46, 47, 48)
    };

    zzz();
}

fn zzz() {()}
