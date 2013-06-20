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

// debugger:print noPadding1
// check:$1 = {x = {0, 1}, y = 2, z = {3, 4, 5}}
// debugger:print noPadding2
// check:$2 = {x = {6, 7}, y = {{8, 9}, 10}}

// debugger:print tupleInternalPadding
// check:$3 = {x = {11, 12}, y = {13, 14}}
// debugger:print structInternalPadding
// check:$4 = {x = {15, 16}, y = {17, 18}}
// debugger:print bothInternallyPadded
// check:$5 = {x = {19, 20, 21}, y = {22, 23}}

// debugger:print singleTuple
// check:$6 = {x = {24, 25, 26}}

// debugger:print tuplePaddedAtEnd
// check:$7 = {x = {27, 28}, y = {29, 30}}
// debugger:print structPaddedAtEnd
// check:$8 = {x = {31, 32}, y = {33, 34}}
// debugger:print bothPaddedAtEnd
// check:$9 = {x = {35, 36, 37}, y = {38, 39}}

// debugger:print mixedPadding
// check:$10 = {x = {{40, 41, 42}, {43, 44}}, y = {45, 46, 47, 48}}

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
    let noPadding1 = NoPadding1 {
        x: (0, 1),
        y: 2,
        z: (3, 4, 5)
    };

    let noPadding2 = NoPadding2 {
        x: (6, 7),
        y: ((8, 9), 10)
    };

    let tupleInternalPadding = TupleInternalPadding {
        x: (11, 12),
        y: (13, 14)
    };

    let structInternalPadding = StructInternalPadding {
        x: (15, 16),
        y: (17, 18)
    };

    let bothInternallyPadded = BothInternallyPadded {
        x: (19, 20, 21),
        y: (22, 23)
    };

    let singleTuple = SingleTuple {
        x: (24, 25, 26)
    };

    let tuplePaddedAtEnd = TuplePaddedAtEnd {
        x: (27, 28),
        y: (29, 30)
    };

    let structPaddedAtEnd = StructPaddedAtEnd {
        x: (31, 32),
        y: (33, 34)
    };

    let bothPaddedAtEnd = BothPaddedAtEnd {
        x: (35, 36, 37),
        y: (38, 39)
    };

    let mixedPadding = MixedPadding {
        x: ((40, 41, 42), (43, 44)),
        y: (45, 46, 47, 48)
    };

    zzz();
}

fn zzz() {()}