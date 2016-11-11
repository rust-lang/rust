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

// min-lldb-version: 310

// compile-flags:-g

// gdb-command:run

// gdb-command:print no_padding1
// gdbg-check:$1 = {x = {__0 = 0, __1 = 1}, y = 2, z = {__0 = 3, __1 = 4, __2 = 5}}
// gdbr-check:$1 = tuple_in_struct::NoPadding1 {x: (0, 1), y: 2, z: (3, 4, 5)}
// gdb-command:print no_padding2
// gdbg-check:$2 = {x = {__0 = 6, __1 = 7}, y = {__0 = {__0 = 8, __1 = 9}, __1 = 10}}
// gdbr-check:$2 = tuple_in_struct::NoPadding2 {x: (6, 7), y: ((8, 9), 10)}

// gdb-command:print tuple_internal_padding
// gdbg-check:$3 = {x = {__0 = 11, __1 = 12}, y = {__0 = 13, __1 = 14}}
// gdbr-check:$3 = tuple_in_struct::TupleInternalPadding {x: (11, 12), y: (13, 14)}
// gdb-command:print struct_internal_padding
// gdbg-check:$4 = {x = {__0 = 15, __1 = 16}, y = {__0 = 17, __1 = 18}}
// gdbr-check:$4 = tuple_in_struct::StructInternalPadding {x: (15, 16), y: (17, 18)}
// gdb-command:print both_internally_padded
// gdbg-check:$5 = {x = {__0 = 19, __1 = 20, __2 = 21}, y = {__0 = 22, __1 = 23}}
// gdbr-check:$5 = tuple_in_struct::BothInternallyPadded {x: (19, 20, 21), y: (22, 23)}

// gdb-command:print single_tuple
// gdbg-check:$6 = {x = {__0 = 24, __1 = 25, __2 = 26}}
// gdbr-check:$6 = tuple_in_struct::SingleTuple {x: (24, 25, 26)}

// gdb-command:print tuple_padded_at_end
// gdbg-check:$7 = {x = {__0 = 27, __1 = 28}, y = {__0 = 29, __1 = 30}}
// gdbr-check:$7 = tuple_in_struct::TuplePaddedAtEnd {x: (27, 28), y: (29, 30)}
// gdb-command:print struct_padded_at_end
// gdbg-check:$8 = {x = {__0 = 31, __1 = 32}, y = {__0 = 33, __1 = 34}}
// gdbr-check:$8 = tuple_in_struct::StructPaddedAtEnd {x: (31, 32), y: (33, 34)}
// gdb-command:print both_padded_at_end
// gdbg-check:$9 = {x = {__0 = 35, __1 = 36, __2 = 37}, y = {__0 = 38, __1 = 39}}
// gdbr-check:$9 = tuple_in_struct::BothPaddedAtEnd {x: (35, 36, 37), y: (38, 39)}

// gdb-command:print mixed_padding
// gdbg-check:$10 = {x = {__0 = {__0 = 40, __1 = 41, __2 = 42}, __1 = {__0 = 43, __1 = 44}}, y = {__0 = 45, __1 = 46, __2 = 47, __3 = 48}}
// gdbr-check:$10 = tuple_in_struct::MixedPadding {x: ((40, 41, 42), (43, 44)), y: (45, 46, 47, 48)}

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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

    zzz(); // #break
}

fn zzz() {()}
