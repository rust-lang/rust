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

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print packed
// gdbg-check:$1 = {x = 123, y = 234, z = 345}
// gdbr-check:$1 = packed_struct_with_destructor::Packed {x: 123, y: 234, z: 345}

// gdb-command:print packedInPacked
// gdbg-check:$2 = {a = 1111, b = {x = 2222, y = 3333, z = 4444}, c = 5555, d = {x = 6666, y = 7777, z = 8888}}
// gdbr-check:$2 = packed_struct_with_destructor::PackedInPacked {a: 1111, b: packed_struct_with_destructor::Packed {x: 2222, y: 3333, z: 4444}, c: 5555, d: packed_struct_with_destructor::Packed {x: 6666, y: 7777, z: 8888}}

// gdb-command:print packedInUnpacked
// gdbg-check:$3 = {a = -1111, b = {x = -2222, y = -3333, z = -4444}, c = -5555, d = {x = -6666, y = -7777, z = -8888}}
// gdbr-check:$3 = packed_struct_with_destructor::PackedInUnpacked {a: -1111, b: packed_struct_with_destructor::Packed {x: -2222, y: -3333, z: -4444}, c: -5555, d: packed_struct_with_destructor::Packed {x: -6666, y: -7777, z: -8888}}

// gdb-command:print unpackedInPacked
// gdbg-check:$4 = {a = 987, b = {x = 876, y = 765, z = 654}, c = {x = 543, y = 432, z = 321}, d = 210}
// gdbr-check:$4 = packed_struct_with_destructor::UnpackedInPacked {a: 987, b: packed_struct_with_destructor::Unpacked {x: 876, y: 765, z: 654}, c: packed_struct_with_destructor::Unpacked {x: 543, y: 432, z: 321}, d: 210}


// gdb-command:print packedInPackedWithDrop
// gdbg-check:$5 = {a = 11, b = {x = 22, y = 33, z = 44}, c = 55, d = {x = 66, y = 77, z = 88}}
// gdbr-check:$5 = packed_struct_with_destructor::PackedInPackedWithDrop {a: 11, b: packed_struct_with_destructor::Packed {x: 22, y: 33, z: 44}, c: 55, d: packed_struct_with_destructor::Packed {x: 66, y: 77, z: 88}}

// gdb-command:print packedInUnpackedWithDrop
// gdbg-check:$6 = {a = -11, b = {x = -22, y = -33, z = -44}, c = -55, d = {x = -66, y = -77, z = -88}}
// gdbr-check:$6 = packed_struct_with_destructor::PackedInUnpackedWithDrop {a: -11, b: packed_struct_with_destructor::Packed {x: -22, y: -33, z: -44}, c: -55, d: packed_struct_with_destructor::Packed {x: -66, y: -77, z: -88}}

// gdb-command:print unpackedInPackedWithDrop
// gdbg-check:$7 = {a = 98, b = {x = 87, y = 76, z = 65}, c = {x = 54, y = 43, z = 32}, d = 21}
// gdbr-check:$7 = packed_struct_with_destructor::UnpackedInPackedWithDrop {a: 98, b: packed_struct_with_destructor::Unpacked {x: 87, y: 76, z: 65}, c: packed_struct_with_destructor::Unpacked {x: 54, y: 43, z: 32}, d: 21}

// gdb-command:print deeplyNested
// gdbg-check:$8 = {a = {a = 1, b = {x = 2, y = 3, z = 4}, c = 5, d = {x = 6, y = 7, z = 8}}, b = {a = 9, b = {x = 10, y = 11, z = 12}, c = {x = 13, y = 14, z = 15}, d = 16}, c = {a = 17, b = {x = 18, y = 19, z = 20}, c = 21, d = {x = 22, y = 23, z = 24}}, d = {a = 25, b = {x = 26, y = 27, z = 28}, c = 29, d = {x = 30, y = 31, z = 32}}, e = {a = 33, b = {x = 34, y = 35, z = 36}, c = {x = 37, y = 38, z = 39}, d = 40}, f = {a = 41, b = {x = 42, y = 43, z = 44}, c = 45, d = {x = 46, y = 47, z = 48}}}
// gdbr-check:$8 = packed_struct_with_destructor::DeeplyNested {a: packed_struct_with_destructor::PackedInPacked {a: 1, b: packed_struct_with_destructor::Packed {x: 2, y: 3, z: 4}, c: 5, d: packed_struct_with_destructor::Packed {x: 6, y: 7, z: 8}}, b: packed_struct_with_destructor::UnpackedInPackedWithDrop {a: 9, b: packed_struct_with_destructor::Unpacked {x: 10, y: 11, z: 12}, c: packed_struct_with_destructor::Unpacked {x: 13, y: 14, z: 15}, d: 16}, c: packed_struct_with_destructor::PackedInUnpacked {a: 17, b: packed_struct_with_destructor::Packed {x: 18, y: 19, z: 20}, c: 21, d: packed_struct_with_destructor::Packed {x: 22, y: 23, z: 24}}, d: packed_struct_with_destructor::PackedInUnpackedWithDrop {a: 25, b: packed_struct_with_destructor::Packed {x: 26, y: 27, z: 28}, c: 29, d: packed_struct_with_destructor::Packed {x: 30, y: 31, z: 32}}, e: packed_struct_with_destructor::UnpackedInPacked {a: 33, b: packed_struct_with_destructor::Unpacked {x: 34, y: 35, z: 36}, c: packed_struct_with_destructor::Unpacked {x: 37, y: 38, z: 39}, d: 40}, f: packed_struct_with_destructor::PackedInPackedWithDrop {a: 41, b: packed_struct_with_destructor::Packed {x: 42, y: 43, z: 44}, c: 45, d: packed_struct_with_destructor::Packed {x: 46, y: 47, z: 48}}}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print packed
// lldb-check:[...]$0 = Packed { x: 123, y: 234, z: 345 }

// lldb-command:print packedInPacked
// lldb-check:[...]$1 = PackedInPacked { a: 1111, b: Packed { x: 2222, y: 3333, z: 4444 }, c: 5555, d: Packed { x: 6666, y: 7777, z: 8888 } }

// lldb-command:print packedInUnpacked
// lldb-check:[...]$2 = PackedInUnpacked { a: -1111, b: Packed { x: -2222, y: -3333, z: -4444 }, c: -5555, d: Packed { x: -6666, y: -7777, z: -8888 } }

// lldb-command:print unpackedInPacked
// lldb-check:[...]$3 = UnpackedInPacked { a: 987, b: Unpacked { x: 876, y: 765, z: 654 }, c: Unpacked { x: 543, y: 432, z: 321 }, d: 210 }

// lldb-command:print packedInPackedWithDrop
// lldb-check:[...]$4 = PackedInPackedWithDrop { a: 11, b: Packed { x: 22, y: 33, z: 44 }, c: 55, d: Packed { x: 66, y: 77, z: 88 } }

// lldb-command:print packedInUnpackedWithDrop
// lldb-check:[...]$5 = PackedInUnpackedWithDrop { a: -11, b: Packed { x: -22, y: -33, z: -44 }, c: -55, d: Packed { x: -66, y: -77, z: -88 } }

// lldb-command:print unpackedInPackedWithDrop
// lldb-check:[...]$6 = UnpackedInPackedWithDrop { a: 98, b: Unpacked { x: 87, y: 76, z: 65 }, c: Unpacked { x: 54, y: 43, z: 32 }, d: 21 }

// lldb-command:print deeplyNested
// lldb-check:[...]$7 = DeeplyNested { a: PackedInPacked { a: 1, b: Packed { x: 2, y: 3, z: 4 }, c: 5, d: Packed { x: 6, y: 7, z: 8 } }, b: UnpackedInPackedWithDrop { a: 9, b: Unpacked { x: 10, y: 11, z: 12 }, c: Unpacked { x: 13, y: 14, z: 15 }, d: 16 }, c: PackedInUnpacked { a: 17, b: Packed { x: 18, y: 19, z: 20 }, c: 21, d: Packed { x: 22, y: 23, z: 24 } }, d: PackedInUnpackedWithDrop { a: 25, b: Packed { x: 26, y: 27, z: 28 }, c: 29, d: Packed { x: 30, y: 31, z: 32 } }, e: UnpackedInPacked { a: 33, b: Unpacked { x: 34, y: 35, z: 36 }, c: Unpacked { x: 37, y: 38, z: 39 }, d: 40 }, f: PackedInPackedWithDrop { a: 41, b: Packed { x: 42, y: 43, z: 44 }, c: 45, d: Packed { x: 46, y: 47, z: 48 } } }


#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[repr(packed)]
struct Packed {
    x: i16,
    y: i32,
    z: i64
}

impl Drop for Packed {
    fn drop(&mut self) {}
}

#[repr(packed)]
struct PackedInPacked {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

struct PackedInUnpacked {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

struct Unpacked {
    x: i64,
    y: i32,
    z: i16
}

impl Drop for Unpacked {
    fn drop(&mut self) {}
}

#[repr(packed)]
struct UnpackedInPacked {
    a: i16,
    b: Unpacked,
    c: Unpacked,
    d: i64
}

#[repr(packed)]
struct PackedInPackedWithDrop {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

impl Drop for PackedInPackedWithDrop {
    fn drop(&mut self) {}
}

struct PackedInUnpackedWithDrop {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

impl Drop for PackedInUnpackedWithDrop {
    fn drop(&mut self) {}
}

#[repr(packed)]
struct UnpackedInPackedWithDrop {
    a: i16,
    b: Unpacked,
    c: Unpacked,
    d: i64
}

impl Drop for UnpackedInPackedWithDrop {
    fn drop(&mut self) {}
}

struct DeeplyNested {
    a: PackedInPacked,
    b: UnpackedInPackedWithDrop,
    c: PackedInUnpacked,
    d: PackedInUnpackedWithDrop,
    e: UnpackedInPacked,
    f: PackedInPackedWithDrop
}

fn main() {
    let packed = Packed { x: 123, y: 234, z: 345 };

    let packedInPacked = PackedInPacked {
        a: 1111,
        b: Packed { x: 2222, y: 3333, z: 4444 },
        c: 5555,
        d: Packed { x: 6666, y: 7777, z: 8888 }
    };

    let packedInUnpacked = PackedInUnpacked {
        a: -1111,
        b: Packed { x: -2222, y: -3333, z: -4444 },
        c: -5555,
        d: Packed { x: -6666, y: -7777, z: -8888 }
    };

    let unpackedInPacked = UnpackedInPacked {
        a: 987,
        b: Unpacked { x: 876, y: 765, z: 654 },
        c: Unpacked { x: 543, y: 432, z: 321 },
        d: 210
    };

    let packedInPackedWithDrop = PackedInPackedWithDrop {
        a: 11,
        b: Packed { x: 22, y: 33, z: 44 },
        c: 55,
        d: Packed { x: 66, y: 77, z: 88 }
    };

    let packedInUnpackedWithDrop = PackedInUnpackedWithDrop {
        a: -11,
        b: Packed { x: -22, y: -33, z: -44 },
        c: -55,
        d: Packed { x: -66, y: -77, z: -88 }
    };

    let unpackedInPackedWithDrop = UnpackedInPackedWithDrop {
        a: 98,
        b: Unpacked { x: 87, y: 76, z: 65 },
        c: Unpacked { x: 54, y: 43, z: 32 },
        d: 21
    };

    let deeplyNested = DeeplyNested {
        a: PackedInPacked {
            a: 1,
            b: Packed { x: 2, y: 3, z: 4 },
            c: 5,
            d: Packed { x: 6, y: 7, z: 8 }
        },
        b: UnpackedInPackedWithDrop {
            a: 9,
            b: Unpacked { x: 10, y: 11, z: 12 },
            c: Unpacked { x: 13, y: 14, z: 15 },
            d: 16
        },
        c: PackedInUnpacked {
            a: 17,
            b: Packed { x: 18, y: 19, z: 20 },
            c: 21,
            d: Packed { x: 22, y: 23, z: 24 }
        },
        d: PackedInUnpackedWithDrop {
            a: 25,
            b: Packed { x: 26, y: 27, z: 28 },
            c: 29,
            d: Packed { x: 30, y: 31, z: 32 }
        },
        e: UnpackedInPacked {
            a: 33,
            b: Unpacked { x: 34, y: 35, z: 36 },
            c: Unpacked { x: 37, y: 38, z: 39 },
            d: 40
        },
        f: PackedInPackedWithDrop {
            a: 41,
            b: Packed { x: 42, y: 43, z: 44 },
            c: 45,
            d: Packed { x: 46, y: 47, z: 48 }
        }
    };

    zzz(); // #break
}

fn zzz() {()}
