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

// debugger:print packed
// check:$1 = {x = 123, y = 234, z = 345}

// debugger:print packedInPacked
// check:$2 = {a = 1111, b = {x = 2222, y = 3333, z = 4444}, c = 5555, d = {x = 6666, y = 7777, z = 8888}}

// debugger:print packedInUnpacked
// check:$3 = {a = -1111, b = {x = -2222, y = -3333, z = -4444}, c = -5555, d = {x = -6666, y = -7777, z = -8888}}

// debugger:print unpackedInPacked
// check:$4 = {a = 987, b = {x = 876, y = 765, z = 654}, c = {x = 543, y = 432, z = 321}, d = 210}


// debugger:print packedInPackedWithDrop
// check:$5 = {a = 11, b = {x = 22, y = 33, z = 44}, c = 55, d = {x = 66, y = 77, z = 88}}

// debugger:print packedInUnpackedWithDrop
// check:$6 = {a = -11, b = {x = -22, y = -33, z = -44}, c = -55, d = {x = -66, y = -77, z = -88}}

// debugger:print unpackedInPackedWithDrop
// check:$7 = {a = 98, b = {x = 87, y = 76, z = 65}, c = {x = 54, y = 43, z = 32}, d = 21}

// debugger:print deeplyNested
// check:$8 = {a = {a = 1, b = {x = 2, y = 3, z = 4}, c = 5, d = {x = 6, y = 7, z = 8}}, b = {a = 9, b = {x = 10, y = 11, z = 12}, c = {x = 13, y = 14, z = 15}, d = 16}, c = {a = 17, b = {x = 18, y = 19, z = 20}, c = 21, d = {x = 22, y = 23, z = 24}}, d = {a = 25, b = {x = 26, y = 27, z = 28}, c = 29, d = {x = 30, y = 31, z = 32}}, e = {a = 33, b = {x = 34, y = 35, z = 36}, c = {x = 37, y = 38, z = 39}, d = 40}, f = {a = 41, b = {x = 42, y = 43, z = 44}, c = 45, d = {x = 46, y = 47, z = 48}}}

#[allow(unused_variable)];

#[packed]
struct Packed {
    x: i16,
    y: i32,
    z: i64
}

impl Drop for Packed {
    fn drop(&self) {}
}

#[packed]
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
    fn drop(&self) {}
}

#[packed]
struct UnpackedInPacked {
    a: i16,
    b: Unpacked,
    c: Unpacked,
    d: i64
}

#[packed]
struct PackedInPackedWithDrop {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

impl Drop for PackedInPackedWithDrop {
    fn drop(&self) {}
}

struct PackedInUnpackedWithDrop {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

impl Drop for PackedInUnpackedWithDrop {
    fn drop(&self) {}
}

#[packed]
struct UnpackedInPackedWithDrop {
    a: i16,
    b: Unpacked,
    c: Unpacked,
    d: i64
}

impl Drop for UnpackedInPackedWithDrop {
    fn drop(&self) {}
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

    zzz();
}

fn zzz() {()}
