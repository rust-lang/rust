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
// debugger:set print pretty off
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print packed
// check:$1 = {x = 123, y = 234, z = 345}

// debugger:print packedInPacked
// check:$2 = {a = 1111, b = {x = 2222, y = 3333, z = 4444}, c = 5555, d = {x = 6666, y = 7777, z = 8888}}

// debugger:print packedInUnpacked
// check:$3 = {a = -1111, b = {x = -2222, y = -3333, z = -4444}, c = -5555, d = {x = -6666, y = -7777, z = -8888}}

// debugger:print unpackedInPacked
// check:$4 = {a = 987, b = {x = 876, y = 765, z = 654, w = 543}, c = {x = 432, y = 321, z = 210, w = 109}, d = -98}

// debugger:print sizeof(packed)
// check:$5 = 14

// debugger:print sizeof(packedInPacked)
// check:$6 = 40

#[allow(unused_variable)];

#[packed]
struct Packed {
    x: i16,
    y: i32,
    z: i64
}

#[packed]
struct PackedInPacked {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

// layout (64 bit): aaaa bbbb bbbb bbbb bb.. .... cccc cccc dddd dddd dddd dd..
struct PackedInUnpacked {
    a: i32,
    b: Packed,
    c: i64,
    d: Packed
}

// layout (64 bit): xx.. yyyy zz.. .... wwww wwww
struct Unpacked {
    x: i16,
    y: i32,
    z: i16,
    w: i64
}

// layout (64 bit): aabb bbbb bbbb bbbb bbbb bbbb bbcc cccc cccc cccc cccc cccc ccdd dddd dd
#[packed]
struct UnpackedInPacked {
    a: i16,
    b: Unpacked,
    c: Unpacked,
    d: i64
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
        b: Unpacked { x: 876, y: 765, z: 654, w: 543 },
        c: Unpacked { x: 432, y: 321, z: 210, w: 109 },
        d: -98
    };

    zzz();
}

fn zzz() {()}
