//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print packed
// gdb-check:$1 = packed_struct::Packed {x: 123, y: 234, z: 345}

// gdb-command:print packedInPacked
// gdb-check:$2 = packed_struct::PackedInPacked {a: 1111, b: packed_struct::Packed {x: 2222, y: 3333, z: 4444}, c: 5555, d: packed_struct::Packed {x: 6666, y: 7777, z: 8888}}

// gdb-command:print packedInUnpacked
// gdb-check:$3 = packed_struct::PackedInUnpacked {a: -1111, b: packed_struct::Packed {x: -2222, y: -3333, z: -4444}, c: -5555, d: packed_struct::Packed {x: -6666, y: -7777, z: -8888}}

// gdb-command:print unpackedInPacked
// gdb-check:$4 = packed_struct::UnpackedInPacked {a: 987, b: packed_struct::Unpacked {x: 876, y: 765, z: 654, w: 543}, c: packed_struct::Unpacked {x: 432, y: 321, z: 210, w: 109}, d: -98}

// gdb-command:print sizeof(packed)
// gdb-check:$5 = 14

// gdb-command:print sizeof(packedInPacked)
// gdb-check:$6 = 40


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v packed
// lldb-check:[...] { x = 123 y = 234 z = 345 }

// lldb-command:v packedInPacked
// lldb-check:[...] { a = 1111 b = { x = 2222 y = 3333 z = 4444 } c = 5555 d = { x = 6666 y = 7777 z = 8888 } }

// lldb-command:v packedInUnpacked
// lldb-check:[...] { a = -1111 b = { x = -2222 y = -3333 z = -4444 } c = -5555 d = { x = -6666 y = -7777 z = -8888 } }

// lldb-command:v unpackedInPacked
// lldb-check:[...] { a = 987 b = { x = 876 y = 765 z = 654 w = 543 } c = { x = 432 y = 321 z = 210 w = 109 } d = -98 }

// lldb-command:expr sizeof(packed)
// lldb-check:[...] 14

// lldb-command:expr sizeof(packedInPacked)
// lldb-check:[...] 40

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[repr(packed)]
struct Packed {
    x: i16,
    y: i32,
    z: i64
}

#[repr(packed)]
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
#[repr(packed)]
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

    zzz(); // #break
}

fn zzz() {()}
