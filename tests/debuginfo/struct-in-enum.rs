//@ min-lldb-version: 1800

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:set print union on
// gdb-command:run

// gdb-command:print case1
// gdb-check:$1 = struct_in_enum::Regular::Case1(0, struct_in_enum::Struct {x: 2088533116, y: 2088533116, z: 31868})

// gdb-command:print case2
// gdb-check:$2 = struct_in_enum::Regular::Case2(0, 1229782938247303441, 4369)

// gdb-command:print univariant
// gdb-check:$3 = struct_in_enum::Univariant::TheOnlyCase(struct_in_enum::Struct {x: 123, y: 456, z: 789})


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v case1
// lldb-check:[...] Case1(0, Struct { x: 2088533116, y: 2088533116, z: 31868 })
// lldb-command:v case2
// lldb-check:[...] Case2(0, 1229782938247303441, 4369)

// lldb-command:v univariant
// lldb-check:[...] TheOnlyCase(Struct { x: 123, y: 456, z: 789 })

#![allow(unused_variables)]

use self::Regular::{Case1, Case2};
use self::Univariant::TheOnlyCase;

struct Struct {
    x: u32,
    y: i32,
    z: i16
}

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular {
    Case1(u64, Struct),
    Case2(u64, u64, i16)
}

enum Univariant {
    TheOnlyCase(Struct)
}

fn main() {

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1 = Case1(0, Struct { x: 2088533116, y: 2088533116, z: 31868 });

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2 = Case2(0, 1229782938247303441, 4369);

    let univariant = TheOnlyCase(Struct { x: 123, y: 456, z: 789 });

    zzz(); // #break
}

fn zzz() {()}
