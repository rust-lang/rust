//@ min-lldb-version: 1800

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *the_a
// gdb-check:$1 = unique_enum::ABC::TheA{x: 0, y: 8970181431921507452}

// gdb-command:print *the_b
// gdb-check:$2 = unique_enum::ABC::TheB(0, 286331153, 286331153)

// gdb-command:print *univariant
// gdb-check:$3 = unique_enum::Univariant::TheOnlyCase(123234)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *the_a
// lldb-check:(unique_enum::ABC) *the_a = { value = { x = 0 y = 8970181431921507452 } $discr$ = 0 }

// lldb-command:v *the_b
// lldb-check:(unique_enum::ABC) *the_b = { value = { 0 = 0 1 = 286331153 2 = 286331153 } $discr$ = 1 }

// lldb-command:v *univariant
// lldb-check:(unique_enum::Univariant) *univariant = { value = { 0 = 123234 } }

#![allow(unused_variables)]

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum ABC {
    TheA { x: i64, y: i64 },
    TheB (i64, i32, i32),
}

// This is a special case since it does not have the implicit discriminant field.
enum Univariant {
    TheOnlyCase(i64)
}

fn main() {

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let the_a: Box<_> = Box::new(ABC::TheA { x: 0, y: 8970181431921507452 });

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let the_b: Box<_> = Box::new(ABC::TheB (0, 286331153, 286331153));

    let univariant: Box<_> = Box::new(Univariant::TheOnlyCase(123234));

    zzz(); // #break
}

fn zzz() {()}
