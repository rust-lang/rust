//@ ignore-lldb: FIXME(#27089)

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print eight_bytes1
// gdb-check:$1 = generic_enum_with_different_disr_sizes::Enum<f64>::Variant1(100)

// gdb-command:print four_bytes1
// gdb-check:$2 = generic_enum_with_different_disr_sizes::Enum<i32>::Variant1(101)

// gdb-command:print two_bytes1
// gdb-check:$3 = generic_enum_with_different_disr_sizes::Enum<i16>::Variant1(102)

// gdb-command:print one_byte1
// gdb-check:$4 = generic_enum_with_different_disr_sizes::Enum<u8>::Variant1(65)


// gdb-command:print eight_bytes2
// gdb-check:$5 = generic_enum_with_different_disr_sizes::Enum<f64>::Variant2(100)

// gdb-command:print four_bytes2
// gdb-check:$6 = generic_enum_with_different_disr_sizes::Enum<i32>::Variant2(101)

// gdb-command:print two_bytes2
// gdb-check:$7 = generic_enum_with_different_disr_sizes::Enum<i16>::Variant2(102)

// gdb-command:print one_byte2
// gdb-check:$8 = generic_enum_with_different_disr_sizes::Enum<u8>::Variant2(65)

// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:v eight_bytes1
// lldb-check:[...] Variant1(100)
// lldb-command:v four_bytes1
// lldb-check:[...] Variant1(101)
// lldb-command:v two_bytes1
// lldb-check:[...] Variant1(102)
// lldb-command:v one_byte1
// lldb-check:[...] Variant1('A')

// lldb-command:v eight_bytes2
// lldb-check:[...] Variant2(100)
// lldb-command:v four_bytes2
// lldb-check:[...] Variant2(101)
// lldb-command:v two_bytes2
// lldb-check:[...] Variant2(102)
// lldb-command:v one_byte2
// lldb-check:[...] Variant2('A')

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]

// This test case makes sure that we get correct type descriptions for the enum
// discriminant of different instantiations of the same generic enum type where,
// dependending on the generic type parameter(s), the discriminant has a
// different size in memory.

enum Enum<T> {
    Variant1(T),
    Variant2(T)
}

fn main() {
    // These are ordered for descending size on purpose
    let eight_bytes1 = Enum::Variant1(100.0f64);
    let four_bytes1 = Enum::Variant1(101i32);
    let two_bytes1 = Enum::Variant1(102i16);
    let one_byte1 = Enum::Variant1(65u8);

    let eight_bytes2 = Enum::Variant2(100.0f64);
    let four_bytes2 = Enum::Variant2(101i32);
    let two_bytes2 = Enum::Variant2(102i16);
    let one_byte2 = Enum::Variant2(65u8);

    zzz(); // #break
}

fn zzz() { () }
