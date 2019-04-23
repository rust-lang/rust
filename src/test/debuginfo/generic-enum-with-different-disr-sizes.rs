// ignore-lldb: FIXME(#27089)
// min-lldb-version: 310

// Require LLVM with DW_TAG_variant_part and a gdb that can read it.
// min-system-llvm-version: 8.0
// min-gdb-version: 8.2

// compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print eight_bytes1
// gdbr-check:$1 = generic_enum_with_different_disr_sizes::Enum<f64>::Variant1(100)

// gdb-command:print four_bytes1
// gdbr-check:$2 = generic_enum_with_different_disr_sizes::Enum<i32>::Variant1(101)

// gdb-command:print two_bytes1
// gdbr-check:$3 = generic_enum_with_different_disr_sizes::Enum<i16>::Variant1(102)

// gdb-command:print one_byte1
// gdbr-check:$4 = generic_enum_with_different_disr_sizes::Enum<u8>::Variant1(65)


// gdb-command:print eight_bytes2
// gdbr-check:$5 = generic_enum_with_different_disr_sizes::Enum<f64>::Variant2(100)

// gdb-command:print four_bytes2
// gdbr-check:$6 = generic_enum_with_different_disr_sizes::Enum<i32>::Variant2(101)

// gdb-command:print two_bytes2
// gdbr-check:$7 = generic_enum_with_different_disr_sizes::Enum<i16>::Variant2(102)

// gdb-command:print one_byte2
// gdbr-check:$8 = generic_enum_with_different_disr_sizes::Enum<u8>::Variant2(65)

// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print eight_bytes1
// lldb-check:[...]$0 = Variant1(100)
// lldb-command:print four_bytes1
// lldb-check:[...]$1 = Variant1(101)
// lldb-command:print two_bytes1
// lldb-check:[...]$2 = Variant1(102)
// lldb-command:print one_byte1
// lldb-check:[...]$3 = Variant1('A')

// lldb-command:print eight_bytes2
// lldb-check:[...]$4 = Variant2(100)
// lldb-command:print four_bytes2
// lldb-check:[...]$5 = Variant2(101)
// lldb-command:print two_bytes2
// lldb-check:[...]$6 = Variant2(102)
// lldb-command:print one_byte2
// lldb-check:[...]$7 = Variant2('A')

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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
