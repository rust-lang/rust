// Require LLVM with DW_TAG_variant_part and a gdb that can read it.
// min-system-llvm-version: 8.0
// min-gdb-version: 8.2

// compile-flags:-g -Z thinlto

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *abc
// gdbr-check:$1 = enum_thinlto::ABC::TheA{x: 0, y: 8970181431921507452}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *abc
// lldbg-check:(enum_thinlto::ABC) $0 = ABC { }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
#[derive(Debug)]
enum ABC {
    TheA { x: i64, y: i64 },
    TheB (i64, i32, i32),
}

fn main() {
    let abc = ABC::TheA { x: 0, y: 0x7c7c_7c7c_7c7c_7c7c };

    f(&abc);
}

fn f(abc: &ABC) {
    zzz(); // #break

    println!("{:?}", abc);
}

fn zzz() {()}
