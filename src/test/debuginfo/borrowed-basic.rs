// Gdb doesn't know about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// compile-flags:-g
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print *bool_ref
// gdb-check:$1 = true

// gdb-command:print *int_ref
// gdb-check:$2 = -1

// gdb-command:print *char_ref
// gdbg-check:$3 = 97
// gdbr-check:$3 = 97 'a'

// gdb-command:print *i8_ref
// gdbg-check:$4 = 68 'D'
// gdbr-check:$4 = 68

// gdb-command:print *i16_ref
// gdb-check:$5 = -16

// gdb-command:print *i32_ref
// gdb-check:$6 = -32

// gdb-command:print *i64_ref
// gdb-check:$7 = -64

// gdb-command:print *uint_ref
// gdb-check:$8 = 1

// gdb-command:print *u8_ref
// gdbg-check:$9 = 100 'd'
// gdbr-check:$9 = 100

// gdb-command:print *u16_ref
// gdb-check:$10 = 16

// gdb-command:print *u32_ref
// gdb-check:$11 = 32

// gdb-command:print *u64_ref
// gdb-check:$12 = 64

// gdb-command:print *f32_ref
// gdb-check:$13 = 2.5

// gdb-command:print *f64_ref
// gdb-check:$14 = 3.5


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print *bool_ref
// lldbg-check:[...]$0 = true
// lldbr-check:(bool) *bool_ref = true

// lldb-command:print *int_ref
// lldbg-check:[...]$1 = -1
// lldbr-check:(isize) *int_ref = -1

// NOTE: only rust-enabled lldb supports 32bit chars
// lldbr-command:print *char_ref
// lldbr-check:(char) *char_ref = 'a'

// lldb-command:print *i8_ref
// lldbg-check:[...]$2 = 'D'
// lldbr-check:(i8) *i8_ref = 68

// lldb-command:print *i16_ref
// lldbg-check:[...]$3 = -16
// lldbr-check:(i16) *i16_ref = -16

// lldb-command:print *i32_ref
// lldbg-check:[...]$4 = -32
// lldbr-check:(i32) *i32_ref = -32

// lldb-command:print *i64_ref
// lldbg-check:[...]$5 = -64
// lldbr-check:(i64) *i64_ref = -64

// lldb-command:print *uint_ref
// lldbg-check:[...]$6 = 1
// lldbr-check:(usize) *uint_ref = 1

// lldb-command:print *u8_ref
// lldbg-check:[...]$7 = 'd'
// lldbr-check:(u8) *u8_ref = 100

// lldb-command:print *u16_ref
// lldbg-check:[...]$8 = 16
// lldbr-check:(u16) *u16_ref = 16

// lldb-command:print *u32_ref
// lldbg-check:[...]$9 = 32
// lldbr-check:(u32) *u32_ref = 32

// lldb-command:print *u64_ref
// lldbg-check:[...]$10 = 64
// lldbr-check:(u64) *u64_ref = 64

// lldb-command:print *f32_ref
// lldbg-check:[...]$11 = 2.5
// lldbr-check:(f32) *f32_ref = 2.5

// lldb-command:print *f64_ref
// lldbg-check:[...]$12 = 3.5
// lldbr-check:(f64) *f64_ref = 3.5

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let bool_val: bool = true;
    let bool_ref: &bool = &bool_val;

    let int_val: isize = -1;
    let int_ref: &isize = &int_val;

    let char_val: char = 'a';
    let char_ref: &char = &char_val;

    let i8_val: i8 = 68;
    let i8_ref: &i8 = &i8_val;

    let i16_val: i16 = -16;
    let i16_ref: &i16 = &i16_val;

    let i32_val: i32 = -32;
    let i32_ref: &i32 = &i32_val;

    let i64_val: i64 = -64;
    let i64_ref: &i64 = &i64_val;

    let uint_val: usize = 1;
    let uint_ref: &usize = &uint_val;

    let u8_val: u8 = 100;
    let u8_ref: &u8 = &u8_val;

    let u16_val: u16 = 16;
    let u16_ref: &u16 = &u16_val;

    let u32_val: u32 = 32;
    let u32_ref: &u32 = &u32_val;

    let u64_val: u64 = 64;
    let u64_ref: &u64 = &u64_val;

    let f32_val: f32 = 2.5;
    let f32_ref: &f32 = &f32_val;

    let f64_val: f64 = 3.5;
    let f64_ref: &f64 = &f64_val;

    zzz(); // #break
}

fn zzz() {()}
