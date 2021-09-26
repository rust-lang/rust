// min-lldb-version: 310

// Gdb doesn't know about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *bool_ref
// gdb-check:$1 = true

// gdb-command:print *int_ref
// gdb-check:$2 = -1

// gdb-command:print *char_ref
// gdbg-check:$3 = 97
// gdbr-check:$3 = 97 'a'

// gdb-command:print/d *i8_ref
// gdb-check:$4 = 68

// gdb-command:print *i16_ref
// gdb-check:$5 = -16

// gdb-command:print *i32_ref
// gdb-check:$6 = -32

// gdb-command:print *i64_ref
// gdb-check:$7 = -64

// gdb-command:print *uint_ref
// gdb-check:$8 = 1

// gdb-command:print/d *u8_ref
// gdb-check:$9 = 100

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

// lldb-command:type format add -f decimal char
// lldb-command:type format add -f decimal 'unsigned char'
// lldb-command:run

// lldb-command:print *bool_ref
// lldbg-check:[...]$0 = true
// lldbr-check:(bool) *bool_ref = true

// lldb-command:print *int_ref
// lldbg-check:[...]$1 = -1
// lldbr-check:(isize) *int_ref = -1

// NOTE: only rust-enabled lldb supports 32bit chars
// lldbr-command:print *char_ref
// lldbr-check:(char) *char_ref = 97

// lldb-command:print *i8_ref
// lldbg-check:[...]$2 = 68
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
// lldbg-check:[...]$7 = 100
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
    let bool_box: Box<bool> = Box::new(true);
    let bool_ref: &bool = &*bool_box;

    let int_box: Box<isize> = Box::new(-1);
    let int_ref: &isize = &*int_box;

    let char_box: Box<char> = Box::new('a');
    let char_ref: &char = &*char_box;

    let i8_box: Box<i8> = Box::new(68);
    let i8_ref: &i8 = &*i8_box;

    let i16_box: Box<i16> = Box::new(-16);
    let i16_ref: &i16 = &*i16_box;

    let i32_box: Box<i32> = Box::new(-32);
    let i32_ref: &i32 = &*i32_box;

    let i64_box: Box<i64> = Box::new(-64);
    let i64_ref: &i64 = &*i64_box;

    let uint_box: Box<usize> = Box::new(1);
    let uint_ref: &usize = &*uint_box;

    let u8_box: Box<u8> = Box::new(100);
    let u8_ref: &u8 = &*u8_box;

    let u16_box: Box<u16> = Box::new(16);
    let u16_ref: &u16 = &*u16_box;

    let u32_box: Box<u32> = Box::new(32);
    let u32_ref: &u32 = &*u32_box;

    let u64_box: Box<u64> = Box::new(64);
    let u64_ref: &u64 = &*u64_box;

    let f32_box: Box<f32> = Box::new(2.5);
    let f32_ref: &f32 = &*f32_box;

    let f64_box: Box<f64> = Box::new(3.5);
    let f64_ref: &f64 = &*f64_box;

    zzz(); // #break
}

fn zzz() {()}
