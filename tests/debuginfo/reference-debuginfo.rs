// Copy of `borrowed-basic.rs` which enables the `ReferencePropagation` MIR pass.
// That pass replaces debuginfo for `a => _x` where `_x = &b` to be `a => &b`,
// and leaves codegen to create a ladder of allocations so as `*a == b`.
//
// compile-flags:-g -Zmir-enable-passes=+ReferencePropagation,-ConstDebugInfo
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:info locals
// gdb-check:*f64_double_ref = 3.5
// gdb-check:*f64_ref = 3.5
// gdb-check:f64_val = 3.5
// gdb-check:*f32_ref = 2.5
// gdb-check:f32_val = 2.5
// gdb-check:*u64_ref = 64
// gdb-check:u64_val = 64
// gdb-check:*u32_ref = 32
// gdb-check:u32_val = 32
// gdb-check:*u16_ref = 16
// gdb-check:u16_val = 16
// gdb-check:*u8_ref = 100
// gdb-check:u8_val = 100
// gdb-check:*uint_ref = 1
// gdb-check:uint_val = 1
// gdb-check:*i64_ref = -64
// gdb-check:i64_val = -64
// gdb-check:*i32_ref = -32
// gdb-check:*i16_ref = -16
// gdb-check:i16_val = -16
// gdb-check:*i8_ref = 68
// gdb-check:i8_val = 68
// gdb-check:*char_ref = 97 'a'
// gdb-check:char_val = 97 'a'
// gdb-check:*int_ref = -1
// gdb-check:int_val = -1
// gdb-check:*bool_ref = true
// gdb-check:bool_val = true


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:frame variable
// lldbr-check:(bool) bool_val = true
// lldbr-check:(bool) *bool_ref = true
// lldbr-check:(long) int_val = -1
// lldbr-check:(long) *int_ref = -1
// lldbr-check:(char32_t) char_val = U+0x00000061 U'a'
// lldbr-check:(char32_t) *char_ref = U+0x00000061 U'a'
// lldbr-check:(char) i8_val = 'D'
// lldbr-check:(char) *i8_ref = 'D'
// lldbr-check:(short) i16_val = -16
// lldbr-check:(short) *i16_ref = -16
// lldbr-check:(int) i32_val = -32
// lldbr-check:(int) *i32_ref = -32
// lldbr-check:(long) i64_val = -64
// lldbr-check:(long) *i64_ref = -64
// lldbr-check:(unsigned long) uint_val = 1
// lldbr-check:(unsigned long) *uint_ref = 1
// lldbr-check:(unsigned char) u8_val = 'd'
// lldbr-check:(unsigned char) *u8_ref = 'd'
// lldbr-check:(unsigned short) u16_val = 16
// lldbr-check:(unsigned short) *u16_ref = 16
// lldbr-check:(unsigned int) u32_val = 32
// lldbr-check:(unsigned int) *u32_ref = 32
// lldbr-check:(unsigned long) u64_val = 64
// lldbr-check:(unsigned long) *u64_ref = 64
// lldbr-check:(float) f32_val = 2.5
// lldbr-check:(float) *f32_ref = 2.5
// lldbr-check:(double) f64_val = 3.5
// lldbr-check:(double) *f64_ref = 3.5
// lldbr-check:(double) *f64_double_ref = 3.5
// lldbr-check:[...]$0 = true

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
    let f64_double_ref: &f64 = &f64_ref;

    zzz(); // #break
}

fn zzz() {()}
