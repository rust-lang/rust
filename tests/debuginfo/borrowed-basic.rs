//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print *bool_ref
// gdb-check:$1 = true

// gdb-command:print *int_ref
// gdb-check:$2 = -1

// gdb-command:print/d *char_ref
// gdb-check:$3 = 97

// gdb-command:print *i8_ref
// gdb-check:$4 = 68

// gdb-command:print *i16_ref
// gdb-check:$5 = -16

// gdb-command:print *i32_ref
// gdb-check:$6 = -32

// gdb-command:print *i64_ref
// gdb-check:$7 = -64

// gdb-command:print *uint_ref
// gdb-check:$8 = 1

// gdb-command:print *u8_ref
// gdb-check:$9 = 100

// gdb-command:print *u16_ref
// gdb-check:$10 = 16

// gdb-command:print *u32_ref
// gdb-check:$11 = 32

// gdb-command:print *u64_ref
// gdb-check:$12 = 64

// gdb-command:print *f16_ref
// gdb-check:$13 = 1.5

// gdb-command:print *f32_ref
// gdb-check:$14 = 2.5

// gdb-command:print *f64_ref
// gdb-check:$15 = 3.5


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v *bool_ref
// lldb-check:[...] true

// lldb-command:v *int_ref
// lldb-check:[...] -1


// lldb-command:v *i8_ref
// lldb-check:[...] 'D'

// lldb-command:v *i16_ref
// lldb-check:[...] -16

// lldb-command:v *i32_ref
// lldb-check:[...] -32

// lldb-command:v *i64_ref
// lldb-check:[...] -64

// lldb-command:v *uint_ref
// lldb-check:[...] 1

// lldb-command:v *u8_ref
// lldb-check:[...] 'd'

// lldb-command:v *u16_ref
// lldb-check:[...] 16

// lldb-command:v *u32_ref
// lldb-check:[...] 32

// lldb-command:v *u64_ref
// lldb-check:[...] 64

// lldb-command:v *f16_ref
// lldb-check:[...] 1.5

// lldb-command:v *f32_ref
// lldb-check:[...] 2.5

// lldb-command:v *f64_ref
// lldb-check:[...] 3.5

#![allow(unused_variables)]
#![feature(f16)]

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

    let f16_val: f16 = 1.5;
    let f16_ref: &f16 = &f16_val;

    let f32_val: f32 = 2.5;
    let f32_ref: &f32 = &f32_val;

    let f64_val: f64 = 3.5;
    let f64_ref: &f64 = &f64_val;

    zzz(); // #break
}

fn zzz() {()}
