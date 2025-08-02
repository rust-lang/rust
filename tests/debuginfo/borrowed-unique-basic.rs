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

// gdb-command:print *f16_ref
// gdb-check:$13 = 1.5

// gdb-command:print *f32_ref
// gdb-check:$14 = 2.5

// gdb-command:print *f64_ref
// gdb-check:$15 = 3.5


// === LLDB TESTS ==================================================================================

// lldb-command:type format add -f decimal char
// lldb-command:type format add -f decimal 'unsigned char'
// lldb-command:run

// lldb-command:v *bool_ref
// lldb-check:[...] true

// lldb-command:v *int_ref
// lldb-check:[...] -1


// lldb-command:v *i8_ref
// lldb-check:[...] 68

// lldb-command:v *i16_ref
// lldb-check:[...] -16

// lldb-command:v *i32_ref
// lldb-check:[...] -32

// lldb-command:v *i64_ref
// lldb-check:[...] -64

// lldb-command:v *uint_ref
// lldb-check:[...] 1

// lldb-command:v *u8_ref
// lldb-check:[...] 100

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

    let f16_box: Box<f16> = Box::new(1.5);
    let f16_ref: &f16 = &*f16_box;

    let f32_box: Box<f32> = Box::new(2.5);
    let f32_ref: &f32 = &*f32_box;

    let f64_box: Box<f64> = Box::new(3.5);
    let f64_ref: &f64 = &*f64_box;

    zzz(); // #break
}

fn zzz() {()}
