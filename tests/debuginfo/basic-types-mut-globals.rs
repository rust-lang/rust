//@ compile-flags:-g
//@ disable-gdb-pretty-printers
//@ ignore-backends: gcc

// FIXME(f128): Merge `apple` revision once Apple releases Xcode with LLVM 22.
//@ revisions: not-apple apple
//@[not-apple] ignore-apple
//@[apple] only-apple
// `f128` support was added to `lldb` in version 22.
//@ min-llvm-lldb-version: 22

//@ gdb-command:run

// Check initializers
//@ gdb-command:print B
//@ gdb-check:$1 = false
//@ gdb-command:print I
//@ gdb-check:$2 = -1
//@ gdb-command:print C
//@ gdb-check:$3 = 97 'a'
//@ gdb-command:print I8
//@ gdb-check:$4 = 68
//@ gdb-command:print I16
//@ gdb-check:$5 = -16
//@ gdb-command:print I32
//@ gdb-check:$6 = -32
//@ gdb-command:print I64
//@ gdb-check:$7 = -64
//@ gdb-command:print U
//@ gdb-check:$8 = 1
//@ gdb-command:print U8
//@ gdb-check:$9 = 100
//@ gdb-command:print U16
//@ gdb-check:$10 = 16
//@ gdb-command:print U32
//@ gdb-check:$11 = 32
//@ gdb-command:print U64
//@ gdb-check:$12 = 64
//@ gdb-command:print F16
//@ gdb-check:$13 = 1.5
//@ gdb-command:print F32
//@ gdb-check:$14 = 2.5
//@ gdb-command:print F64
//@ gdb-check:$15 = 3.5
// FIXME(f128): gdb doesn't support Rust `f128` yet.
//@ gdb-command:continue

// Check new values
//@ gdb-command:print B
//@ gdb-check:$16 = true
//@ gdb-command:print I
//@ gdb-check:$17 = 2
//@ gdb-command:print C
//@ gdb-check:$18 = 102 'f'
//@ gdb-command:print I8
//@ gdb-check:$19 = 78
//@ gdb-command:print I16
//@ gdb-check:$20 = -26
//@ gdb-command:print I32
//@ gdb-check:$21 = -12
//@ gdb-command:print I64
//@ gdb-check:$22 = -54
//@ gdb-command:print U
//@ gdb-check:$23 = 5
//@ gdb-command:print U8
//@ gdb-check:$24 = 20
//@ gdb-command:print U16
//@ gdb-check:$25 = 32
//@ gdb-command:print U32
//@ gdb-check:$26 = 16
//@ gdb-command:print U64
//@ gdb-check:$27 = 128
//@ gdb-command:print F16
//@ gdb-check:$28 = 2.25
//@ gdb-command:print F32
//@ gdb-check:$29 = 5.75
//@ gdb-command:print F64
//@ gdb-check:$30 = 9.25
// FIXME(f128): gdb doesn't support Rust `f128` yet.

//@ lldb-command:run

// Check initializers
//@ lldb-command:v B
//@ lldb-check: ::B = false
//@ lldb-command:v I
//@ lldb-check: ::I = -1
//@ lldb-command:v C
//@ lldb-check: ::C = U+0x00000061 U'a'
//@ lldb-command:v/d I8
//@ lldb-check: ::I8 = 68
//@ lldb-command:v I16
//@ lldb-check: ::I16 = -16
//@ lldb-command:v I32
//@ lldb-check: ::I32 = -32
//@ lldb-command:v I64
//@ lldb-check: ::I64 = -64
//@ lldb-command:v U
//@ lldb-check: ::U = 1
//@ lldb-command:v/d U8
//@ lldb-check: ::U8 = 100
//@ lldb-command:v U16
//@ lldb-check: ::U16 = 16
//@ lldb-command:v U32
//@ lldb-check: ::U32 = 32
//@ lldb-command:v U64
//@ lldb-check: ::U64 = 64
//@ lldb-command:v F16
//@ lldb-check: ::F16 = 1.5
//@ lldb-command:v F32
//@ lldb-check: ::F32 = 2.5
//@ lldb-command:v F64
//@ lldb-check: ::F64 = 3.5
//@ lldb-command:v F128
//@[not-apple] lldb-check: ::F128 = 4.5
//@ lldb-command:continue

// Check new values
//@ lldb-command:v B
//@ lldb-check: ::B = true
//@ lldb-command:v I
//@ lldb-check: ::I = 2
//@ lldb-command:v C
//@ lldb-check: ::C = U+0x00000066 U'f'
//@ lldb-command:v/d I8
//@ lldb-check: ::I8 = 78
//@ lldb-command:v I16
//@ lldb-check: ::I16 = -26
//@ lldb-command:v I32
//@ lldb-check: ::I32 = -12
//@ lldb-command:v I64
//@ lldb-check: ::I64 = -54
//@ lldb-command:v U
//@ lldb-check: ::U = 5
//@ lldb-command:v/d U8
//@ lldb-check: ::U8 = 20
//@ lldb-command:v U16
//@ lldb-check: ::U16 = 32
//@ lldb-command:v U32
//@ lldb-check: ::U32 = 16
//@ lldb-command:v U64
//@ lldb-check: ::U64 = 128
//@ lldb-command:v F16
//@ lldb-check: ::F16 = 2.25
//@ lldb-command:v F32
//@ lldb-check: ::F32 = 5.75
//@ lldb-command:v F64
//@ lldb-check: ::F64 = 9.25
//@ lldb-command:v F128
//@[not-apple] lldb-check: ::F128 = 12.75

#![allow(unused_variables)]
#![feature(f16, f128)]

static mut B: bool = false;
static mut I: isize = -1;
static mut C: char = 'a';
static mut I8: i8 = 68;
static mut I16: i16 = -16;
static mut I32: i32 = -32;
static mut I64: i64 = -64;
static mut U: usize = 1;
static mut U8: u8 = 100;
static mut U16: u16 = 16;
static mut U32: u32 = 32;
static mut U64: u64 = 64;
static mut F16: f16 = 1.5;
static mut F32: f32 = 2.5;
static mut F64: f64 = 3.5;
static mut F128: f128 = 4.5;

fn main() {
    _zzz(); // #break

    unsafe {
        B = true;
        I = 2;
        C = 'f';
        I8 = 78;
        I16 = -26;
        I32 = -12;
        I64 = -54;
        U = 5;
        U8 = 20;
        U16 = 32;
        U32 = 16;
        U64 = 128;
        F16 = 2.25;
        F32 = 5.75;
        F64 = 9.25;
        F128 = 12.75;
    }

    _zzz(); // #break
}

fn _zzz() {()}
