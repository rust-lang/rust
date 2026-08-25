//@ revisions: lto no-lto lto-apple no-lto-apple

//@ compile-flags:-g --crate-name=basic_types_globals
//@ disable-gdb-pretty-printers

// FIXME(f128): Merge `-apple` revisions once Apple releases Xcode with LLVM 22.
//@ [lto] ignore-apple
//@ [no-lto] ignore-apple
//@ [lto-apple] only-apple
//@ [no-lto-apple] only-apple
//@ [lto] compile-flags:-C lto
//@ [lto] no-prefer-dynamic
//@ [lto-apple] compile-flags:-C lto
//@ [lto-apple] no-prefer-dynamic
//@ ignore-backends: gcc
// `f128` support was added to `lldb` in version 22.
//@ min-llvm-lldb-version: 22

//@ lldb-command:run
//@ lldb-command:v basic_types_globals::B
//@ lldb-check:[...]basic_types_globals::B = false
//@ lldb-command:v basic_types_globals::I
//@ lldb-check:[...]basic_types_globals::I = -1
//@ lldb-command:v basic_types_globals::C
//@ lldb-check:[...]basic_types_globals::C = U+0x00000061 U'a'
//@ lldb-command:v basic_types_globals::I8
//@ lldb-check:[...]basic_types_globals::I8 = 68
//@ lldb-command:v basic_types_globals::I16
//@ lldb-check:[...]basic_types_globals::I16 = -16
//@ lldb-command:v basic_types_globals::I32
//@ lldb-check:[...]basic_types_globals::I32 = -32
//@ lldb-command:v basic_types_globals::I64
//@ lldb-check:[...]basic_types_globals::I64 = -64
//@ lldb-command:v basic_types_globals::U
//@ lldb-check:[...]basic_types_globals::U = 1
//@ lldb-command:v basic_types_globals::U8
//@ lldb-check:[...]basic_types_globals::U8 = 100
//@ lldb-command:v basic_types_globals::U16
//@ lldb-check:[...]basic_types_globals::U16 = 16
//@ lldb-command:v basic_types_globals::U32
//@ lldb-check:[...]basic_types_globals::U32 = 32
//@ lldb-command:v basic_types_globals::U64
//@ lldb-check:[...]basic_types_globals::U64 = 64
//@ lldb-command:v basic_types_globals::F16
//@ lldb-check:[...]basic_types_globals::F16 = 1.5
//@ lldb-command:v basic_types_globals::F32
//@ lldb-check:[...]basic_types_globals::F32 = 2.5
//@ lldb-command:v basic_types_globals::F64
//@ lldb-check:[...]basic_types_globals::F64 = 3.5
//@ lldb-command:v F128
//@[no-lto] lldb-check:[...]basic_types_globals::F128 = 4.5
//@[lto] lldb-check:[...]basic_types_globals::F128 = 4.5

//@ gdb-command:run
//@ gdb-command:print B
//@ gdb-check:$1 = false
//@ gdb-command:print I
//@ gdb-check:$2 = -1
//@ gdb-command:print/d C
//@ gdb-check:$3 = 97
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

#![allow(unused_variables)]
#![feature(f16, f128)]

// N.B. These are `mut` only so they don't constant fold away.
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

    let a = unsafe { (B, I, C, I8, I16, I32, I64, U, U8, U16, U32, U64, F32, F64, F128) };
    // FIXME(f16): Including f16 and f32 in the same tuple emits `__gnu_h2f_ieee`, which
    // does not exist on some targets like PowerPC (fixed in llvm22).
    // See https://github.com/llvm/llvm-project/issues/97981 and
    // https://github.com/rust-lang/compiler-builtins/issues/655
    let b = unsafe { F16 };
}

fn _zzz() {
    ()
}
