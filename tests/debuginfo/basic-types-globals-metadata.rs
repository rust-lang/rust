//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// gdb-command:run
// gdb-command:whatis basic_types_globals_metadata::B
// gdb-check:type = bool
// gdb-command:whatis basic_types_globals_metadata::I
// gdb-check:type = isize
// gdb-command:whatis basic_types_globals_metadata::C
// gdb-check:type = char
// gdb-command:whatis basic_types_globals_metadata::I8
// gdb-check:type = i8
// gdb-command:whatis basic_types_globals_metadata::I16
// gdb-check:type = i16
// gdb-command:whatis basic_types_globals_metadata::I32
// gdb-check:type = i32
// gdb-command:whatis basic_types_globals_metadata::I64
// gdb-check:type = i64
// gdb-command:whatis basic_types_globals_metadata::U
// gdb-check:type = usize
// gdb-command:whatis basic_types_globals_metadata::U8
// gdb-check:type = u8
// gdb-command:whatis basic_types_globals_metadata::U16
// gdb-check:type = u16
// gdb-command:whatis basic_types_globals_metadata::U32
// gdb-check:type = u32
// gdb-command:whatis basic_types_globals_metadata::U64
// gdb-check:type = u64
// gdb-command:whatis basic_types_globals_metadata::F16
// gdb-check:type = f16
// gdb-command:whatis basic_types_globals_metadata::F32
// gdb-check:type = f32
// gdb-command:whatis basic_types_globals_metadata::F64
// gdb-check:type = f64
// gdb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(f16)]

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

fn main() {
    _zzz(); // #break

    let a = unsafe { (B, I, C, I8, I16, I32, I64, U, U8, U16, U32, U64, F32, F64) };
    // FIXME: Including f16 and f32 in the same tuple emits `__gnu_h2f_ieee`, which
    // does not exist on some targets like PowerPC.
    // See https://github.com/llvm/llvm-project/issues/97981 and
    // https://github.com/rust-lang/compiler-builtins/issues/655
    let b = unsafe { F16 };
}

fn _zzz() {()}
