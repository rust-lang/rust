// Caveats - gdb prints any 8-bit value (meaning rust I8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// min-lldb-version: 310
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155

// compile-flags:-g
// gdb-command:run
// gdbg-command:print 'basic_types_globals::B'
// gdbr-command:print B
// gdb-check:$1 = false
// gdbg-command:print 'basic_types_globals::I'
// gdbr-command:print I
// gdb-check:$2 = -1
// gdbg-command:print 'basic_types_globals::C'
// gdbr-command:print C
// gdbg-check:$3 = 97
// gdbr-check:$3 = 97 'a'
// gdbg-command:print/d 'basic_types_globals::I8'
// gdbr-command:print I8
// gdb-check:$4 = 68
// gdbg-command:print 'basic_types_globals::I16'
// gdbr-command:print I16
// gdb-check:$5 = -16
// gdbg-command:print 'basic_types_globals::I32'
// gdbr-command:print I32
// gdb-check:$6 = -32
// gdbg-command:print 'basic_types_globals::I64'
// gdbr-command:print I64
// gdb-check:$7 = -64
// gdbg-command:print 'basic_types_globals::U'
// gdbr-command:print U
// gdb-check:$8 = 1
// gdbg-command:print/d 'basic_types_globals::U8'
// gdbr-command:print U8
// gdb-check:$9 = 100
// gdbg-command:print 'basic_types_globals::U16'
// gdbr-command:print U16
// gdb-check:$10 = 16
// gdbg-command:print 'basic_types_globals::U32'
// gdbr-command:print U32
// gdb-check:$11 = 32
// gdbg-command:print 'basic_types_globals::U64'
// gdbr-command:print U64
// gdb-check:$12 = 64
// gdbg-command:print 'basic_types_globals::F32'
// gdbr-command:print F32
// gdb-check:$13 = 2.5
// gdbg-command:print 'basic_types_globals::F64'
// gdbr-command:print F64
// gdb-check:$14 = 3.5
// gdb-command:continue

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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
static mut F32: f32 = 2.5;
static mut F64: f64 = 3.5;

fn main() {
    _zzz(); // #break

    let a = unsafe { (B, I, C, I8, I16, I32, I64, U, U8, U16, U32, U64, F32, F64) };
}

fn _zzz() {()}
