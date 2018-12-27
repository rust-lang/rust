// Caveats - gdb prints any 8-bit value (meaning rust i8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// min-lldb-version: 310

// This fails on lldb 6.0.1 on x86-64 Fedora 28; so mark it macOS-only
// for now.
// only-macos

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = false
// gdb-command:print i
// gdb-check:$2 = -1
// gdb-command:print c
// gdbg-check:$3 = 97
// gdbr-check:$3 = 97 'a'
// gdb-command:print/d i8
// gdb-check:$4 = 68
// gdb-command:print i16
// gdb-check:$5 = -16
// gdb-command:print i32
// gdb-check:$6 = -32
// gdb-command:print i64
// gdb-check:$7 = -64
// gdb-command:print u
// gdb-check:$8 = 1
// gdb-command:print/d u8
// gdb-check:$9 = 100
// gdb-command:print u16
// gdb-check:$10 = 16
// gdb-command:print u32
// gdb-check:$11 = 32
// gdb-command:print u64
// gdb-check:$12 = 64
// gdb-command:print f32
// gdb-check:$13 = 2.5
// gdb-command:print f64
// gdb-check:$14 = 3.5


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print b
// lldbg-check:[...]$0 = false
// lldbr-check:(bool) b = false
// lldb-command:print i
// lldbg-check:[...]$1 = -1
// lldbr-check:(isize) i = -1

// NOTE: only rust-enabled lldb supports 32bit chars
// lldbr-command:print c
// lldbr-check:(char) c = 'a'

// lldb-command:print i8
// lldbg-check:[...]$2 = 'D'
// lldbr-check:(i8) i8 = 68
// lldb-command:print i16
// lldbg-check:[...]$3 = -16
// lldbr-check:(i16) i16 = -16
// lldb-command:print i32
// lldbg-check:[...]$4 = -32
// lldbr-check:(i32) i32 = -32
// lldb-command:print i64
// lldbg-check:[...]$5 = -64
// lldbr-check:(i64) i64 = -64
// lldb-command:print u
// lldbg-check:[...]$6 = 1
// lldbr-check:(usize) u = 1
// lldb-command:print u8
// lldbg-check:[...]$7 = 'd'
// lldbr-check:(u8) u8 = 100
// lldb-command:print u16
// lldbg-check:[...]$8 = 16
// lldbr-check:(u16) u16 = 16
// lldb-command:print u32
// lldbg-check:[...]$9 = 32
// lldbr-check:(u32) u32 = 32
// lldb-command:print u64
// lldbg-check:[...]$10 = 64
// lldbr-check:(u64) u64 = 64
// lldb-command:print f32
// lldbg-check:[...]$11 = 2.5
// lldbr-check:(f32) f32 = 2.5
// lldb-command:print f64
// lldbg-check:[...]$12 = 3.5
// lldbr-check:(f64) f64 = 3.5

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let b: bool = false;
    let i: isize = -1;
    let c: char = 'a';
    let i8: i8 = 68;
    let i16: i16 = -16;
    let i32: i32 = -32;
    let i64: i64 = -64;
    let u: usize = 1;
    let u8: u8 = 100;
    let u16: u16 = 16;
    let u32: u32 = 32;
    let u64: u64 = 64;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    _zzz(); // #break
}

fn _zzz() {()}
