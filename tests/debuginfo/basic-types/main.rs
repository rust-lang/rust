// Caveats - gdb prints any 8-bit value (meaning rust i8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

//@ compile-flags:-g
//@ disable-gdb-pretty-printers
//@ ignore-backends: gcc
//@ min-llvm-lldb-version: 21.1.0

// This version corresponds to swift 6.2.3/lldb 19.1.5
//@ min-apple-lldb-version: 1703.0.236.21

//@ min-gdb-version: 16.1

// === GDB TESTS ===================================================================================

//@ gdb-command:run
//@ gdb-repr:b
//@ gdb-repr:b
//@ gdb-repr:i
//@ gdb-repr:c
//@ gdb-repr:i8
//@ gdb-repr:i16
//@ gdb-repr:i32
//@ gdb-repr:i64
//@ gdb-repr:u
//@ gdb-repr:u8
//@ gdb-repr:u16
//@ gdb-repr:u32
//@ gdb-repr:u64
//@ gdb-repr:f16
//@ gdb-repr:f32
//@ gdb-repr:f64
//@ gdb-repr:s

// === LLDB TESTS ==================================================================================

//@ lldb-command:run
//@ lldb-repr:b
//@ lldb-repr:i
//@ lldb-repr:c
//@ lldb-repr:i8
//@ lldb-repr:i16
//@ lldb-repr:i32
//@ lldb-repr:i64
//@ lldb-repr:u
//@ lldb-repr:u8
//@ lldb-repr:u16
//@ lldb-repr:u32
//@ lldb-repr:u64
//@ lldb-repr:f32
//@ lldb-repr:f64

// === CDB TESTS ===================================================================================

//@ cdb-command:g
//@ cdb-command:dx b
//@ cdb-check:b                : false [Type: bool]
//@ cdb-command:dx i
//@ cdb-check:i                : -1 [Type: [...]]
//@ cdb-command:dx c
//@ cdb-check:c                : 0x61 'a' [Type: char32_t]
//@ cdb-command:dx i8
//@ cdb-check:i8               : 68 [Type: char]
//@ cdb-command:dx i16
//@ cdb-check:i16              : -16 [Type: short]
//@ cdb-command:dx i32
//@ cdb-check:i32              : -32 [Type: int]
//@ cdb-command:dx i64
//@ cdb-check:i64              : -64 [Type: __int64]
//@ cdb-command:dx u
//@ cdb-check:u                : 0x1 [Type: [...]]
//@ cdb-command:dx u8
//@ cdb-check:u8               : 0x64 [Type: unsigned char]
//@ cdb-command:dx u16
//@ cdb-check:u16              : 0x10 [Type: unsigned short]
//@ cdb-command:dx u32
//@ cdb-check:u32              : 0x20 [Type: unsigned int]
//@ cdb-command:dx u64
//@ cdb-check:u64              : 0x40 [Type: unsigned __int64]
//@ cdb-command:dx f16
//@ cdb-check:f16              : 1.500000 [Type: f16]
//@ cdb-command:dx f32
//@ cdb-check:f32              : 2.500000 [Type: float]
//@ cdb-command:dx f64
//@ cdb-check:f64              : 3.500000 [Type: double]
//@ cdb-command:.enable_unicode 1
// FIXME(#88840): The latest version of the Windows SDK broke the visualizer for str.
//@ cdb-command:dx  s
//@ cdb-check:s                : [...] [Type: ref$<str$>]

#![allow(unused_variables)]
#![feature(f16)]

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
    let f16: f16 = 1.5;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    let s: &str = "Hello, World!";
    _zzz(); // #break
}

fn _zzz() {
    ()
}
