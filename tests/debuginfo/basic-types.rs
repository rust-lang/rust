// Caveats - gdb prints any 8-bit value (meaning rust i8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = false
// gdb-command:print i
// gdb-check:$2 = -1
// gdb-command:print c
// gdb-check:$3 = 97 'a'
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
// gdb-command:print f16
// gdb-check:$13 = 1.5
// gdb-command:print f32
// gdb-check:$14 = 2.5
// gdb-command:print f64
// gdb-check:$15 = 3.5
// gdb-command:print s
// gdb-check:$16 = "Hello, World!"

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v b
// lldb-check:[...] false
// lldb-command:v i
// lldb-check:[...] -1

// lldb-command:v i8
// lldb-check:[...] 'D'
// lldb-command:v i16
// lldb-check:[...] -16
// lldb-command:v i32
// lldb-check:[...] -32
// lldb-command:v i64
// lldb-check:[...] -64
// lldb-command:v u
// lldb-check:[...] 1
// lldb-command:v u8
// lldb-check:[...] 'd'
// lldb-command:v u16
// lldb-check:[...] 16
// lldb-command:v u32
// lldb-check:[...] 32
// lldb-command:v u64
// lldb-check:[...] 64
// lldb-command:v f32
// lldb-check:[...] 2.5
// lldb-command:v f64
// lldb-check:[...] 3.5

// === CDB TESTS ===================================================================================

// cdb-command:g
// cdb-command:dx b
// cdb-check:b                : false [Type: bool]
// cdb-command:dx i
// cdb-check:i                : -1 [Type: [...]]
// cdb-command:dx c
// cdb-check:c                : 0x61 'a' [Type: char32_t]
// cdb-command:dx i8
// cdb-check:i8               : 68 [Type: char]
// cdb-command:dx i16
// cdb-check:i16              : -16 [Type: short]
// cdb-command:dx i32
// cdb-check:i32              : -32 [Type: int]
// cdb-command:dx i64
// cdb-check:i64              : -64 [Type: __int64]
// cdb-command:dx u
// cdb-check:u                : 0x1 [Type: [...]]
// cdb-command:dx u8
// cdb-check:u8               : 0x64 [Type: unsigned char]
// cdb-command:dx u16
// cdb-check:u16              : 0x10 [Type: unsigned short]
// cdb-command:dx u32
// cdb-check:u32              : 0x20 [Type: unsigned int]
// cdb-command:dx u64
// cdb-check:u64              : 0x40 [Type: unsigned __int64]
// cdb-command:dx f16
// cdb-check:f16              : 1.500000 [Type: f16]
// cdb-command:dx f32
// cdb-check:f32              : 2.500000 [Type: float]
// cdb-command:dx f64
// cdb-check:f64              : 3.500000 [Type: double]
// cdb-command:.enable_unicode 1
// FIXME(#88840): The latest version of the Windows SDK broke the visualizer for str.
// cdb-command:dx  s
// cdb-check:s                : [...] [Type: ref$<str$>]

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
