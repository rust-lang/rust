//@ compile-flags:-g

//@ ignore-windows-gnu: #128981

// Note: u128 visualization was not supported in 10.0.22621.3233 but was fixed in 10.0.26100.2161.

// FIXME(#133107): this is temporarily marked as `only-64bit` because of course 32-bit msvc has
// a different integer width and thus underlying integer type display. Only marked as such to
// unblock the tree.
//@ only-64bit
//@ min-cdb-version: 10.0.26100.2161

// Tests the visualizations for `NonZero<T>`, `Wrapping<T>` and
// `Atomic{Bool,I8,I16,I32,I64,Isize,U8,U16,U32,U64,Usize}` located in `libcore.natvis`.

// === CDB TESTS ==================================================================================
// cdb-command: g

// cdb-command: dx nz_i8
// cdb-check:nz_i8            : 11 [Type: core::num::nonzero::NonZero<i8>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<i8>]

// cdb-command: dx nz_i16
// cdb-check:nz_i16           : 22 [Type: core::num::nonzero::NonZero<i16>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<i16>]

// cdb-command: dx nz_i32
// cdb-check:nz_i32           : 33 [Type: core::num::nonzero::NonZero<i32>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<i32>]

// cdb-command: dx nz_i64
// cdb-check:nz_i64           : 44 [Type: core::num::nonzero::NonZero<i64>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<i64>]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx nz_i128
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<i128>]

// cdb-command: dx nz_isize
// cdb-check:nz_isize         : 66 [Type: core::num::nonzero::NonZero<isize>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<isize>]

// cdb-command: dx nz_u8
// cdb-check:nz_u8            : 0x4d [Type: core::num::nonzero::NonZero<u8>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<u8>]

// cdb-command: dx nz_u16
// cdb-check:nz_u16           : 0x58 [Type: core::num::nonzero::NonZero<u16>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<u16>]

// cdb-command: dx nz_u32
// cdb-check:nz_u32           : 0x63 [Type: core::num::nonzero::NonZero<u32>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<u32>]

// cdb-command: dx nz_u64
// cdb-check:nz_u64           : 0x64 [Type: core::num::nonzero::NonZero<u64>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<u64>]

// cdb-command: dx nz_u128
// cdb-check:nz_u128          : 111 [Type: core::num::nonzero::NonZero<u128>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<u128>]

// cdb-command: dx nz_usize
// cdb-check:nz_usize         : 0x7a [Type: core::num::nonzero::NonZero<usize>]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZero<usize>]

// cdb-command: dx w_i8
// cdb-check:w_i8             : 10 [Type: core::num::wrapping::Wrapping<i8>]
// cdb-check:    [+0x000] __0              : 10 [Type: char]

// cdb-command: dx w_i16
// cdb-check:w_i16            : 20 [Type: core::num::wrapping::Wrapping<i16>]
// cdb-check:    [+0x000] __0              : 20 [Type: short]

// cdb-command: dx w_i32
// cdb-check:w_i32            : 30 [Type: core::num::wrapping::Wrapping<i32>]
// cdb-check:    [+0x000] __0              : 30 [Type: int]

// cdb-command: dx w_i64
// cdb-check:w_i64            : 40 [Type: core::num::wrapping::Wrapping<i64>]
// cdb-check:    [+0x000] __0              : 40 [Type: __int64]

// cdb-command: dx w_i128
// cdb-check:w_i128           : 50 [Type: core::num::wrapping::Wrapping<i128>]
// cdb-check:    [+0x000] __0              : 50 [Type: i128]

// cdb-command: dx w_isize
// cdb-check:w_isize          : 60 [Type: core::num::wrapping::Wrapping<isize>]
// cdb-check:    [+0x000] __0              : 60 [Type: __int64]

// cdb-command: dx w_u8
// cdb-check:w_u8             : 0x46 [Type: core::num::wrapping::Wrapping<u8>]
// cdb-check:    [+0x000] __0              : 0x46 [Type: unsigned char]

// cdb-command: dx w_u16
// cdb-check:w_u16            : 0x50 [Type: core::num::wrapping::Wrapping<u16>]
// cdb-check:    [+0x000] __0              : 0x50 [Type: unsigned short]

// cdb-command: dx w_u32
// cdb-check:w_u32            : 0x5a [Type: core::num::wrapping::Wrapping<u32>]
// cdb-check:    [+0x000] __0              : 0x5a [Type: unsigned int]

// cdb-command: dx w_u64
// cdb-check:w_u64            : 0x64 [Type: core::num::wrapping::Wrapping<u64>]
// cdb-check:    [+0x000] __0              : 0x64 [Type: unsigned __int64]

// cdb-command: dx w_u128
// cdb-check:w_u128           : 110 [Type: core::num::wrapping::Wrapping<u128>]
// cdb-check:    [+0x000] __0              : 110 [Type: u128]

// cdb-command: dx w_usize
// cdb-check:w_usize          : 0x78 [Type: core::num::wrapping::Wrapping<usize>]
// cdb-check:    [+0x000] __0              : 0x78 [Type: unsigned __int64]

// cdb-command: dx a_bool_t
// cdb-check:a_bool_t         : true [Type: core::sync::atomic::AtomicBool]
// cdb-check:    [+0x000] v                : 0x1 [Type: core::cell::UnsafeCell<u8>]

// cdb-command: dx a_bool_f
// cdb-check:a_bool_f         : false [Type: core::sync::atomic::AtomicBool]
// cdb-check:    [+0x000] v                : 0x0 [Type: core::cell::UnsafeCell<u8>]

// cdb-command: dx a_i8
// cdb-check:a_i8             : 2 [Type: core::sync::atomic::AtomicI8]
// cdb-check:    [+0x000] v                : 2 [Type: core::cell::UnsafeCell<i8>]

// cdb-command: dx a_i16
// cdb-check:a_i16            : 4 [Type: core::sync::atomic::AtomicI16]
// cdb-check:    [+0x000] v                : 4 [Type: core::cell::UnsafeCell<i16>]

// cdb-command: dx a_i32
// cdb-check:a_i32            : 8 [Type: core::sync::atomic::AtomicI32]
// cdb-check:    [+0x000] v                : 8 [Type: core::cell::UnsafeCell<i32>]

// cdb-command: dx a_i64
// cdb-check:a_i64            : 16 [Type: core::sync::atomic::AtomicI64]
// cdb-check:    [+0x000] v                : 16 [Type: core::cell::UnsafeCell<i64>]

// cdb-command: dx a_isize
// cdb-check:a_isize          : 32 [Type: core::sync::atomic::AtomicIsize]
// cdb-check:    [+0x000] v                : 32 [Type: core::cell::UnsafeCell<isize>]

// cdb-command: dx a_u8
// cdb-check:a_u8             : 0x40 [Type: core::sync::atomic::AtomicU8]
// cdb-check:    [+0x000] v                : 0x40 [Type: core::cell::UnsafeCell<u8>]

// cdb-command: dx a_u16
// cdb-check:a_u16            : 0x80 [Type: core::sync::atomic::AtomicU16]
// cdb-check:    [+0x000] v                : 0x80 [Type: core::cell::UnsafeCell<u16>]

// cdb-command: dx a_u32
// cdb-check:a_u32            : 0x100 [Type: core::sync::atomic::AtomicU32]
// cdb-check:    [+0x000] v                : 0x100 [Type: core::cell::UnsafeCell<u32>]

// cdb-command: dx a_u64
// cdb-check:a_u64            : 0x200 [Type: core::sync::atomic::AtomicU64]
// cdb-check:    [+0x000] v                : 0x200 [Type: core::cell::UnsafeCell<u64>]

// cdb-command: dx a_usize
// cdb-check:a_usize          : 0x400 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [+0x000] v                : 0x400 [Type: core::cell::UnsafeCell<usize>]


// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print/d nz_i8
// gdb-check:[...]$1 = 11

// gdb-command:print nz_i16
// gdb-check:[...]$2 = 22

// gdb-command:print nz_i32
// gdb-check:[...]$3 = 33

// gdb-command:print nz_i64
// gdb-check:[...]$4 = 44

// gdb-command:print nz_i128
// gdb-check:[...]$5 = 55

// gdb-command:print nz_isize
// gdb-check:[...]$6 = 66

// gdb-command:print/d nz_u8
// gdb-check:[...]$7 = 77

// gdb-command:print nz_u16
// gdb-check:[...]$8 = 88

// gdb-command:print nz_u32
// gdb-check:[...]$9 = 99

// gdb-command:print nz_u64
// gdb-check:[...]$10 = 100

// gdb-command:print nz_u128
// gdb-check:[...]$11 = 111

// gdb-command:print nz_usize
// gdb-check:[...]$12 = 122



// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v/d nz_i8
// lldb-check:[...] 11 { __0 = { 0 = 11 } }

// lldb-command:v nz_i16
// lldb-check:[...] 22 { __0 = { 0 = 22 } }

// lldb-command:v nz_i32
// lldb-check:[...] 33 { __0 = { 0 = 33 } }

// lldb-command:v nz_i64
// lldb-check:[...] 44 { __0 = { 0 = 44 } }

// lldb-command:v nz_i128
// lldb-check:[...] 55 { __0 = { 0 = 55 } }

// lldb-command:v nz_isize
// lldb-check:[...] 66 { __0 = { 0 = 66 } }

// lldb-command:v/d nz_u8
// lldb-check:[...] 77 { __0 = { 0 = 77 } }

// lldb-command:v nz_u16
// lldb-check:[...] 88 { __0 = { 0 = 88 } }

// lldb-command:v nz_u32
// lldb-check:[...] 99 { __0 = { 0 = 99 } }

// lldb-command:v nz_u64
// lldb-check:[...] 100 { __0 = { 0 = 100 } }

// lldb-command:v nz_u128
// lldb-check:[...] 111 { __0 = { 0 = 111 } }

// lldb-command:v nz_usize
// lldb-check:[...] 122 { __0 = { 0 = 122 } }

use std::num::*;
use std::sync::atomic::*;

fn main() {
    let nz_i8 = NonZero::new(11i8).unwrap();
    let nz_i16 = NonZero::new(22i16).unwrap();
    let nz_i32 = NonZero::new(33i32).unwrap();
    let nz_i64 = NonZero::new(44i64).unwrap();
    let nz_i128 = NonZero::new(55i128).unwrap();
    let nz_isize = NonZero::new(66isize).unwrap();

    let nz_u8 = NonZero::new(77u8).unwrap();
    let nz_u16 = NonZero::new(88u16).unwrap();
    let nz_u32 = NonZero::new(99u32).unwrap();
    let nz_u64 = NonZero::new(100u64).unwrap();
    let nz_u128 = NonZero::new(111u128).unwrap();
    let nz_usize = NonZero::new(122usize).unwrap();

    let w_i8 = Wrapping(10i8);
    let w_i16 = Wrapping(20i16);
    let w_i32 = Wrapping(30i32);
    let w_i64 = Wrapping(40i64);
    let w_i128 = Wrapping(50i128);
    let w_isize = Wrapping(60isize);

    let w_u8 = Wrapping(70u8);
    let w_u16 = Wrapping(80u16);
    let w_u32 = Wrapping(90u32);
    let w_u64 = Wrapping(100u64);
    let w_u128 = Wrapping(110u128);
    let w_usize = Wrapping(120usize);

    let a_bool_t = AtomicBool::new(true);
    let a_bool_f = AtomicBool::new(false);

    let a_i8 = AtomicI8::new(2);
    let a_i16 = AtomicI16::new(4);
    let a_i32 = AtomicI32::new(8);
    let a_i64 = AtomicI64::new(16);
    let a_isize = AtomicIsize::new(32);

    let a_u8 = AtomicU8::new(64);
    let a_u16 = AtomicU16::new(128);
    let a_u32 = AtomicU32::new(256);
    let a_u64 = AtomicU64::new(512);
    let a_usize = AtomicUsize::new(1024);

    zzz(); // #break
}

fn zzz() { }
