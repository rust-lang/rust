// compile-flags:-g

// min-gdb-version: 8.1
// ignore-windows-gnu // emit_debug_gdb_scripts is disabled on Windows

// Tests the visualizations for `NonZero{I,U}{8,16,32,64,128,size}`, `Wrapping<T>` and
// `Atomic{Bool,I8,I16,I32,I64,Isize,U8,U16,U32,U64,Usize}` located in `libcore.natvis`.

// === CDB TESTS ==================================================================================
// cdb-command: g

// cdb-command: dx nz_i8
// cdb-check:nz_i8            : 11 [Type: core::num::nonzero::NonZeroI8]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI8]

// cdb-command: dx nz_i16
// cdb-check:nz_i16           : 22 [Type: core::num::nonzero::NonZeroI16]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI16]

// cdb-command: dx nz_i32
// cdb-check:nz_i32           : 33 [Type: core::num::nonzero::NonZeroI32]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI32]

// cdb-command: dx nz_i64
// cdb-check:nz_i64           : 44 [Type: core::num::nonzero::NonZeroI64]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI64]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx nz_i128
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI128]

// cdb-command: dx nz_isize
// cdb-check:nz_isize         : 66 [Type: core::num::nonzero::NonZeroIsize]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroIsize]

// cdb-command: dx nz_u8
// cdb-check:nz_u8            : 0x4d [Type: core::num::nonzero::NonZeroU8]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU8]

// cdb-command: dx nz_u16
// cdb-check:nz_u16           : 0x58 [Type: core::num::nonzero::NonZeroU16]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU16]

// cdb-command: dx nz_u32
// cdb-check:nz_u32           : 0x63 [Type: core::num::nonzero::NonZeroU32]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU32]

// cdb-command: dx nz_u64
// cdb-check:nz_u64           : 0x64 [Type: core::num::nonzero::NonZeroU64]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU64]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx nz_u128
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU128]

// cdb-command: dx nz_usize
// cdb-check:nz_usize         : 0x7a [Type: core::num::nonzero::NonZeroUsize]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroUsize]

// cdb-command: dx w_i8
// cdb-check:w_i8             : 10 [Type: core::num::wrapping::Wrapping<i8>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i8>]

// cdb-command: dx w_i16
// cdb-check:w_i16            : 20 [Type: core::num::wrapping::Wrapping<i16>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i16>]

// cdb-command: dx w_i32
// cdb-check:w_i32            : 30 [Type: core::num::wrapping::Wrapping<i32>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i32>]

// cdb-command: dx w_i64
// cdb-check:w_i64            : 40 [Type: core::num::wrapping::Wrapping<i64>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i64>]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx w_i128
// cdb-check:w_i128           [Type: core::num::wrapping::Wrapping<i128>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i128>]

// cdb-command: dx w_isize
// cdb-check:w_isize          : 60 [Type: core::num::wrapping::Wrapping<isize>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<isize>]

// cdb-command: dx w_u8
// cdb-check:w_u8             : 0x46 [Type: core::num::wrapping::Wrapping<u8>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u8>]

// cdb-command: dx w_u16
// cdb-check:w_u16            : 0x50 [Type: core::num::wrapping::Wrapping<u16>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u16>]

// cdb-command: dx w_u32
// cdb-check:w_u32            : 0x5a [Type: core::num::wrapping::Wrapping<u32>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u32>]

// cdb-command: dx w_u64
// cdb-check:w_u64            : 0x64 [Type: core::num::wrapping::Wrapping<u64>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u64>]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx w_u128
// cdb-check:w_u128           [Type: core::num::wrapping::Wrapping<u128>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u128>]

// cdb-command: dx w_usize
// cdb-check:w_usize          : 0x78 [Type: core::num::wrapping::Wrapping<usize>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<usize>]

// cdb-command: dx a_bool_t
// cdb-check:a_bool_t         : true [Type: core::sync::atomic::AtomicBool]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicBool]

// cdb-command: dx a_bool_f
// cdb-check:a_bool_f         : false [Type: core::sync::atomic::AtomicBool]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicBool]

// cdb-command: dx a_i8
// cdb-check:a_i8             : 2 [Type: core::sync::atomic::AtomicI8]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicI8]

// cdb-command: dx a_i16
// cdb-check:a_i16            : 4 [Type: core::sync::atomic::AtomicI16]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicI16]

// cdb-command: dx a_i32
// cdb-check:a_i32            : 8 [Type: core::sync::atomic::AtomicI32]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicI32]

// cdb-command: dx a_i64
// cdb-check:a_i64            : 16 [Type: core::sync::atomic::AtomicI64]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicI64]

// cdb-command: dx a_isize
// cdb-check:a_isize          : 32 [Type: core::sync::atomic::AtomicIsize]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicIsize]

// cdb-command: dx a_u8
// cdb-check:a_u8             : 0x40 [Type: core::sync::atomic::AtomicU8]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicU8]

// cdb-command: dx a_u16
// cdb-check:a_u16            : 0x80 [Type: core::sync::atomic::AtomicU16]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicU16]

// cdb-command: dx a_u32
// cdb-check:a_u32            : 0x100 [Type: core::sync::atomic::AtomicU32]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicU32]

// cdb-command: dx a_u64
// cdb-check:a_u64            : 0x200 [Type: core::sync::atomic::AtomicU64]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicU64]

// cdb-command: dx a_usize
// cdb-check:a_usize          : 0x400 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [<Raw View>]     [Type: core::sync::atomic::AtomicUsize]


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

// lldb-command:print/d nz_i8
// lldb-check:[...]$0 = 11 { __0 = 11 }

// lldb-command:print nz_i16
// lldb-check:[...]$1 = 22 { __0 = 22 }

// lldb-command:print nz_i32
// lldb-check:[...]$2 = 33 { __0 = 33 }

// lldb-command:print nz_i64
// lldb-check:[...]$3 = 44 { __0 = 44 }

// lldb-command:print nz_i128
// lldb-check:[...]$4 = 55 { __0 = 55 }

// lldb-command:print nz_isize
// lldb-check:[...]$5 = 66 { __0 = 66 }

// lldb-command:print/d nz_u8
// lldb-check:[...]$6 = 77 { __0 = 77 }

// lldb-command:print nz_u16
// lldb-check:[...]$7 = 88 { __0 = 88 }

// lldb-command:print nz_u32
// lldb-check:[...]$8 = 99 { __0 = 99 }

// lldb-command:print nz_u64
// lldb-check:[...]$9 = 100 { __0 = 100 }

// lldb-command:print nz_u128
// lldb-check:[...]$10 = 111 { __0 = 111 }

// lldb-command:print nz_usize
// lldb-check:[...]$11 = 122 { __0 = 122 }


use std::num::*;
use std::sync::atomic::*;

fn main() {
    let nz_i8 = NonZeroI8::new(11).unwrap();
    let nz_i16 = NonZeroI16::new(22).unwrap();
    let nz_i32 = NonZeroI32::new(33).unwrap();
    let nz_i64 = NonZeroI64::new(44).unwrap();
    let nz_i128 = NonZeroI128::new(55).unwrap();
    let nz_isize = NonZeroIsize::new(66).unwrap();

    let nz_u8 = NonZeroU8::new(77).unwrap();
    let nz_u16 = NonZeroU16::new(88).unwrap();
    let nz_u32 = NonZeroU32::new(99).unwrap();
    let nz_u64 = NonZeroU64::new(100).unwrap();
    let nz_u128 = NonZeroU128::new(111).unwrap();
    let nz_usize = NonZeroUsize::new(122).unwrap();

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
