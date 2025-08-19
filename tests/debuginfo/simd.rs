// Need a fix for LLDB first...
//@ ignore-lldb

// FIXME: LLVM generates invalid debug info for variables requiring
// dynamic stack realignment, which is the case on s390x for vector
// types with non-vector ABI.
//@ ignore-s390x

//@ compile-flags:-g
//@ disable-gdb-pretty-printers
// gdb-command:run

// gdb-command:print vi8x16
// gdb-check:$1 = simd::i8x16 ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
// gdb-command:print vi16x8
// gdb-check:$2 = simd::i16x8 ([16, 17, 18, 19, 20, 21, 22, 23])
// gdb-command:print vi32x4
// gdb-check:$3 = simd::i32x4 ([24, 25, 26, 27])
// gdb-command:print vi64x2
// gdb-check:$4 = simd::i64x2 ([28, 29])

// gdb-command:print vu8x16
// gdb-check:$5 = simd::u8x16 ([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
// gdb-command:print vu16x8
// gdb-check:$6 = simd::u16x8 ([46, 47, 48, 49, 50, 51, 52, 53])
// gdb-command:print vu32x4
// gdb-check:$7 = simd::u32x4 ([54, 55, 56, 57])
// gdb-command:print vu64x2
// gdb-check:$8 = simd::u64x2 ([58, 59])

// gdb-command:print vf32x4
// gdb-check:$9 = simd::f32x4 ([60.5, 61.5, 62.5, 63.5])
// gdb-command:print vf64x2
// gdb-check:$10 = simd::f64x2 ([64.5, 65.5])

// gdb-command:continue

#![allow(unused_variables)]
#![feature(repr_simd)]

#[repr(simd)]
struct i8x16([i8; 16]);
#[repr(simd)]
struct i16x8([i16; 8]);
#[repr(simd)]
struct i32x4([i32; 4]);
#[repr(simd)]
struct i64x2([i64; 2]);
#[repr(simd)]
struct u8x16([u8; 16]);
#[repr(simd)]
struct u16x8([u16; 8]);
#[repr(simd)]
struct u32x4([u32; 4]);
#[repr(simd)]
struct u64x2([u64; 2]);
#[repr(simd)]
struct f32x4([f32; 4]);
#[repr(simd)]
struct f64x2([f64; 2]);

fn main() {

    let vi8x16 = i8x16([0, 1, 2, 3, 4, 5, 6, 7,
                      8, 9, 10, 11, 12, 13, 14, 15]);

    let vi16x8 = i16x8([16, 17, 18, 19, 20, 21, 22, 23]);
    let vi32x4 = i32x4([24, 25, 26, 27]);
    let vi64x2 = i64x2([28, 29]);

    let vu8x16 = u8x16([30, 31, 32, 33, 34, 35, 36, 37,
                      38, 39, 40, 41, 42, 43, 44, 45]);
    let vu16x8 = u16x8([46, 47, 48, 49, 50, 51, 52, 53]);
    let vu32x4 = u32x4([54, 55, 56, 57]);
    let vu64x2 = u64x2([58, 59]);

    let vf32x4 = f32x4([60.5f32, 61.5f32, 62.5f32, 63.5f32]);
    let vf64x2 = f64x2([64.5f64, 65.5f64]);

    zzz(); // #break
}

#[inline(never)]
fn zzz() { () }
