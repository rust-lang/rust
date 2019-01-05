// Need a fix for LLDB first...
// ignore-lldb
// ignore-tidy-linelength

// FIXME: LLVM generates invalid debug info for variables requiring
// dynamic stack realignment, which is the case on s390x for vector
// types with non-vector ABI.
// ignore-s390x

// compile-flags:-g
// gdb-command:run

// gdbg-command:print/d vi8x16
// gdbr-command:print vi8x16
// gdbg-check:$1 = {__0 = 0, __1 = 1, __2 = 2, __3 = 3, __4 = 4, __5 = 5, __6 = 6, __7 = 7, __8 = 8, __9 = 9, __10 = 10, __11 = 11, __12 = 12, __13 = 13, __14 = 14, __15 = 15}
// gdbr-check:$1 = simd::i8x16 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
// gdbg-command:print/d vi16x8
// gdbr-command:print vi16x8
// gdbg-check:$2 = {__0 = 16, __1 = 17, __2 = 18, __3 = 19, __4 = 20, __5 = 21, __6 = 22, __7 = 23}
// gdbr-check:$2 = simd::i16x8 (16, 17, 18, 19, 20, 21, 22, 23)
// gdbg-command:print/d vi32x4
// gdbr-command:print vi32x4
// gdbg-check:$3 = {__0 = 24, __1 = 25, __2 = 26, __3 = 27}
// gdbr-check:$3 = simd::i32x4 (24, 25, 26, 27)
// gdbg-command:print/d vi64x2
// gdbr-command:print vi64x2
// gdbg-check:$4 = {__0 = 28, __1 = 29}
// gdbr-check:$4 = simd::i64x2 (28, 29)

// gdbg-command:print/d vu8x16
// gdbr-command:print vu8x16
// gdbg-check:$5 = {__0 = 30, __1 = 31, __2 = 32, __3 = 33, __4 = 34, __5 = 35, __6 = 36, __7 = 37, __8 = 38, __9 = 39, __10 = 40, __11 = 41, __12 = 42, __13 = 43, __14 = 44, __15 = 45}
// gdbr-check:$5 = simd::u8x16 (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)
// gdbg-command:print/d vu16x8
// gdbr-command:print vu16x8
// gdbg-check:$6 = {__0 = 46, __1 = 47, __2 = 48, __3 = 49, __4 = 50, __5 = 51, __6 = 52, __7 = 53}
// gdbr-check:$6 = simd::u16x8 (46, 47, 48, 49, 50, 51, 52, 53)
// gdbg-command:print/d vu32x4
// gdbr-command:print vu32x4
// gdbg-check:$7 = {__0 = 54, __1 = 55, __2 = 56, __3 = 57}
// gdbr-check:$7 = simd::u32x4 (54, 55, 56, 57)
// gdbg-command:print/d vu64x2
// gdbr-command:print vu64x2
// gdbg-check:$8 = {__0 = 58, __1 = 59}
// gdbr-check:$8 = simd::u64x2 (58, 59)

// gdb-command:print vf32x4
// gdbg-check:$9 = {__0 = 60.5, __1 = 61.5, __2 = 62.5, __3 = 63.5}
// gdbr-check:$9 = simd::f32x4 (60.5, 61.5, 62.5, 63.5)
// gdb-command:print vf64x2
// gdbg-check:$10 = {__0 = 64.5, __1 = 65.5}
// gdbr-check:$10 = simd::f64x2 (64.5, 65.5)

// gdb-command:continue

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]
#![feature(repr_simd)]

#[repr(simd)]
struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
struct i64x2(i64, i64);
#[repr(simd)]
struct u8x16(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);
#[repr(simd)]
struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);
#[repr(simd)]
struct u32x4(u32, u32, u32, u32);
#[repr(simd)]
struct u64x2(u64, u64);
#[repr(simd)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
struct f64x2(f64, f64);

fn main() {

    let vi8x16 = i8x16(0, 1, 2, 3, 4, 5, 6, 7,
                      8, 9, 10, 11, 12, 13, 14, 15);

    let vi16x8 = i16x8(16, 17, 18, 19, 20, 21, 22, 23);
    let vi32x4 = i32x4(24, 25, 26, 27);
    let vi64x2 = i64x2(28, 29);

    let vu8x16 = u8x16(30, 31, 32, 33, 34, 35, 36, 37,
                      38, 39, 40, 41, 42, 43, 44, 45);
    let vu16x8 = u16x8(46, 47, 48, 49, 50, 51, 52, 53);
    let vu32x4 = u32x4(54, 55, 56, 57);
    let vu64x2 = u64x2(58, 59);

    let vf32x4 = f32x4(60.5f32, 61.5f32, 62.5f32, 63.5f32);
    let vf64x2 = f64x2(64.5f64, 65.5f64);

    zzz(); // #break
}

#[inline(never)]
fn zzz() { () }
