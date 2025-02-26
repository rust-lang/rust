// Tests that overflowing or bound-exceeding operations
// are correctly linted including when they are const promoted

// We are using "-Z deduplicate-diagnostics=yes" because different
// build configurations emit different number of duplicate diagnostics
// and this flag lets us test them all with a single .rs file like this

//@ revisions: noopt opt opt_with_overflow_checks
//@ [noopt]compile-flags: -C opt-level=0 -Z deduplicate-diagnostics=yes
//@ [opt]compile-flags: -O
//@ [opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O -Z deduplicate-diagnostics=yes
//@ build-fail
//@ ignore-pass (test tests codegen-time behaviour)
//@ normalize-stderr: "shift left by `(64|32)_usize`, which" -> "shift left by `%BITS%`, which"
//@ normalize-stderr: "shift right by `(64|32)_usize`, which" -> "shift right by `%BITS%`, which"

#![deny(arithmetic_overflow)]

#[cfg(target_pointer_width = "32")]
const BITS: usize = 32;
#[cfg(target_pointer_width = "64")]
const BITS: usize = 64;

use std::thread;

fn main() {
    // Shift left
    let _n = 1u8 << 8;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1u8 << 8);   //~ ERROR: arithmetic operation will overflow

    let _n = 1u16 << 16; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u16 << 16); //~ ERROR: arithmetic operation will overflow

    let _n = 1u32 << 32; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u32 << 32); //~ ERROR: arithmetic operation will overflow

    let _n = 1u64 << 64; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u64 << 64); //~ ERROR: arithmetic operation will overflow

    let _n = 1u128 << 128; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u128 << 128); //~ ERROR: arithmetic operation will overflow

    let _n = 1i8 << 8;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1i8 << 8);   //~ ERROR: arithmetic operation will overflow

    let _n = 1i16 << 16; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i16 << 16); //~ ERROR: arithmetic operation will overflow

    let _n = 1i32 << 32; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i32 << 32); //~ ERROR: arithmetic operation will overflow

    let _n = 1i64 << 64; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i64 << 64); //~ ERROR: arithmetic operation will overflow

    let _n = 1i128 << 128; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i128 << 128); //~ ERROR: arithmetic operation will overflow

    let _n = 1_isize << BITS; //~ ERROR: arithmetic operation will overflow
    let _n = &(1_isize << BITS); //~ ERROR: arithmetic operation will overflow

    let _n = 1_usize << BITS; //~ ERROR: arithmetic operation will overflow
    let _n = &(1_usize << BITS); //~ ERROR: arithmetic operation will overflow

    let _n = 1 << -1; //~ ERROR: arithmetic operation will overflow
    let _n = &(1 << -1); //~ ERROR: arithmetic operation will overflow

    // Shift right

    let _n = -1_i64 >> 64;  //~ ERROR: arithmetic operation will overflow
    let _n = -1_i32 >> 32;  //~ ERROR: arithmetic operation will overflow
    let _n = -1_i32 >> -1;  //~ ERROR: arithmetic operation will overflow

    let _n = 1u8 >> 8;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1u8 >> 8);   //~ ERROR: arithmetic operation will overflow

    let _n = 1u16 >> 16; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u16 >> 16); //~ ERROR: arithmetic operation will overflow

    let _n = 1u32 >> 32; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u32 >> 32); //~ ERROR: arithmetic operation will overflow

    let _n = 1u64 >> 64; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u64 >> 64); //~ ERROR: arithmetic operation will overflow

    let _n = 1u128 >> 128; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u128 >> 128); //~ ERROR: arithmetic operation will overflow

    let _n = 1i8 >> 8;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1i8 >> 8);   //~ ERROR: arithmetic operation will overflow

    let _n = 1i16 >> 16; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i16 >> 16); //~ ERROR: arithmetic operation will overflow

    let _n = 1i32 >> 32; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i32 >> 32); //~ ERROR: arithmetic operation will overflow

    let _n = 1i64 >> 64; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i64 >> 64); //~ ERROR: arithmetic operation will overflow

    let _n = 1i128 >> 128; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i128 >> 128); //~ ERROR: arithmetic operation will overflow

    let _n = 1_isize >> BITS; //~ ERROR: arithmetic operation will overflow
    let _n = &(1_isize >> BITS); //~ ERROR: arithmetic operation will overflow

    let _n = 1_usize >> BITS; //~ ERROR: arithmetic operation will overflow
    let _n = &(1_usize >> BITS); //~ ERROR: arithmetic operation will overflow

    let _n = 1i64 >> [64][0];  //~ ERROR: arithmetic operation will overflow
    let _n = &(1i64 >> [64][0]);  //~ ERROR: arithmetic operation will overflow

    // Addition
    let _n = 1u8 + u8::MAX;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1u8 + u8::MAX);   //~ ERROR: arithmetic operation will overflow

    let _n = 1u16 + u16::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u16 + u16::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1u32 + u32::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u32 + u32::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1u64 + u64::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u64 + u64::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1u128 + u128::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u128 + u128::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1i8 + i8::MAX;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1i8 + i8::MAX);   //~ ERROR: arithmetic operation will overflow

    let _n = 1i16 + i16::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i16 + i16::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1i32 + i32::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i32 + i32::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1i64 + i64::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i64 + i64::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1i128 + i128::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1i128 + i128::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1isize + isize::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1isize + isize::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1usize + usize::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(1usize + usize::MAX); //~ ERROR: arithmetic operation will overflow


    // Subtraction
    let _n = 1u8 - 5;   //~ ERROR: arithmetic operation will overflow
    let _n = &(1u8 - 5);   //~ ERROR: arithmetic operation will overflow

    let _n = 1u16 - 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u16 - 5); //~ ERROR: arithmetic operation will overflow

    let _n = 1u32 - 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u32 - 5); //~ ERROR: arithmetic operation will overflow

    let _n = 1u64 - 5 ; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u64 - 5); //~ ERROR: arithmetic operation will overflow

    let _n = 1u128 - 5 ; //~ ERROR: arithmetic operation will overflow
    let _n = &(1u128 - 5); //~ ERROR: arithmetic operation will overflow

    let _n = -5i8 - i8::MAX;   //~ ERROR: arithmetic operation will overflow
    let _n = &(-5i8 - i8::MAX);   //~ ERROR: arithmetic operation will overflow

    let _n = -5i16 - i16::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(-5i16 - i16::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = -5i32 - i32::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(-5i32 - i32::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = -5i64 - i64::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(-5i64 - i64::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = -5i128 - i128::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(-5i128 - i128::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = -5isize - isize::MAX; //~ ERROR: arithmetic operation will overflow
    let _n = &(-5isize - isize::MAX); //~ ERROR: arithmetic operation will overflow

    let _n = 1usize - 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(1usize - 5); //~ ERROR: arithmetic operation will overflow

    let _n = -i8::MIN; //~ ERROR this arithmetic operation will overflow
    let _n = &(-i8::MIN); //~ ERROR this arithmetic operation will overflow


    // Multiplication
    let _n = u8::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(u8::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = u16::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(u16::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = u32::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(u32::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = u64::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(u64::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = u128::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(u128::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = usize::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(usize::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = i8::MAX * i8::MAX;   //~ ERROR: arithmetic operation will overflow
    let _n = &(i8::MAX * i8::MAX);   //~ ERROR: arithmetic operation will overflow

    let _n = i16::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(i16::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = i32::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(i32::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = i64::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(i64::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = i128::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(i128::MAX * 5); //~ ERROR: arithmetic operation will overflow

    let _n = isize::MAX * 5; //~ ERROR: arithmetic operation will overflow
    let _n = &(isize::MAX * 5); //~ ERROR: arithmetic operation will overflow


    // Division
    let _n = 1u8 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u8 / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u16 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u16 / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u32 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u32 / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u64 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u64 / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u128 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u128 / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1usize / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1usize / 0); //~ ERROR: this operation will panic at runtime

    let _n = 1i8 / 0;   //~ ERROR: this operation will panic at runtime
    let _n = &(1i8 / 0);   //~ ERROR: this operation will panic at runtime
    let _n = i8::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i8::MIN / -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i16 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i16 / 0); //~ ERROR: this operation will panic at runtime
    let _n = i16::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i16::MIN / -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i32 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i32 / 0); //~ ERROR: this operation will panic at runtime
    let _n = i32::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i32::MIN / -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i64 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i64 / 0); //~ ERROR: this operation will panic at runtime
    let _n = i64::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i64::MIN / -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i128 / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i128 / 0); //~ ERROR: this operation will panic at runtime
    let _n = i128::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i128::MIN / -1); //~ ERROR: this operation will panic at runtime

    let _n = 1isize / 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1isize / 0); //~ ERROR: this operation will panic at runtime
    let _n = isize::MIN / -1; //~ ERROR: this operation will panic at runtime
    let _n = &(isize::MIN / -1); //~ ERROR: this operation will panic at runtime


    // Modulus
    let _n = 1u8 % 0;   //~ ERROR: this operation will panic at runtime
    let _n = &(1u8 % 0);   //~ ERROR: this operation will panic at runtime

    let _n = 1u16 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u16 % 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u32 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u32 % 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u64 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u64 % 0); //~ ERROR: this operation will panic at runtime

    let _n = 1u128 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1u128 % 0); //~ ERROR: this operation will panic at runtime

    let _n = 1usize % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1usize % 0); //~ ERROR: this operation will panic at runtime

    let _n = 1i8 % 0;   //~ ERROR: this operation will panic at runtime
    let _n = &(1i8 % 0);   //~ ERROR: this operation will panic at runtime
    let _n = i8::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i8::MIN % -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i16 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i16 % 0); //~ ERROR: this operation will panic at runtime
    let _n = i16::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i16::MIN % -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i32 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i32 % 0); //~ ERROR: this operation will panic at runtime
    let _n = i32::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i32::MIN % -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i64 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i64 % 0); //~ ERROR: this operation will panic at runtime
    let _n = i64::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i64::MIN % -1); //~ ERROR: this operation will panic at runtime

    let _n = 1i128 % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1i128 % 0); //~ ERROR: this operation will panic at runtime
    let _n = i128::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(i128::MIN % -1); //~ ERROR: this operation will panic at runtime

    let _n = 1isize % 0; //~ ERROR: this operation will panic at runtime
    let _n = &(1isize % 0); //~ ERROR: this operation will panic at runtime
    let _n = isize::MIN % -1; //~ ERROR: this operation will panic at runtime
    let _n = &(isize::MIN % -1); //~ ERROR: this operation will panic at runtime

    // Out of bounds access
    let _n = [1, 2, 3][4]; //~ ERROR: this operation will panic at runtime
    let _n = &([1, 2, 3][4]); //~ ERROR: this operation will panic at runtime
}
