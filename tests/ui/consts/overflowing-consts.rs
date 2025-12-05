// Tests that overflowing or bound-exceeding operations
// for compile-time consts raise errors

//@ revisions: noopt opt opt_with_overflow_checks
//@ [noopt]compile-flags: -C opt-level=0
//@ [opt]compile-flags: -O
//@ [opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O
//@ ignore-pass (test tests codegen-time behaviour)
//@ normalize-stderr: "shift left by `(64|32)_usize`, which" -> "shift left by `%BITS%`, which"
//@ normalize-stderr: "shift right by `(64|32)_usize`, which" -> "shift right by `%BITS%`, which"

#[cfg(target_pointer_width = "32")]
const BITS: usize = 32;
#[cfg(target_pointer_width = "64")]
const BITS: usize = 64;

// Shift left
const _NI8_SHL: i8 = 1i8 << 8; //~ ERROR: overflow
const _NI8_SHL_P: &i8 = &(1i8 << 8); //~ ERROR: overflow

const _NI16_SHL: i16 = 1i16 << 16; //~ ERROR: overflow
const _NI16_SHL_P: &i16 = &(1i16 << 16); //~ ERROR: overflow

const _NI32_SHL: i32 = 1i32 << 32; //~ ERROR: overflow
const _NI32_SHL_P: &i32 = &(1i32 << 32); //~ ERROR: overflow

const _NI64_SHL: i64 = 1i64 << 64; //~ ERROR: overflow
const _NI64_SHL_P: &i64 = &(1i64 << 64); //~ ERROR: overflow

const _NI128_SHL: i128 = 1i128 << 128; //~ ERROR: overflow
const _NI128_SHL_P: &i128 = &(1i128 << 128); //~ ERROR: overflow

const _NU8_SHL: u8 = 1u8 << 8; //~ ERROR: overflow
const _NU8_SHL_P: &u8 = &(1u8 << 8); //~ ERROR: overflow

const _NU16_SHL: u16 = 1u16 << 16; //~ ERROR: overflow
const _NU16_SHL_P: &u16 = &(1u16 << 16); //~ ERROR: overflow

const _NU32_SHL: u32 = 1u32 << 32; //~ ERROR: overflow
const _NU32_SHL_P: &u32 = &(1u32 << 32); //~ ERROR: overflow

const _NU64_SHL: u64 = 1u64 << 64; //~ ERROR: overflow
const _NU64_SHL_P: &u64 = &(1u64 << 64); //~ ERROR: overflow

const _NU128_SHL: u128 = 1u128 << 128; //~ ERROR: overflow
const _NU128_SHL_P: &u128 = &(1u128 << 128); //~ ERROR: overflow

const _NISIZE_SHL: isize = 1isize << BITS; //~ ERROR: overflow
const _NISIZE_SHL_P: &isize = &(1isize << BITS); //~ ERROR: overflow

const _NUSIZE_SHL: usize = 1usize << BITS; //~ ERROR: overflow
const _NUSIZE_SHL_P: &usize = &(1usize << BITS); //~ ERROR: overflow

// Shift right
const _NI8_SHR: i8 = 1i8 >> 8; //~ ERROR: overflow
const _NI8_SHR_P: &i8 = &(1i8 >> 8); //~ ERROR: overflow

const _NI16_SHR: i16 = 1i16 >> 16; //~ ERROR: overflow
const _NI16_SHR_P: &i16 = &(1i16 >> 16); //~ ERROR: overflow

const _NI32_SHR: i32 = 1i32 >> 32; //~ ERROR: overflow
const _NI32_SHR_P: &i32 = &(1i32 >> 32); //~ ERROR: overflow

const _NI64_SHR: i64 = 1i64 >> 64; //~ ERROR: overflow
const _NI64_SHR_P: &i64 = &(1i64 >> 64); //~ ERROR: overflow

const _NI128_SHR: i128 = 1i128 >> 128; //~ ERROR: overflow
const _NI128_SHR_P: &i128 = &(1i128 >> 128); //~ ERROR: overflow

const _NU8_SHR: u8 = 1u8 >> 8; //~ ERROR: overflow
const _NU8_SHR_P: &u8 = &(1u8 >> 8); //~ ERROR: overflow

const _NU16_SHR: u16 = 1u16 >> 16; //~ ERROR: overflow
const _NU16_SHR_P: &u16 = &(1u16 >> 16); //~ ERROR: overflow

const _NU32_SHR: u32 = 1u32 >> 32; //~ ERROR: overflow
const _NU32_SHR_P: &u32 = &(1u32 >> 32); //~ ERROR: overflow

const _NU64_SHR: u64 = 1u64 >> 64; //~ ERROR: overflow
const _NU64_SHR_P: &u64 = &(1u64 >> 64); //~ ERROR: overflow

const _NU128_SHR: u128 = 1u128 >> 128; //~ ERROR: overflow
const _NU128_SHR_P: &u128 = &(1u128 >> 128); //~ ERROR: overflow

const _NISIZE_SHR: isize = 1isize >> BITS; //~ ERROR: overflow
const _NISIZE_SHR_P: &isize = &(1isize >> BITS); //~ ERROR: overflow

const _NUSIZE_SHR: usize = 1usize >> BITS; //~ ERROR: overflow
const _NUSIZE_SHR_P: &usize = &(1usize >> BITS); //~ ERROR: overflow

// Addition
const _NI8_ADD: i8 = 1i8 + i8::MAX; //~ ERROR: overflow
const _NI8_ADD_P: &i8 = &(1i8 + i8::MAX); //~ ERROR: overflow

const _NI16_ADD: i16 = 1i16 + i16::MAX; //~ ERROR: overflow
const _NI16_ADD_P: &i16 = &(1i16 + i16::MAX); //~ ERROR: overflow

const _NI32_ADD: i32 = 1i32 + i32::MAX; //~ ERROR: overflow
const _NI32_ADD_P: &i32 = &(1i32 + i32::MAX); //~ ERROR: overflow

const _NI64_ADD: i64 = 1i64 + i64::MAX; //~ ERROR: overflow
const _NI64_ADD_P: &i64 = &(1i64 + i64::MAX); //~ ERROR: overflow

const _NI128_ADD: i128 = 1i128 + i128::MAX; //~ ERROR: overflow
const _NI128_ADD_P: &i128 = &(1i128 + i128::MAX); //~ ERROR: overflow

const _NU8_ADD: u8 = 1u8 + u8::MAX; //~ ERROR: overflow
const _NU8_ADD_P: &u8 = &(1u8 + u8::MAX); //~ ERROR: overflow

const _NU16_ADD: u16 = 1u16 + u16::MAX; //~ ERROR: overflow
const _NU16_ADD_P: &u16 = &(1u16 + u16::MAX); //~ ERROR: overflow

const _NU32_ADD: u32 = 1u32 + u32::MAX; //~ ERROR: overflow
const _NU32_ADD_P: &u32 = &(1u32 + u32::MAX); //~ ERROR: overflow

const _NU64_ADD: u64 = 1u64 + u64::MAX; //~ ERROR: overflow
const _NU64_ADD_P: &u64 = &(1u64 + u64::MAX); //~ ERROR: overflow

const _NU128_ADD: u128 = 1u128 + u128::MAX; //~ ERROR: overflow
const _NU128_ADD_P: &u128 = &(1u128 + u128::MAX); //~ ERROR: overflow

const _NISIZE_ADD: isize = 1isize + isize::MAX; //~ ERROR: overflow
const _NISIZE_ADD_P: &isize = &(1isize + isize::MAX); //~ ERROR: overflow

const _NUSIZE_ADD: usize = 1usize + usize::MAX; //~ ERROR: overflow
const _NUSIZE_ADD_P: &usize = &(1usize + usize::MAX); //~ ERROR: overflow

// Subtraction
const _NI8_SUB: i8 = -5i8 - i8::MAX; //~ ERROR: overflow
const _NI8_SUB_P: &i8 = &(-5i8 - i8::MAX); //~ ERROR: overflow

const _NI16_SUB: i16 = -5i16 - i16::MAX; //~ ERROR: overflow
const _NI16_SUB_P: &i16 = &(-5i16 - i16::MAX); //~ ERROR: overflow

const _NI32_SUB: i32 = -5i32 - i32::MAX; //~ ERROR: overflow
const _NI32_SUB_P: &i32 = &(-5i32 - i32::MAX); //~ ERROR: overflow

const _NI64_SUB: i64 = -5i64 - i64::MAX; //~ ERROR: overflow
const _NI64_SUB_P: &i64 = &(-5i64 - i64::MAX); //~ ERROR: overflow

const _NI128_SUB: i128 = -5i128 - i128::MAX; //~ ERROR: overflow
const _NI128_SUB_P: &i128 = &(-5i128 - i128::MAX); //~ ERROR: overflow

const _NU8_SUB: u8 = 1u8 - 5; //~ ERROR: overflow
const _NU8_SUB_P: &u8 = &(1u8 - 5); //~ ERROR: overflow

const _NU16_SUB: u16 = 1u16 - 5; //~ ERROR: overflow
const _NU16_SUB_P: &u16 = &(1u16 - 5); //~ ERROR: overflow

const _NU32_SUB: u32 = 1u32 - 5; //~ ERROR: overflow
const _NU32_SUB_P: &u32 = &(1u32 - 5); //~ ERROR: overflow

const _NU64_SUB: u64 = 1u64 - 5; //~ ERROR: overflow
const _NU64_SUB_P: &u64 = &(1u64 - 5); //~ ERROR: overflow

const _NU128_SUB: u128 = 1u128 - 5; //~ ERROR: overflow
const _NU128_SUB_P: &u128 = &(1u128 - 5); //~ ERROR: overflow

const _NISIZE_SUB: isize = -5isize - isize::MAX; //~ ERROR: overflow
const _NISIZE_SUB_P: &isize = &(-5isize - isize::MAX); //~ ERROR: overflow

const _NUSIZE_SUB: usize = 1usize - 5; //~ ERROR: overflow
const _NUSIZE_SUB_P: &usize = &(1usize - 5); //~ ERROR: overflow

// Multiplication
const _NI8_MUL: i8 = i8::MAX * 5; //~ ERROR: overflow
const _NI8_MUL_P: &i8 = &(i8::MAX * 5); //~ ERROR: overflow

const _NI16_MUL: i16 = i16::MAX * 5; //~ ERROR: overflow
const _NI16_MUL_P: &i16 = &(i16::MAX * 5); //~ ERROR: overflow

const _NI32_MUL: i32 = i32::MAX * 5; //~ ERROR: overflow
const _NI32_MUL_P: &i32 = &(i32::MAX * 5); //~ ERROR: overflow

const _NI64_MUL: i64 = i64::MAX * 5; //~ ERROR: overflow
const _NI64_MUL_P: &i64 = &(i64::MAX * 5); //~ ERROR: overflow

const _NI128_MUL: i128 = i128::MAX * 5; //~ ERROR: overflow
const _NI128_MUL_P: &i128 = &(i128::MAX * 5); //~ ERROR: overflow

const _NU8_MUL: u8 = u8::MAX * 5; //~ ERROR: overflow
const _NU8_MUL_P: &u8 = &(u8::MAX * 5); //~ ERROR: overflow

const _NU16_MUL: u16 = u16::MAX * 5; //~ ERROR: overflow
const _NU16_MUL_P: &u16 = &(u16::MAX * 5); //~ ERROR: overflow

const _NU32_MUL: u32 = u32::MAX * 5; //~ ERROR: overflow
const _NU32_MUL_P: &u32 = &(u32::MAX * 5); //~ ERROR: overflow

const _NU64_MUL: u64 = u64::MAX * 5; //~ ERROR: overflow
const _NU64_MUL_P: &u64 = &(u64::MAX * 5); //~ ERROR: overflow

const _NU128_MUL: u128 = u128::MAX * 5; //~ ERROR: overflow
const _NU128_MUL_P: &u128 = &(u128::MAX * 5); //~ ERROR: overflow

const _NISIZE_MUL: isize = isize::MAX * 5; //~ ERROR: overflow
const _NISIZE_MUL_P: &isize = &(isize::MAX * 5); //~ ERROR: overflow

const _NUSIZE_MUL: usize = usize::MAX * 5; //~ ERROR: overflow
const _NUSIZE_MUL_P: &usize = &(usize::MAX * 5); //~ ERROR: overflow

// Division
const _NI8_DIV: i8 = 1i8 / 0; //~ ERROR: by zero
const _NI8_DIV_P: &i8 = &(1i8 / 0); //~ ERROR: by zero

const _NI16_DIV: i16 = 1i16 / 0; //~ ERROR: by zero
const _NI16_DIV_P: &i16 = &(1i16 / 0); //~ ERROR: by zero

const _NI32_DIV: i32 = 1i32 / 0; //~ ERROR: by zero
const _NI32_DIV_P: &i32 = &(1i32 / 0); //~ ERROR: by zero

const _NI64_DIV: i64 = 1i64 / 0; //~ ERROR: by zero
const _NI64_DIV_P: &i64 = &(1i64 / 0); //~ ERROR: by zero

const _NI128_DIV: i128 = 1i128 / 0; //~ ERROR: by zero
const _NI128_DIV_P: &i128 = &(1i128 / 0); //~ ERROR: by zero

const _NU8_DIV: u8 = 1u8 / 0; //~ ERROR: by zero
const _NU8_DIV_P: &u8 = &(1u8 / 0); //~ ERROR: by zero

const _NU16_DIV: u16 = 1u16 / 0; //~ ERROR: by zero
const _NU16_DIV_P: &u16 = &(1u16 / 0); //~ ERROR: by zero

const _NU32_DIV: u32 = 1u32 / 0; //~ ERROR: by zero
const _NU32_DIV_P: &u32 = &(1u32 / 0); //~ ERROR: by zero

const _NU64_DIV: u64 = 1u64 / 0; //~ ERROR: by zero
const _NU64_DIV_P: &u64 = &(1u64 / 0); //~ ERROR: by zero

const _NU128_DIV: u128 = 1u128 / 0; //~ ERROR: by zero
const _NU128_DIV_P: &u128 = &(1u128 / 0); //~ ERROR: by zero

const _NISIZE_DIV: isize = 1isize / 0; //~ ERROR: by zero
const _NISIZE_DIV_P: &isize = &(1isize / 0); //~ ERROR: by zero

const _NUSIZE_DIV: usize = 1usize / 0; //~ ERROR: by zero
const _NUSIZE_DIV_P: &usize = &(1usize / 0); //~ ERROR: by zero

// Modulus
const _NI8_MOD: i8 = 1i8 % 0; //~ ERROR: divisor of zero
const _NI8_MOD_P: &i8 = &(1i8 % 0); //~ ERROR: divisor of zero

const _NI16_MOD: i16 = 1i16 % 0; //~ ERROR: divisor of zero
const _NI16_MOD_P: &i16 = &(1i16 % 0); //~ ERROR: divisor of zero

const _NI32_MOD: i32 = 1i32 % 0; //~ ERROR: divisor of zero
const _NI32_MOD_P: &i32 = &(1i32 % 0); //~ ERROR: divisor of zero

const _NI64_MOD: i64 = 1i64 % 0; //~ ERROR: divisor of zero
const _NI64_MOD_P: &i64 = &(1i64 % 0); //~ ERROR: divisor of zero

const _NI128_MOD: i128 = 1i128 % 0; //~ ERROR: divisor of zero
const _NI128_MOD_P: &i128 = &(1i128 % 0); //~ ERROR: divisor of zero

const _NU8_MOD: u8 = 1u8 % 0; //~ ERROR: divisor of zero
const _NU8_MOD_P: &u8 = &(1u8 % 0); //~ ERROR: divisor of zero

const _NU16_MOD: u16 = 1u16 % 0; //~ ERROR: divisor of zero
const _NU16_MOD_P: &u16 = &(1u16 % 0); //~ ERROR: divisor of zero

const _NU32_MOD: u32 = 1u32 % 0; //~ ERROR: divisor of zero
const _NU32_MOD_P: &u32 = &(1u32 % 0); //~ ERROR: divisor of zero

const _NU64_MOD: u64 = 1u64 % 0; //~ ERROR: divisor of zero
const _NU64_MOD_P: &u64 = &(1u64 % 0); //~ ERROR: divisor of zero

const _NU128_MOD: u128 = 1u128 % 0; //~ ERROR: divisor of zero
const _NU128_MOD_P: &u128 = &(1u128 % 0); //~ ERROR: divisor of zero

const _NISIZE_MOD: isize = 1isize % 0; //~ ERROR: divisor of zero
const _NISIZE_MOD_P: &isize = &(1isize % 0); //~ ERROR: divisor of zero

const _NUSIZE_MOD: usize = 1usize % 0; //~ ERROR: divisor of zero
const _NUSIZE_MOD_P: &usize = &(1usize % 0); //~ ERROR: divisor of zero

// Out of bounds access
const _NI32_OOB: i32 = [1, 2, 3][4]; //~ ERROR: the length is 3 but the index is 4
const _NI32_OOB_P: &i32 = &([1, 2, 3][4]); //~ ERROR: the length is 3 but the index is 4

pub fn main() {}
