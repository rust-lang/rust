// run-pass
#![feature(const_int_wrapping)]

const ADD_A: u32 = 200u32.wrapping_add(55);
const ADD_B: u32 = 200u32.wrapping_add(u32::max_value());

const SUB_A: u32 = 100u32.wrapping_sub(100);
const SUB_B: u32 = 100u32.wrapping_sub(u32::max_value());

const MUL_A: u8 = 10u8.wrapping_mul(12);
const MUL_B: u8 = 25u8.wrapping_mul(12);

const SHL_A: u32 = 1u32.wrapping_shl(7);
const SHL_B: u32 = 1u32.wrapping_shl(128);

const SHR_A: u32 = 128u32.wrapping_shr(7);
const SHR_B: u32 = 128u32.wrapping_shr(128);

const NEG_A: u32 = 5u32.wrapping_neg();
const NEG_B: u32 = 1234567890u32.wrapping_neg();

const ABS_POS: i32 = 10i32.wrapping_abs();
const ABS_NEG: i32 = (-10i32).wrapping_abs();
const ABS_MIN: i32 = i32::min_value().wrapping_abs();

const DIV_A: i8 = 8i8.wrapping_div(2);
const DIV_B: i8 = 8i8.wrapping_div(3);
const DIV_C: i8 = i8::min_value().wrapping_div(-1i8);
const DIV_D: u8 = 8u8.wrapping_div(2);
const DIV_E: u8 = 8u8.wrapping_div(3);

const REM_A: i8 = 8i8.wrapping_rem(2);
const REM_B: i8 = 8i8.wrapping_rem(3);
const REM_C: i8 = i8::min_value().wrapping_rem(-1i8);
const REM_D: u8 = 8u8.wrapping_rem(2);
const REM_E: u8 = 8u8.wrapping_rem(3);

fn main() {
    assert_eq!(ADD_A, 255);
    assert_eq!(ADD_B, 199);

    assert_eq!(SUB_A, 0);
    assert_eq!(SUB_B, 101);

    assert_eq!(MUL_A, 120);
    assert_eq!(MUL_B, 44);

    assert_eq!(SHL_A, 128);
    assert_eq!(SHL_B, 1);

    assert_eq!(SHR_A, 1);
    assert_eq!(SHR_B, 128);

    assert_eq!(NEG_A, 4294967291);
    assert_eq!(NEG_B, 3060399406);

    assert_eq!(ABS_POS, 10);
    assert_eq!(ABS_NEG, 10);
    assert_eq!(ABS_MIN, i32::min_value());

    assert_eq!(DIV_A, 4i8);
    assert_eq!(DIV_B, 2i8);
    assert_eq!(DIV_C, i8::min_value());
    assert_eq!(DIV_D, 4u8);
    assert_eq!(DIV_E, 2u8);

    assert_eq!(REM_A, 0i8);
    assert_eq!(REM_B, 2i8);
    assert_eq!(REM_C, 0i8);
    assert_eq!(REM_D, 0u8);
    assert_eq!(REM_E, 2u8);
}
