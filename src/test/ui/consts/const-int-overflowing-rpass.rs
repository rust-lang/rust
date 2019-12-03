// run-pass
#![feature(const_int_overflowing)]

const ADD_A: (u32, bool) = 5u32.overflowing_add(2);
const ADD_B: (u32, bool) = u32::max_value().overflowing_add(1);

const SUB_A: (u32, bool) = 5u32.overflowing_sub(2);
const SUB_B: (u32, bool) = 0u32.overflowing_sub(1);

const MUL_A: (u32, bool) = 5u32.overflowing_mul(2);
const MUL_B: (u32, bool) = 1_000_000_000u32.overflowing_mul(10);

const SHL_A: (u32, bool) = 0x1u32.overflowing_shl(4);
const SHL_B: (u32, bool) = 0x1u32.overflowing_shl(132);

const SHR_A: (u32, bool) = 0x10u32.overflowing_shr(4);
const SHR_B: (u32, bool) = 0x10u32.overflowing_shr(132);

const NEG_A: (u32, bool) = 0u32.overflowing_neg();
const NEG_B: (u32, bool) = core::u32::MAX.overflowing_neg();

const ABS_POS: (i32, bool) = 10i32.overflowing_abs();
const ABS_NEG: (i32, bool) = (-10i32).overflowing_abs();
const ABS_MIN: (i32, bool) = i32::min_value().overflowing_abs();

const DIV_A: (i8, bool) = 8i8.overflowing_div(2);
const DIV_B: (i8, bool) = 8i8.overflowing_div(3);
const DIV_C: (i8, bool) = i8::min_value().overflowing_div(-1i8);
const DIV_D: (u8, bool) = 8u8.overflowing_div(2);
const DIV_E: (u8, bool) = 8u8.overflowing_div(3);

const REM_A: (i8, bool) = 8i8.overflowing_rem(2);
const REM_B: (i8, bool) = 8i8.overflowing_rem(3);
const REM_C: (i8, bool) = i8::min_value().overflowing_rem(-1i8);
const REM_D: (u8, bool) = 8u8.overflowing_rem(2);
const REM_E: (u8, bool) = 8u8.overflowing_rem(3);

fn main() {
    assert_eq!(ADD_A, (7, false));
    assert_eq!(ADD_B, (0, true));

    assert_eq!(SUB_A, (3, false));
    assert_eq!(SUB_B, (u32::max_value(), true));

    assert_eq!(MUL_A, (10, false));
    assert_eq!(MUL_B, (1410065408, true));

    assert_eq!(SHL_A, (0x10, false));
    assert_eq!(SHL_B, (0x10, true));

    assert_eq!(SHR_A, (0x1, false));
    assert_eq!(SHR_B, (0x1, true));

    assert_eq!(NEG_A, (0, false));
    assert_eq!(NEG_B, (1, true));

    assert_eq!(ABS_POS, (10, false));
    assert_eq!(ABS_NEG, (10, false));
    assert_eq!(ABS_MIN, (i32::min_value(), true));

    assert_eq!(DIV_A, (4i8, false));
    assert_eq!(DIV_B, (2i8, false));
    assert_eq!(DIV_C, (i8::min_value(), true));
    assert_eq!(DIV_D, (4u8, false));
    assert_eq!(DIV_E, (2u8, false));

    assert_eq!(REM_A, (0i8, false));
    assert_eq!(REM_B, (2i8, false));
    assert_eq!(REM_C, (0i8, true));
    assert_eq!(REM_D, (0u8, false));
    assert_eq!(REM_E, (2u8, false));
}
