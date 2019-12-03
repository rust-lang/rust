// run-pass
#![feature(const_int_checked)]

macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $(assert_eq!($exp, $ident);)+
        }
    }
}

assert_same_const! {
    const CHECKED_ADD_I32_A: Option<i32> = 5i32.checked_add(2);
    const CHECKED_ADD_I8_A: Option<i8> = 127i8.checked_add(2);
    const CHECKED_ADD_U8_A: Option<u8> = 255u8.checked_add(2);

    const CHECKED_SUB_I32_A: Option<i32> = 5i32.checked_sub(2);
    const CHECKED_SUB_I8_A: Option<i8> = (-127 as i8).checked_sub(2);
    const CHECKED_SUB_U8_A: Option<u8> = 1u8.checked_sub(2);

    const CHECKED_MUL_I32_A: Option<i32> = 5i32.checked_mul(7777);
    const CHECKED_MUL_I8_A: Option<i8> = (-127 as i8).checked_mul(-99);
    const CHECKED_MUL_U8_A: Option<u8> = 1u8.checked_mul(3);

    // Needs intrinsics::unchecked_div.
    // const CHECKED_DIV_I32_A: Option<i32> = 5i32.checked_div(7777);
    // const CHECKED_DIV_I8_A: Option<i8> = (-127 as i8).checked_div(-99);
    // const CHECKED_DIV_U8_A: Option<u8> = 1u8.checked_div(3);

    // Needs intrinsics::unchecked_rem.
    // const CHECKED_REM_I32_A: Option<i32> = 5i32.checked_rem(7777);
    // const CHECKED_REM_I8_A: Option<i8> = (-127 as i8).checked_rem(-99);
    // const CHECKED_REM_U8_A: Option<u8> = 1u8.checked_rem(3);
    // const CHECKED_REM_U8_B: Option<u8> = 1u8.checked_rem(0);

    const CHECKED_NEG_I32_A: Option<i32> = 5i32.checked_neg();
    const CHECKED_NEG_I8_A: Option<i8> = (-127 as i8).checked_neg();
    const CHECKED_NEG_U8_A: Option<u8> = 1u8.checked_neg();
    const CHECKED_NEG_U8_B: Option<u8> = u8::min_value().checked_neg();

    const CHECKED_SHL_I32_A: Option<i32> = 5i32.checked_shl(77777);
    const CHECKED_SHL_I8_A: Option<i8> = (-127 as i8).checked_shl(2);
    const CHECKED_SHL_U8_A: Option<u8> = 1u8.checked_shl(8);
    const CHECKED_SHL_U8_B: Option<u8> = 1u8.checked_shl(0);

    const CHECKED_SHR_I32_A: Option<i32> = 5i32.checked_shr(77777);
    const CHECKED_SHR_I8_A: Option<i8> = (-127 as i8).checked_shr(2);
    const CHECKED_SHR_U8_A: Option<u8> = 1u8.checked_shr(8);
    const CHECKED_SHR_U8_B: Option<u8> = 1u8.checked_shr(0);

    const CHECKED_ABS_I32_A: Option<i32> = 5i32.checked_abs();
    const CHECKED_ABS_I8_A: Option<i8> = (-127 as i8).checked_abs();
    const CHECKED_ABS_I8_B: Option<i8> = 1i8.checked_abs();
    const CHECKED_ABS_I8_C: Option<i8> = i8::min_value().checked_abs();
}
