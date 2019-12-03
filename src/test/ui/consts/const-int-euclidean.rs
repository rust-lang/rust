// run-pass
#![feature(const_int_euclidean)]
#![feature(saturating_neg)]

macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $(assert_eq!($exp, $ident);)+
        }
    }
}

assert_same_const! {
    const CHECKED_DIV_I32_A: Option<i32> = 5i32.checked_div_euclid(7777);
    const CHECKED_DIV_I8_A: Option<i8> = (-127 as i8).checked_div_euclid(-99);
    const CHECKED_DIV_I8_B: Option<i8> = (-127 as i8).checked_div_euclid(1);
    const CHECKED_DIV_I8_C: Option<i8> = i8::min_value().checked_div_euclid(-1);
    const CHECKED_DIV_U8_A: Option<u8> = 1u8.checked_div_euclid(3);

    const CHECKED_REM_I32_A: Option<i32> = 5i32.checked_rem_euclid(7777);
    const CHECKED_REM_I8_A: Option<i8> = (-127 as i8).checked_rem_euclid(-99);
    const CHECKED_REM_I8_B: Option<i8> = (-127 as i8).checked_rem_euclid(0);
    const CHECKED_REM_I8_C: Option<i8> = i8::min_value().checked_rem_euclid(-1);
    const CHECKED_REM_U8_A: Option<u8> = 1u8.checked_rem_euclid(3);

    const WRAPPING_DIV_I32_A: i32 = 5i32.wrapping_div_euclid(7777);
    const WRAPPING_DIV_I8_A: i8 = (-127 as i8).wrapping_div_euclid(-99);
    const WRAPPING_DIV_I8_B: i8 = (-127 as i8).wrapping_div_euclid(1);
    const WRAPPING_DIV_I8_C: i8 = i8::min_value().wrapping_div_euclid(-1);
    const WRAPPING_DIV_U8_A: u8 = 1u8.wrapping_div_euclid(3);

    const WRAPPING_REM_I32_A: i32 = 5i32.wrapping_rem_euclid(7777);
    const WRAPPING_REM_I8_A: i8 = (-127 as i8).wrapping_rem_euclid(-99);
    const WRAPPING_REM_I8_B: i8 = (-127 as i8).wrapping_rem_euclid(1);
    const WRAPPING_REM_I8_C: i8 = i8::min_value().wrapping_rem_euclid(-1);
    const WRAPPING_REM_U8_A: u8 = 1u8.wrapping_rem_euclid(3);

    const OVERFLOWING_DIV_I32_A: (i32, bool) = 5i32.overflowing_div_euclid(7777);
    const OVERFLOWING_DIV_I8_A: (i8, bool) = (-127 as i8).overflowing_div_euclid(-99);
    const OVERFLOWING_DIV_I8_B: (i8, bool) = (-127 as i8).overflowing_div_euclid(1);
    const OVERFLOWING_DIV_I8_C: (i8, bool) = i8::min_value().overflowing_div_euclid(-1);
    const OVERFLOWING_DIV_U8_A: (u8, bool) = 1u8.overflowing_div_euclid(3);

    const OVERFLOWING_REM_I32_A: (i32, bool) = 5i32.overflowing_rem_euclid(7777);
    const OVERFLOWING_REM_I8_A: (i8, bool) = (-127 as i8).overflowing_rem_euclid(-99);
    const OVERFLOWING_REM_I8_B: (i8, bool) = (-127 as i8).overflowing_rem_euclid(1);
    const OVERFLOWING_REM_I8_C: (i8, bool) = i8::min_value().overflowing_rem_euclid(-1);
    const OVERFLOWING_REM_U8_A: (u8, bool) = 1u8.overflowing_rem_euclid(3);
}
