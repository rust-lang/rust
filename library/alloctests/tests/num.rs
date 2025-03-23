use std::fmt::{Debug, Display};
use std::str::FromStr;

fn assert_nb<Int: ToString + FromStr + Debug + Display + Eq>(value: Int) {
    let s = value.to_string();
    let s2 = format!("s: {}.", value);

    assert_eq!(format!("s: {s}."), s2);
    let Ok(ret) = Int::from_str(&s) else {
        panic!("failed to convert into to string");
    };
    assert_eq!(ret, value);
}

macro_rules! uint_to_s {
    ($($fn_name:ident, $int:ident,)+) => {
        $(
            #[test]
            fn $fn_name() {
                assert_nb::<$int>($int::MIN);
                assert_nb::<$int>($int::MAX);
                assert_nb::<$int>(1);
                assert_nb::<$int>($int::MIN / 2);
                assert_nb::<$int>($int::MAX / 2);
            }
        )+
    }
}
macro_rules! int_to_s {
    ($($fn_name:ident, $int:ident,)+) => {
        $(
            #[test]
            fn $fn_name() {
                assert_nb::<$int>($int::MIN);
                assert_nb::<$int>($int::MAX);
                assert_nb::<$int>(1);
                assert_nb::<$int>(0);
                assert_nb::<$int>(-1);
                assert_nb::<$int>($int::MIN / 2);
                assert_nb::<$int>($int::MAX / 2);
            }
        )+
    }
}

int_to_s!(
    test_i8_to_string,
    i8,
    test_i16_to_string,
    i16,
    test_i32_to_string,
    i32,
    test_i64_to_string,
    i64,
    test_i128_to_string,
    i128,
);
uint_to_s!(
    test_u8_to_string,
    u8,
    test_u16_to_string,
    u16,
    test_u32_to_string,
    u32,
    test_u64_to_string,
    u64,
    test_u128_to_string,
    u128,
);
