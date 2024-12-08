use crate::ops::Mul;

#[test]
fn test_saturating_add_uint() {
    assert_eq!(3_usize.saturating_add(5_usize), 8_usize);
    assert_eq!(3_usize.saturating_add(usize::MAX - 1), usize::MAX);
    assert_eq!(usize::MAX.saturating_add(usize::MAX), usize::MAX);
    assert_eq!((usize::MAX - 2).saturating_add(1), usize::MAX - 1);
}

#[test]
fn test_saturating_sub_uint() {
    assert_eq!(5_usize.saturating_sub(3_usize), 2_usize);
    assert_eq!(3_usize.saturating_sub(5_usize), 0_usize);
    assert_eq!(0_usize.saturating_sub(1_usize), 0_usize);
    assert_eq!((usize::MAX - 1).saturating_sub(usize::MAX), 0);
}

#[test]
fn test_saturating_add_int() {
    assert_eq!(3i32.saturating_add(5), 8);
    assert_eq!(3isize.saturating_add(isize::MAX - 1), isize::MAX);
    assert_eq!(isize::MAX.saturating_add(isize::MAX), isize::MAX);
    assert_eq!((isize::MAX - 2).saturating_add(1), isize::MAX - 1);
    assert_eq!(3i32.saturating_add(-5), -2);
    assert_eq!(isize::MIN.saturating_add(-1), isize::MIN);
    assert_eq!((-2isize).saturating_add(-isize::MAX), isize::MIN);
}

#[test]
fn test_saturating_sub_int() {
    assert_eq!(3i32.saturating_sub(5), -2);
    assert_eq!(isize::MIN.saturating_sub(1), isize::MIN);
    assert_eq!((-2isize).saturating_sub(isize::MAX), isize::MIN);
    assert_eq!(3i32.saturating_sub(-5), 8);
    assert_eq!(3isize.saturating_sub(-(isize::MAX - 1)), isize::MAX);
    assert_eq!(isize::MAX.saturating_sub(-isize::MAX), isize::MAX);
    assert_eq!((isize::MAX - 2).saturating_sub(-1), isize::MAX - 1);
}

#[test]
fn test_checked_add() {
    let five_less = usize::MAX - 5;
    assert_eq!(five_less.checked_add(0), Some(usize::MAX - 5));
    assert_eq!(five_less.checked_add(1), Some(usize::MAX - 4));
    assert_eq!(five_less.checked_add(2), Some(usize::MAX - 3));
    assert_eq!(five_less.checked_add(3), Some(usize::MAX - 2));
    assert_eq!(five_less.checked_add(4), Some(usize::MAX - 1));
    assert_eq!(five_less.checked_add(5), Some(usize::MAX));
    assert_eq!(five_less.checked_add(6), None);
    assert_eq!(five_less.checked_add(7), None);
}

#[test]
fn test_checked_sub() {
    assert_eq!(5_usize.checked_sub(0), Some(5));
    assert_eq!(5_usize.checked_sub(1), Some(4));
    assert_eq!(5_usize.checked_sub(2), Some(3));
    assert_eq!(5_usize.checked_sub(3), Some(2));
    assert_eq!(5_usize.checked_sub(4), Some(1));
    assert_eq!(5_usize.checked_sub(5), Some(0));
    assert_eq!(5_usize.checked_sub(6), None);
    assert_eq!(5_usize.checked_sub(7), None);
}

#[test]
fn test_checked_mul() {
    let third = usize::MAX / 3;
    assert_eq!(third.checked_mul(0), Some(0));
    assert_eq!(third.checked_mul(1), Some(third));
    assert_eq!(third.checked_mul(2), Some(third * 2));
    assert_eq!(third.checked_mul(3), Some(third * 3));
    assert_eq!(third.checked_mul(4), None);
}

macro_rules! test_is_power_of_two {
    ($test_name:ident, $T:ident) => {
        #[test]
        fn $test_name() {
            assert_eq!((0 as $T).is_power_of_two(), false);
            assert_eq!((1 as $T).is_power_of_two(), true);
            assert_eq!((2 as $T).is_power_of_two(), true);
            assert_eq!((3 as $T).is_power_of_two(), false);
            assert_eq!((4 as $T).is_power_of_two(), true);
            assert_eq!((5 as $T).is_power_of_two(), false);
            assert_eq!(($T::MAX / 2 + 1).is_power_of_two(), true);
        }
    };
}

test_is_power_of_two! { test_is_power_of_two_u8, u8 }
test_is_power_of_two! { test_is_power_of_two_u16, u16 }
test_is_power_of_two! { test_is_power_of_two_u32, u32 }
test_is_power_of_two! { test_is_power_of_two_u64, u64 }
test_is_power_of_two! { test_is_power_of_two_uint, usize }

macro_rules! test_next_power_of_two {
    ($test_name:ident, $T:ident) => {
        #[test]
        fn $test_name() {
            assert_eq!((0 as $T).next_power_of_two(), 1);
            let mut next_power = 1;
            for i in 1 as $T..40 {
                assert_eq!(i.next_power_of_two(), next_power);
                if i == next_power {
                    next_power *= 2
                }
            }
        }
    };
}

test_next_power_of_two! { test_next_power_of_two_u8, u8 }
test_next_power_of_two! { test_next_power_of_two_u16, u16 }
test_next_power_of_two! { test_next_power_of_two_u32, u32 }
test_next_power_of_two! { test_next_power_of_two_u64, u64 }
test_next_power_of_two! { test_next_power_of_two_uint, usize }

macro_rules! test_checked_next_power_of_two {
    ($test_name:ident, $T:ident) => {
        #[test]
        fn $test_name() {
            assert_eq!((0 as $T).checked_next_power_of_two(), Some(1));
            let smax = $T::MAX >> 1;
            assert_eq!(smax.checked_next_power_of_two(), Some(smax + 1));
            assert_eq!((smax + 1).checked_next_power_of_two(), Some(smax + 1));
            assert_eq!((smax + 2).checked_next_power_of_two(), None);
            assert_eq!(($T::MAX - 1).checked_next_power_of_two(), None);
            assert_eq!($T::MAX.checked_next_power_of_two(), None);
            let mut next_power = 1;
            for i in 1 as $T..40 {
                assert_eq!(i.checked_next_power_of_two(), Some(next_power));
                if i == next_power {
                    next_power *= 2
                }
            }
        }
    };
}

test_checked_next_power_of_two! { test_checked_next_power_of_two_u8, u8 }
test_checked_next_power_of_two! { test_checked_next_power_of_two_u16, u16 }
test_checked_next_power_of_two! { test_checked_next_power_of_two_u32, u32 }
test_checked_next_power_of_two! { test_checked_next_power_of_two_u64, u64 }
test_checked_next_power_of_two! { test_checked_next_power_of_two_uint, usize }

#[test]
fn test_pow() {
    fn naive_pow<T: Mul<Output = T> + Copy>(one: T, base: T, exp: usize) -> T {
        (0..exp).fold(one, |acc, _| acc * base)
    }
    macro_rules! assert_pow {
        (($num:expr, $exp:expr) => $expected:expr) => {{
            let result = $num.pow($exp);
            assert_eq!(result, $expected);
            assert_eq!(result, naive_pow(1, $num, $exp));
        }};
    }
    assert_pow!((3u32,     0 ) => 1);
    assert_pow!((5u32,     1 ) => 5);
    assert_pow!((-4i32,    2 ) => 16);
    assert_pow!((8u32,     3 ) => 512);
    assert_pow!((2u64,     50) => 1125899906842624);
}

#[test]
fn test_uint_to_str_overflow() {
    let mut u8_val: u8 = 255;
    assert_eq!(u8_val.to_string(), "255");

    u8_val = u8_val.wrapping_add(1);
    assert_eq!(u8_val.to_string(), "0");

    let mut u16_val: u16 = 65_535;
    assert_eq!(u16_val.to_string(), "65535");

    u16_val = u16_val.wrapping_add(1);
    assert_eq!(u16_val.to_string(), "0");

    let mut u32_val: u32 = 4_294_967_295;
    assert_eq!(u32_val.to_string(), "4294967295");

    u32_val = u32_val.wrapping_add(1);
    assert_eq!(u32_val.to_string(), "0");

    let mut u64_val: u64 = 18_446_744_073_709_551_615;
    assert_eq!(u64_val.to_string(), "18446744073709551615");

    u64_val = u64_val.wrapping_add(1);
    assert_eq!(u64_val.to_string(), "0");
}

fn from_str<T: crate::str::FromStr>(t: &str) -> Option<T> {
    crate::str::FromStr::from_str(t).ok()
}

#[test]
fn test_uint_from_str_overflow() {
    let mut u8_val: u8 = 255;
    assert_eq!(from_str::<u8>("255"), Some(u8_val));
    assert_eq!(from_str::<u8>("256"), None);

    u8_val = u8_val.wrapping_add(1);
    assert_eq!(from_str::<u8>("0"), Some(u8_val));
    assert_eq!(from_str::<u8>("-1"), None);

    let mut u16_val: u16 = 65_535;
    assert_eq!(from_str::<u16>("65535"), Some(u16_val));
    assert_eq!(from_str::<u16>("65536"), None);

    u16_val = u16_val.wrapping_add(1);
    assert_eq!(from_str::<u16>("0"), Some(u16_val));
    assert_eq!(from_str::<u16>("-1"), None);

    let mut u32_val: u32 = 4_294_967_295;
    assert_eq!(from_str::<u32>("4294967295"), Some(u32_val));
    assert_eq!(from_str::<u32>("4294967296"), None);

    u32_val = u32_val.wrapping_add(1);
    assert_eq!(from_str::<u32>("0"), Some(u32_val));
    assert_eq!(from_str::<u32>("-1"), None);

    let mut u64_val: u64 = 18_446_744_073_709_551_615;
    assert_eq!(from_str::<u64>("18446744073709551615"), Some(u64_val));
    assert_eq!(from_str::<u64>("18446744073709551616"), None);

    u64_val = u64_val.wrapping_add(1);
    assert_eq!(from_str::<u64>("0"), Some(u64_val));
    assert_eq!(from_str::<u64>("-1"), None);
}
