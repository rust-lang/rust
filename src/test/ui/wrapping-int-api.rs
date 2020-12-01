// run-pass
// Test inherent wrapping_* methods for {i,u}{size,8,16,32,64}.

// Don't warn about overflowing ops on 32-bit platforms
#![cfg_attr(target_pointer_width = "32", allow(const_err))]

fn main() {
    assert_eq!(   i8::MAX.wrapping_add(1),    i8::MIN);
    assert_eq!(  i16::MAX.wrapping_add(1),   i16::MIN);
    assert_eq!(  i32::MAX.wrapping_add(1),   i32::MIN);
    assert_eq!(  i64::MAX.wrapping_add(1),   i64::MIN);
    assert_eq!(isize::MAX.wrapping_add(1), isize::MIN);

    assert_eq!(   i8::MIN.wrapping_sub(1),    i8::MAX);
    assert_eq!(  i16::MIN.wrapping_sub(1),   i16::MAX);
    assert_eq!(  i32::MIN.wrapping_sub(1),   i32::MAX);
    assert_eq!(  i64::MIN.wrapping_sub(1),   i64::MAX);
    assert_eq!(isize::MIN.wrapping_sub(1), isize::MAX);

    assert_eq!(   u8::MAX.wrapping_add(1),    u8::MIN);
    assert_eq!(  u16::MAX.wrapping_add(1),   u16::MIN);
    assert_eq!(  u32::MAX.wrapping_add(1),   u32::MIN);
    assert_eq!(  u64::MAX.wrapping_add(1),   u64::MIN);
    assert_eq!(usize::MAX.wrapping_add(1), usize::MIN);

    assert_eq!(   u8::MIN.wrapping_sub(1),    u8::MAX);
    assert_eq!(  u16::MIN.wrapping_sub(1),   u16::MAX);
    assert_eq!(  u32::MIN.wrapping_sub(1),   u32::MAX);
    assert_eq!(  u64::MIN.wrapping_sub(1),   u64::MAX);
    assert_eq!(usize::MIN.wrapping_sub(1), usize::MAX);

    assert_eq!((0xfe_u8 as i8).wrapping_mul(16),
               (0xe0_u8 as i8));
    assert_eq!((0xfedc_u16 as i16).wrapping_mul(16),
               (0xedc0_u16 as i16));
    assert_eq!((0xfedc_ba98_u32 as i32).wrapping_mul(16),
               (0xedcb_a980_u32 as i32));
    assert_eq!((0xfedc_ba98_7654_3217_u64 as i64).wrapping_mul(16),
               (0xedcb_a987_6543_2170_u64 as i64));

    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            assert_eq!((0xfedc_ba98_u32 as isize).wrapping_mul(16),
                       (0xedcb_a980_u32 as isize));
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            assert_eq!((0xfedc_ba98_7654_3217_u64 as isize).wrapping_mul(16),
                       (0xedcb_a987_6543_2170_u64 as isize));
        }
    }

    assert_eq!((0xfe as u8).wrapping_mul(16),
               (0xe0 as u8));
    assert_eq!((0xfedc as u16).wrapping_mul(16),
               (0xedc0 as u16));
    assert_eq!((0xfedc_ba98 as u32).wrapping_mul(16),
               (0xedcb_a980 as u32));
    assert_eq!((0xfedc_ba98_7654_3217 as u64).wrapping_mul(16),
               (0xedcb_a987_6543_2170 as u64));

    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            assert_eq!((0xfedc_ba98 as usize).wrapping_mul(16),
                       (0xedcb_a980 as usize));
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            assert_eq!((0xfedc_ba98_7654_3217 as usize).wrapping_mul(16),
                       (0xedcb_a987_6543_2170 as usize));
        }
    }

    macro_rules! check_mul_no_wrap {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_mul($f), ($e) * $f); }
    }
    macro_rules! check_mul_wraps {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_mul($f), $e); }
    }

    check_mul_no_wrap!(0xfe_u8 as i8, -1);
    check_mul_no_wrap!(0xfedc_u16 as i16, -1);
    check_mul_no_wrap!(0xfedc_ba98_u32 as i32, -1);
    check_mul_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -1);
    check_mul_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, -1);

    check_mul_no_wrap!(0xfe_u8 as i8, -2);
    check_mul_no_wrap!(0xfedc_u16 as i16, -2);
    check_mul_no_wrap!(0xfedc_ba98_u32 as i32, -2);
    check_mul_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -2);
    check_mul_no_wrap!(0xfedc_ba98_fedc_ba98_u64 as u64 as isize, -2);

    check_mul_no_wrap!(0xfe_u8 as i8, 2);
    check_mul_no_wrap!(0xfedc_u16 as i16, 2);
    check_mul_no_wrap!(0xfedc_ba98_u32 as i32, 2);
    check_mul_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, 2);
    check_mul_no_wrap!(0xfedc_ba98_fedc_ba98_u64 as u64 as isize, 2);

    check_mul_wraps!(0x80_u8 as i8, -1);
    check_mul_wraps!(0x8000_u16 as i16, -1);
    check_mul_wraps!(0x8000_0000_u32 as i32, -1);
    check_mul_wraps!(0x8000_0000_0000_0000_u64 as i64, -1);
    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            check_mul_wraps!(0x8000_0000_u32 as isize, -1);
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            check_mul_wraps!(0x8000_0000_0000_0000_u64 as isize, -1);
        }
    }

    macro_rules! check_div_no_wrap {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_div($f), ($e) / $f); }
    }
    macro_rules! check_div_wraps {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_div($f), $e); }
    }

    check_div_no_wrap!(0xfe_u8 as i8, -1);
    check_div_no_wrap!(0xfedc_u16 as i16, -1);
    check_div_no_wrap!(0xfedc_ba98_u32 as i32, -1);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -1);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, -1);

    check_div_no_wrap!(0xfe_u8 as i8, -2);
    check_div_no_wrap!(0xfedc_u16 as i16, -2);
    check_div_no_wrap!(0xfedc_ba98_u32 as i32, -2);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -2);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, -2);

    check_div_no_wrap!(0xfe_u8 as i8, 2);
    check_div_no_wrap!(0xfedc_u16 as i16, 2);
    check_div_no_wrap!(0xfedc_ba98_u32 as i32, 2);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, 2);
    check_div_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, 2);

    check_div_wraps!(-128 as i8, -1);
    check_div_wraps!(0x8000_u16 as i16, -1);
    check_div_wraps!(0x8000_0000_u32 as i32, -1);
    check_div_wraps!(0x8000_0000_0000_0000_u64 as i64, -1);
    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            check_div_wraps!(0x8000_0000_u32 as isize, -1);
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            check_div_wraps!(0x8000_0000_0000_0000_u64 as isize, -1);
        }
    }


    macro_rules! check_rem_no_wrap {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_rem($f), ($e) % $f); }
    }
    macro_rules! check_rem_wraps {
        ($e:expr, $f:expr) => { assert_eq!(($e).wrapping_rem($f), 0); }
    }

    check_rem_no_wrap!(0xfe_u8 as i8, -1);
    check_rem_no_wrap!(0xfedc_u16 as i16, -1);
    check_rem_no_wrap!(0xfedc_ba98_u32 as i32, -1);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -1);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, -1);

    check_rem_no_wrap!(0xfe_u8 as i8, -2);
    check_rem_no_wrap!(0xfedc_u16 as i16, -2);
    check_rem_no_wrap!(0xfedc_ba98_u32 as i32, -2);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, -2);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, -2);

    check_rem_no_wrap!(0xfe_u8 as i8, 2);
    check_rem_no_wrap!(0xfedc_u16 as i16, 2);
    check_rem_no_wrap!(0xfedc_ba98_u32 as i32, 2);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64, 2);
    check_rem_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize, 2);

    check_rem_wraps!(0x80_u8 as i8, -1);
    check_rem_wraps!(0x8000_u16 as i16, -1);
    check_rem_wraps!(0x8000_0000_u32 as i32, -1);
    check_rem_wraps!(0x8000_0000_0000_0000_u64 as i64, -1);
    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            check_rem_wraps!(0x8000_0000_u32 as isize, -1);
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            check_rem_wraps!(0x8000_0000_0000_0000_u64 as isize, -1);
        }
    }

    macro_rules! check_neg_no_wrap {
        ($e:expr) => { assert_eq!(($e).wrapping_neg(), -($e)); }
    }
    macro_rules! check_neg_wraps {
        ($e:expr) => { assert_eq!(($e).wrapping_neg(),  ($e)); }
    }

    check_neg_no_wrap!(0xfe_u8 as i8);
    check_neg_no_wrap!(0xfedc_u16 as i16);
    check_neg_no_wrap!(0xfedc_ba98_u32 as i32);
    check_neg_no_wrap!(0xfedc_ba98_7654_3217_u64 as i64);
    check_neg_no_wrap!(0xfedc_ba98_7654_3217_u64 as u64 as isize);

    check_neg_wraps!(0x80_u8 as i8);
    check_neg_wraps!(0x8000_u16 as i16);
    check_neg_wraps!(0x8000_0000_u32 as i32);
    check_neg_wraps!(0x8000_0000_0000_0000_u64 as i64);
    match () {
        #[cfg(target_pointer_width = "32")]
        () => {
            check_neg_wraps!(0x8000_0000_u32 as isize);
        }
        #[cfg(target_pointer_width = "64")]
        () => {
            check_neg_wraps!(0x8000_0000_0000_0000_u64 as isize);
        }
    }

}
