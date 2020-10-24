// run-pass
// Catch mistakes in the overflowing literals lint.
#![deny(overflowing_literals)]

pub fn main() {
    assert_eq!(0xffffffff, (!0 as u32));
    assert_eq!(4294967295, (!0 as u32));
    assert_eq!(0xffffffffffffffff, (!0 as u64));
    assert_eq!(18446744073709551615, (!0 as u64));

    assert_eq!((-2147483648i32).wrapping_sub(1), 2147483647);

    assert_eq!(-3.40282356e+38_f32, f32::MIN);
    assert_eq!(3.40282356e+38_f32, f32::MAX);
    assert_eq!(-1.7976931348623158e+308_f64, f64::MIN);
    assert_eq!(1.7976931348623158e+308_f64, f64::MAX);
}
