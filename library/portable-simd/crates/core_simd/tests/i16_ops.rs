#![feature(portable_simd)]
use core_simd::i16x2;

#[macro_use]
mod ops_macros;
impl_signed_tests! { i16 }

#[test]
fn max_is_not_lexicographic() {
    let a = i16x2::splat(10);
    let b = i16x2::from_array([-4, 12]);
    assert_eq!(a.max(b), i16x2::from_array([10, 12]));
}

#[test]
fn min_is_not_lexicographic() {
    let a = i16x2::splat(10);
    let b = i16x2::from_array([12, -4]);
    assert_eq!(a.min(b), i16x2::from_array([10, -4]));
}

#[test]
fn clamp_is_not_lexicographic() {
    let a = i16x2::splat(10);
    let lo = i16x2::from_array([-12, -4]);
    let up = i16x2::from_array([-4, 12]);
    assert_eq!(a.clamp(lo, up), i16x2::from_array([-4, 10]));

    let x = i16x2::from_array([1, 10]);
    let y = x.clamp(i16x2::splat(0), i16x2::splat(9));
    assert_eq!(y, i16x2::from_array([1, 9]));
}
