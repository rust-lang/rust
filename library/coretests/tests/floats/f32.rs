use core::f32;

use super::assert_biteq;

// FIXME(#140515): mingw has an incorrect fma https://sourceforge.net/p/mingw-w64/bugs/848/
#[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)]
#[test]
fn test_mul_add() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_biteq!(f32::math::mul_add(12.3f32, 4.5, 6.7), 62.05);
    assert_biteq!(f32::math::mul_add(-12.3f32, -4.5, -6.7), 48.65);
    assert_biteq!(f32::math::mul_add(0.0f32, 8.9, 1.2), 1.2);
    assert_biteq!(f32::math::mul_add(3.4f32, -0.0, 5.6), 5.6);
    assert!(f32::math::mul_add(nan, 7.8, 9.0).is_nan());
    assert_biteq!(f32::math::mul_add(inf, 7.8, 9.0), inf);
    assert_biteq!(f32::math::mul_add(neg_inf, 7.8, 9.0), neg_inf);
    assert_biteq!(f32::math::mul_add(8.9f32, inf, 3.2), inf);
    assert_biteq!(f32::math::mul_add(-3.2f32, 2.4, neg_inf), neg_inf);
}
