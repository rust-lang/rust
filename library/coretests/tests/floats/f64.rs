use core::f64;

use super::assert_biteq;

// FIXME(#140515): mingw has an incorrect fma https://sourceforge.net/p/mingw-w64/bugs/848/
#[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)]
#[test]
fn test_mul_add() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_biteq!(12.3f64.mul_add(4.5, 6.7), 62.050000000000004);
    assert_biteq!((-12.3f64).mul_add(-4.5, -6.7), 48.650000000000006);
    assert_biteq!(0.0f64.mul_add(8.9, 1.2), 1.2);
    assert_biteq!(3.4f64.mul_add(-0.0, 5.6), 5.6);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_biteq!(inf.mul_add(7.8, 9.0), inf);
    assert_biteq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_biteq!(8.9f64.mul_add(inf, 3.2), inf);
    assert_biteq!((-3.2f64).mul_add(2.4, neg_inf), neg_inf);
}
