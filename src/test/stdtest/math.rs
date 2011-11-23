use std;

import std::math::*;
import std::float;
import c_int = std::ctypes::c_int;

#[test]
fn test_max_min() {
    assert max(0, 1) == 1;
    assert min(0, 1) == 0;
    assert max(0, -1) == 0;
    assert min(0, -1) == -1;
    assert max(0.0, 1.0) == 1.0;
    assert min(0.0, 1.0) == 0.0;
}

// FIXME use macros to execute the tests below for all float types

#[test]
fn test_trig() {
    assert sin(0.0) == 0.0;
    assert sin(-0.0) == 0.0;
    assert float::isNaN(sin(float::infinity));
    assert float::isNaN(sin(float::neg_infinity));

    assert cos(0.0) == 1.0;
    assert cos(-0.0) == 1.0;
    assert float::isNaN(cos(float::infinity));
    assert float::isNaN(cos(float::neg_infinity));

    assert tan(0.0) == 0.0;
    assert tan(-0.0) == 0.0;;
    assert float::isNaN(tan(float::infinity));
    assert float::isNaN(tan(float::neg_infinity));
}

#[test]
fn test_inv_trig() {
    assert asin(0.0) == 0.0;
    assert asin(-0.0) == -0.0;
    assert float::isNaN(asin(1.1));
    assert float::isNaN(asin(-1.1));

    assert acos(1.0) == 0.0;
    assert float::isNaN(acos(1.1));
    assert float::isNaN(acos(-1.1));

    assert atan(0.0) == 0.0;
    assert atan(-0.0) == 0.0;
    assert atan(float::infinity) == consts::frac_pi_2;
    assert atan(float::neg_infinity) == - consts::frac_pi_2;

    assert atan2(0.0, -0.0) == consts::pi;
    assert atan2(-0.0, -0.0) == -consts::pi;

    assert atan2(0.0, 0.0) == 0.0;
    assert atan2(-0.0, 0.0) == -0.0;

    assert atan2(0.0, -1.0) == consts::pi;
    assert atan2(-0.0, -1.0) == -consts::pi;

    assert atan2(0.0, 1.0) == 0.0;
    assert atan2(-0.0, 1.0) == -0.0;

    assert atan2(1.0, 0.0) == consts::frac_pi_2;
    assert atan2(1.0, -0.0) == consts::frac_pi_2;
}

#[test]
fn test_pow() {
    assert pow(2.0, 4.0) == 16.0;

    assert pow(0.0, -3.0) == float::infinity;
    assert pow(-0.0, -3.0) == float::neg_infinity;

    assert pow(0.0, -4.0) == float::infinity;
    assert pow(-0.0, -4.0) == float::infinity;

    assert pow(0.0, 3.0) == 0.0;
    assert pow(-0.0, 3.0) == -0.0;
    assert pow(0.0, 4.0) == 0.0;
    assert pow(-0.0, 4.0) == 0.0;

    assert pow(-1.0, float::infinity) == 1.0;
    assert pow(-1.0, float::neg_infinity) == 1.0;

    assert pow(1.0, 4.0) == 1.0;
    assert pow(1.0, 0.0) == 1.0;
    assert pow(1.0, -0.0) == 1.0;
    assert pow(1.0, float::NaN) == 1.0;
    assert pow(1.0, float::infinity) == 1.0;
    assert pow(1.0, float::neg_infinity) == 1.0;
    assert pow(1.0, -3.0) == 1.0;
    assert pow(1.0, -4.0) == 1.0;

    assert pow(4.0, 0.0) == 1.0;
    assert pow(0.0, 0.0) == 1.0;
    assert pow(-0.0, 0.0) == 1.0;
    assert pow(float::NaN, 0.0) == 1.0;
    assert pow(float::infinity, 0.0) == 1.0;
    assert pow(float::neg_infinity, 0.0) == 1.0;
    assert pow(-3.0, 0.0) == 1.0;
    assert pow(-4.0, 0.0) == 1.0;

    assert pow(4.0, -0.0) == 1.0;
    assert pow(0.0, -0.0) == 1.0;
    assert pow(-0.0, -0.0) == 1.0;
    assert pow(float::NaN, -0.0) == 1.0;
    assert pow(float::infinity, -0.0) == 1.0;
    assert pow(float::neg_infinity, -0.0) == 1.0;
    assert pow(-3.0, -0.0) == 1.0;
    assert pow(-4.0, -0.0) == 1.0;

    assert float::isNaN(pow(-1.0, -1.5));
    assert float::isNaN(pow(-1.0, 1.5));

    assert float::isNaN(pow(-1.2, -1.5));
    assert float::isNaN(pow(-1.2, 1.5));

    assert pow(0.5, float::neg_infinity) == float::infinity;
    assert pow(-0.5, float::neg_infinity) == float::infinity;

    assert pow(1.5, float::neg_infinity) == 0.0;
    assert pow(-1.5, float::neg_infinity) == 0.0;

    assert pow(0.5, float::infinity) == 0.0;
    assert pow(-0.5, float::infinity) == 0.0;

    assert pow(-1.5, float::infinity) == float::infinity;
    assert pow(1.5, float::infinity) == float::infinity;

    assert pow(float::neg_infinity, -3.0) == -0.0;
    assert pow(float::neg_infinity, -4.0) == 0.0;

    assert pow(float::neg_infinity, 3.0) == float::neg_infinity;
    assert pow(float::neg_infinity, 4.0) == float::infinity;

    assert pow(float::infinity, -16.0) == 0.0;
    assert pow(float::infinity, 16.0) == float::infinity;
}

#[test]
fn test_exp_and_mod() {
    assert exp(0.0) == 1.0;
    assert exp(-0.0) == 1.0;
    assert exp(float::neg_infinity) == 0.0;
    assert exp(float::infinity) == float::infinity;

    let d1: c_int = 1 as c_int;
    assert frexp(0.0, d1) == 0.0;
    assert frexp(-0.0, d1) == 0.0;
    assert frexp(float::infinity, d1) == float::infinity;
    assert frexp(float::neg_infinity, d1) == float::neg_infinity;
    assert float::isNaN(frexp(float::NaN, d1));

    let d2: float = 1.0;
    assert modf(float::infinity, d2) == 0.0;
    assert modf(float::neg_infinity, d2) == -0.0;
    assert float::isNaN(modf(float::NaN, d2));
}

#[test]
fn test_round_and_abs() {
    assert abs(0.0) == 0.0;
    assert abs(-0.0) == 0.0;
    assert abs(float::infinity) == float::infinity;
    assert abs(float::neg_infinity) == float::infinity;

    assert abs(-2.5) == 2.5;
    assert abs(2.5) == 2.5;

    assert ceil(0.0) == 0.0;
    assert ceil(-0.0) == -0.0;
    assert ceil(float::infinity) == float::infinity;
    assert ceil(float::neg_infinity) == float::neg_infinity;

    assert ceil(1.9) == 2.0;
    assert ceil(-1.9) == -1.0;

    assert floor(0.0) == 0.0;
    assert floor(-0.0) == -0.0;
    assert floor(float::infinity) == float::infinity;
    assert floor(float::neg_infinity) == float::neg_infinity;

    assert floor(1.9) == 1.0;
    assert floor(-1.9) == -2.0;

    assert trunc(0.0) == 0.0;
    assert trunc(-0.0) == -0.0;
    assert trunc(float::infinity) == float::infinity;
    assert trunc(float::neg_infinity) == float::neg_infinity;

    assert trunc(1.5) == 1.0;
    assert trunc(1.2) == 1.0;
    assert trunc(1.0) == 1.0;
    assert trunc(1.9) == 1.0;
    assert trunc(-1.5) == -1.0;
    assert trunc(-1.2) == -1.0;
    assert trunc(-1.0) == -1.0;
    assert trunc(-1.9) == -1.0;

    assert round(0.0) == 0.0;
    assert round(-0.0) == -0.0;
    assert round(float::infinity) == float::infinity;
    assert round(float::neg_infinity) == float::neg_infinity;

    assert rint(0.0) == 0.0;
    assert rint(-0.0) == -0.0;
    assert rint(float::infinity) == float::infinity;
    assert rint(float::neg_infinity) == float::neg_infinity;
}

#[test]
fn test_hyp_trig() {
    assert sinh(0.0) == 0.0;
    assert sinh(-0.0) == 0.0;
    assert sinh(float::infinity) == float::infinity;
    assert sinh(float::neg_infinity) == float::neg_infinity;

    assert cosh(0.0) == 1.0;
    assert cosh(-0.0) == 1.0;
    assert cosh(float::infinity) == float::infinity;
    assert cosh(float::neg_infinity) == float::infinity;

    assert tanh(0.0) == 0.0;
    assert tanh(-0.0) == 0.0;
    assert tanh(float::infinity) == 1.0;
    assert tanh(float::neg_infinity) == -1.0;
}

#[test]
fn test_sqrt() {
    assert sqrt(9.0) == 3.0;
    assert sqrt(4.0) == 2.0;
    assert sqrt(1.0) == 1.0;
    assert sqrt(0.0) == 0.0;
}


#[test]
fn test_angle() {
    fn angle(vec: (float, float)) -> float {
        alt vec {
          (0f, y) when y < 0f { 1.5 * consts::pi }
          (0f, y) { 0.5 * consts::pi }
          (x, y) { std::math::atan(y / x) }
        }
    }
    assert angle((1f, 0f)) == 0f;
    assert angle((1f, 1f)) == 0.25 * consts::pi;
    assert angle((0f, 1f)) == 0.5 * consts::pi;
}


#[test]
fn test_log_functions() {
    assert ln(1.0) == 0.0;
    assert log2(1.0) == 0.0;
    assert log10(1.0) == 0.0;

    assert ln(consts::e) == 1.0;
    assert log2(2.0) == 1.0;
    assert log10(10.0) == 1.0;

    assert ln(consts::e*consts::e*consts::e*consts::e) == 4.0;
    assert log2(256.0) == 8.0;
    assert log10(1000.0) == 3.0;

    assert ln(0.0) == float::neg_infinity;
    assert log2(0.0) == float::neg_infinity;
    assert log10(0.0) == float::neg_infinity;

    assert ln(-0.0) == float::neg_infinity;
    assert log2(-0.0) == float::neg_infinity;
    assert log10(-0.0) == float::neg_infinity;

    assert float::isNaN(ln(-1.0));
    assert float::isNaN(log2(-1.0));
    assert float::isNaN(log10(-1.0));

    assert ln(float::infinity) == float::infinity;
    assert log2(float::infinity) == float::infinity;
    assert log10(float::infinity) == float::infinity;

    assert ln1p(0.0) == 0.0;
    assert ln1p(-0.0) == 0.0;
    assert ln1p(-1.0) == float::neg_infinity;
    assert float::isNaN(ln1p(-2.0f));
    assert ln1p(float::infinity) == float::infinity;
}