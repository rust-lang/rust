use std;
import std::math::*;
import std::float;

#[test]
fn test_sqrt() {
    assert sqrt(9.0) == 3.0;
    assert sqrt(4.0) == 2.0;
    assert sqrt(1.0) == 1.0;
    assert sqrt(0.0) == 0.0;
}

#[test]
fn test_max_min() {
    assert max(0, 1) == 1;
    assert min(0, 1) == 0;
    assert max(0, -1) == 0;
    assert min(0, -1) == -1;
    assert max(0.0, 1.0) == 1.0;
    assert min(0.0, 1.0) == 0.0;
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
#[ignore]
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