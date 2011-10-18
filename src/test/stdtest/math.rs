use std;
import std::math::*;

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
          (0f, y) when y < 0f { 1.5 * std::math::pi }
          (0f, y) { 0.5 * std::math::pi }
          (x, y) { std::math::atan(y / x) }
        }
    }
    assert angle((1f, 0f)) == 0f;
    assert angle((1f, 1f)) == 0.25 * pi;
    assert angle((0f, 1f)) == 0.5 * pi;
}
