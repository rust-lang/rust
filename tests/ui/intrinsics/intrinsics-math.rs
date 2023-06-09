// run-pass
// ignore-emscripten fma not implemented in emscripten

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
}

pub fn main() {
    use std::f32;
    use std::f64;

    assert_approx_eq!(64f32.sqrt(), 8f32);
    assert_approx_eq!(64f64.sqrt(), 8f64);

    assert_approx_eq!(25f32.powi(-2), 0.0016f32);
    assert_approx_eq!(23.2f64.powi(2), 538.24f64);

    assert_approx_eq!(0f32.sin(), 0f32);
    assert_approx_eq!((f64::consts::PI / 2f64).sin(), 1f64);

    assert_approx_eq!(0f32.cos(), 1f32);
    assert_approx_eq!((f64::consts::PI * 2f64).cos(), 1f64);

    assert_approx_eq!(25f32.powf(-2f32), 0.0016f32);
    assert_approx_eq!(400f64.powf(0.5f64), 20f64);

    assert_approx_eq!((1f32.exp() - f32::consts::E).abs(), 0f32);
    assert_approx_eq!(1f64.exp(), f64::consts::E);

    assert_approx_eq!(10f32.exp2(), 1024f32);
    assert_approx_eq!(50f64.exp2(), 1125899906842624f64);

    assert_approx_eq!((f32::consts::E.ln() - 1f32).abs(), 0f32);
    assert_approx_eq!(1f64.ln(), 0f64);

    assert_approx_eq!(10f32.log10(), 1f32);
    assert_approx_eq!(f64::consts::E.log10(), f64::consts::LOG10_E);

    assert_approx_eq!(8f32.log2(), 3f32);
    assert_approx_eq!(f64::consts::E.log2(), f64::consts::LOG2_E);

    assert_approx_eq!(1.0f32.mul_add(2.0f32, 5.0f32), 7.0f32);
    assert_approx_eq!(0.0f64.mul_add(-2.0f64, f64::consts::E), f64::consts::E);

    assert_approx_eq!((-1.0f32).abs(), 1.0f32);
    assert_approx_eq!(34.2f64.abs(), 34.2f64);

    assert_approx_eq!(3.8f32.floor(), 3.0f32);
    assert_approx_eq!((-1.1f64).floor(), -2.0f64);

    assert_approx_eq!((-2.3f32).ceil(), -2.0f32);
    assert_approx_eq!(3.8f64.ceil(), 4.0f64);

    assert_approx_eq!(0.1f32.trunc(), 0.0f32);
    assert_approx_eq!((-0.1f64).trunc(), 0.0f64);
}
