// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: The Rust Project Developers (see https://thanks.rust-lang.org)

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6, "{} is not approximately equal to {}", *a, *b);
    }};
}

fn ldexp(a: f64, b: i32) -> f64 {
    extern "C" {
        fn ldexp(x: f64, n: i32) -> f64;
    }
    unsafe { ldexp(a, b) }
}

pub fn main() {
    use std::f32;
    use std::f64;

    assert_approx_eq!(64f32.sqrt(), 8f32);
    assert_approx_eq!(64f64.sqrt(), 8f64);

    assert_approx_eq!(25f32.powi(-2), 0.0016f32);
    assert_approx_eq!(23.2f64.powi(2), 538.24f64);

    assert_approx_eq!(25f32.powf(-2f32), 0.0016f32);
    assert_approx_eq!(400f64.powf(0.5f64), 20f64);

    assert_approx_eq!(1f32.exp(), f32::consts::E);
    assert_approx_eq!(1f64.exp(), f64::consts::E);

    assert_approx_eq!(1f32.exp_m1(), f32::consts::E - 1.0);
    assert_approx_eq!(1f64.exp_m1(), f64::consts::E - 1.0);

    assert_approx_eq!(10f32.exp2(), 1024f32);
    assert_approx_eq!(50f64.exp2(), 1125899906842624f64);

    assert_approx_eq!(f32::consts::E.ln(), 1f32);
    assert_approx_eq!(1f64.ln(), 0f64);

    assert_approx_eq!(0f32.ln_1p(), 0f32);
    assert_approx_eq!(0f64.ln_1p(), 0f64);

    assert_approx_eq!(10f32.log10(), 1f32);
    assert_approx_eq!(f64::consts::E.log10(), f64::consts::LOG10_E);

    assert_approx_eq!(8f32.log2(), 3f32);
    assert_approx_eq!(f64::consts::E.log2(), f64::consts::LOG2_E);

    assert_approx_eq!(3.0f32.mul_add(2.0f32, 5.0f32), 11.0);
    assert_eq!(0.0f32.mul_add(-2.0, f32::consts::E), f32::consts::E);
    assert_approx_eq!(3.0f64.mul_add(2.0, 5.0), 11.0);
    assert_eq!(0.0f64.mul_add(-2.0f64, f64::consts::E), f64::consts::E);
    assert_eq!((-3.2f32).mul_add(2.4, f32::NEG_INFINITY), f32::NEG_INFINITY);
    assert_eq!((-3.2f64).mul_add(2.4, f64::NEG_INFINITY), f64::NEG_INFINITY);

    assert_approx_eq!((-1.0f32).abs(), 1.0f32);
    assert_approx_eq!(34.2f64.abs(), 34.2f64);

    #[allow(deprecated)]
    {
        assert_approx_eq!(5.0f32.abs_sub(3.0), 2.0);
        assert_approx_eq!(3.0f64.abs_sub(5.0), 0.0);
    }

    assert_approx_eq!(3.8f32.floor(), 3.0f32);
    assert_approx_eq!((-1.1f64).floor(), -2.0f64);

    assert_approx_eq!((-2.3f32).ceil(), -2.0f32);
    assert_approx_eq!(3.8f64.ceil(), 4.0f64);

    assert_approx_eq!(0.1f32.trunc(), 0.0f32);
    assert_approx_eq!((-0.1f64).trunc(), 0.0f64);

    assert_approx_eq!(27.0f32.cbrt(), 3.0f32);
    assert_approx_eq!(27.0f64.cbrt(), 3.0f64);

    assert_approx_eq!(3.0f32.hypot(4.0f32), 5.0f32);
    assert_approx_eq!(3.0f64.hypot(4.0f64), 5.0f64);

    assert_eq!(3.3_f32.round(), 3.0);
    assert_eq!(3.3_f64.round(), 3.0);

    assert_eq!(ldexp(0.65f64, 3i32), 5.2f64);
    assert_eq!(ldexp(1.42, 0xFFFF), f64::INFINITY);
    assert_eq!(ldexp(1.42, -0xFFFF), 0f64);

    // Trigonometric functions.

    assert_approx_eq!(0f32.sin(), 0f32);
    assert_approx_eq!((f64::consts::PI / 2f64).sin(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.sin().asin(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.sin().asin(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.sinh(), 1.1752012f32);
    assert_approx_eq!(1.0f64.sinh(), 1.1752012f64);
    assert_approx_eq!(2.0f32.asinh(), 1.443635475178810342493276740273105f32);
    assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);

    assert_approx_eq!(0f32.cos(), 1f32);
    assert_approx_eq!((f64::consts::PI * 2f64).cos(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.cos().acos(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.cos().acos(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.cosh(), 1.54308f32);
    assert_approx_eq!(1.0f64.cosh(), 1.54308f64);
    assert_approx_eq!(2.0f32.acosh(), 1.31695789692481670862504634730796844f32);
    assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);

    assert_approx_eq!(1.0f32.tan(), 1.557408f32);
    assert_approx_eq!(1.0f64.tan(), 1.557408f64);
    assert_approx_eq!(1.0_f32, 1.0_f32.tan().atan());
    assert_approx_eq!(1.0_f64, 1.0_f64.tan().atan());
    assert_approx_eq!(1.0f32.atan2(2.0f32), 0.46364761f32);
    assert_approx_eq!(1.0f32.atan2(2.0f32), 0.46364761f32);

    assert_approx_eq!(
        1.0f32.tanh(),
        (1.0 - f32::consts::E.powi(-2)) / (1.0 + f32::consts::E.powi(-2))
    );
    assert_approx_eq!(
        1.0f64.tanh(),
        (1.0 - f64::consts::E.powi(-2)) / (1.0 + f64::consts::E.powi(-2))
    );
    assert_approx_eq!(0.5f32.atanh(), 0.54930614433405484569762261846126285f32);
    assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);
}
