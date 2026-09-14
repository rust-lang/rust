use core::num::Complex;

use crate::math::libm_math::generic::{fmax, ilogb, scalbn};
use crate::support::{CastInto, Float};

/// Returns the quotient of `(a + ib)` and `(c + id)`.
///
/// This implementation uses the standard formula, but has special behavior when the output
/// of that formula has both a real and imaginary component that are NaN.
fn complex_div<F: Float>(mut a: F, mut b: F, mut c: F, mut d: F) -> Complex<F>
where
    u32: CastInto<F::Int>,
{
    let max = fmax(c.abs(), d.abs());
    let mut ilogbw = 0;
    if max.is_finite() && max != F::ZERO {
        ilogbw = ilogb(max);
        c = scalbn(c, -ilogbw);
        d = scalbn(d, -ilogbw);
    }

    let denom = c * c + d * d;
    let mut z = Complex::new(
        scalbn((a * c + b * d) / denom, -ilogbw),
        scalbn((b * c - a * d) / denom, -ilogbw),
    );

    // The fast path: exit when at least one component is not NaN.
    if !(z.re.is_nan() && z.im.is_nan()) {
        return z;
    }

    let signed_unit_if_inf = |x: F| {
        let mag = if x.is_infinite() { F::ONE } else { F::ZERO };
        mag.copysign(x)
    };

    if denom == F::ZERO && (!a.is_nan() || !b.is_nan()) {
        z.re = F::INFINITY.copysign(c) * a;
        z.im = F::INFINITY.copysign(c) * b;
    } else if (a.is_infinite() || b.is_infinite()) && c.is_finite() && d.is_finite() {
        a = signed_unit_if_inf(a);
        b = signed_unit_if_inf(b);
        z.re = F::INFINITY * (a * c + b * d);
        z.im = F::INFINITY * (b * c - a * d);
    } else if max.is_infinite() && a.is_finite() && b.is_finite() {
        c = signed_unit_if_inf(c);
        d = signed_unit_if_inf(d);
        z.re = F::ZERO * (a * c + b * d);
        z.im = F::ZERO * (b * c - a * d);
    }

    z
}

intrinsics! {
    #[cfg(all(f16_enabled, not(x86_no_sse2)))]
    pub extern "C" fn __rust_divhc3(a: f16, b: f16, c: f16, d: f16) -> core::num::Complex<f16> {
        complex_div(a, b, c, d)
    }

    pub extern "C" fn __rust_divsc3(a: f32, b: f32, c: f32, d: f32) -> core::num::Complex<f32> {
        complex_div(a, b, c, d)
    }

    pub extern "C" fn __rust_divdc3(a: f64, b: f64, c: f64, d: f64) -> core::num::Complex<f64> {
        complex_div(a, b, c, d)
    }

    #[cfg(f128_enabled)]
    pub extern "C" fn __rust_divtc3(a: f128, b: f128, c: f128, d: f128) -> core::num::Complex<f128> {
        complex_div(a, b, c, d)
    }
}
