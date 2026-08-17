use core::num::Complex;

use crate::support::Float;

/// Returns the product of `a + ib` and `c + id`.
///
/// The standard formula is `(ac - bd) + (ad + bc)i`, but this function has custom behavior when
/// both the real and imaginary components of that expression are NaN.
fn complex_mul<F: Float>(mut a: F, mut b: F, mut c: F, mut d: F) -> Complex<F> {
    let ac = a * c;
    let bd = b * d;
    let ad = a * d;
    let bc = b * c;

    let z = Complex::new(ac - bd, ad + bc);

    // The fast path: exit when at least one component is not NaN.
    if !(z.re.is_nan() && z.im.is_nan()) {
        return z;
    }

    let zero_if_nan = |x: F| if x.is_nan() { F::ZERO.copysign(x) } else { x };

    let signed_unit_if_inf = |x: F| {
        let mag = if x.is_infinite() { F::ONE } else { F::ZERO };
        mag.copysign(x)
    };

    let mut recalc = false;

    if a.is_infinite() || b.is_infinite() {
        a = signed_unit_if_inf(a);
        b = signed_unit_if_inf(b);
        c = zero_if_nan(c);
        d = zero_if_nan(d);
        recalc = true;
    }

    if c.is_infinite() || d.is_infinite() {
        c = signed_unit_if_inf(c);
        d = signed_unit_if_inf(d);
        a = zero_if_nan(a);
        b = zero_if_nan(b);
        recalc = true;
    }

    if !recalc && (ac.is_infinite() || bd.is_infinite() || ad.is_infinite() || bc.is_infinite()) {
        a = zero_if_nan(a);
        b = zero_if_nan(b);
        c = zero_if_nan(c);
        d = zero_if_nan(d);
        recalc = true;
    }

    if !recalc {
        return z;
    }

    Complex::new(F::INFINITY * (a * c - b * d), F::INFINITY * (a * d + b * c))
}

intrinsics! {
    #[cfg(all(f16_enabled, not(x86_no_sse2)))]
    pub extern "C" fn __rust_mulhc3(a: f16, b: f16, c: f16, d: f16) -> core::num::Complex<f16> {
        complex_mul(a, b, c, d)
    }

    pub extern "C" fn __rust_mulsc3(a: f32, b: f32, c: f32, d: f32) -> core::num::Complex<f32> {
        complex_mul(a, b, c, d)
    }

    pub extern "C" fn __rust_muldc3(a: f64, b: f64, c: f64, d: f64) -> core::num::Complex<f64> {
        complex_mul(a, b, c, d)
    }

    #[cfg(f128_enabled)]
    pub extern "C" fn __rust_multc3(a: f128, b: f128, c: f128, d: f128) -> core::num::Complex<f128> {
        complex_mul(a, b, c, d)
    }
}
