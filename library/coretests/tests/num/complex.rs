use core::num::{Complex, Wrapping};

#[test]
fn complex_default() {
    assert_eq!(Complex::<i32>::default(), Complex::new(0, 0));
    assert_eq!(Complex::<f32>::default(), Complex::new(0.0, 0.0));

    // The default is the additive unit.
    let a = Complex::new(1, 2);
    assert_eq!(a + Complex::<i32>::default(), a);
    assert_eq!(Complex::<i32>::default() + a, a);
}

#[test]
fn complex_addition() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    assert_eq!(a + 8, Complex::new(a.re + 8, a.im));

    let a = Complex::new(Wrapping(1u8), Wrapping(2));
    let b = Complex::new(Wrapping(3u8), Wrapping(4));
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    let c = a + Wrapping(u8::MAX);
    assert_eq!(c, Complex::new(a.re + Wrapping(u8::MAX), a.im));
    assert_eq!(c.re.0, 1u8.wrapping_add(u8::MAX));

    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    assert_eq!(a + 8.0, Complex::new(a.re + 8.0, a.im));
}

#[test]
fn complex_subtraction() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    assert_eq!(a - 8, Complex::new(a.re - 8, a.im));

    let a = Complex::new(Wrapping(1u8), Wrapping(2));
    let b = Complex::new(Wrapping(3u8), Wrapping(4));
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    let c = a - Wrapping(u8::MAX);
    assert_eq!(c, Complex::new(a.re - Wrapping(u8::MAX), a.im));
    assert_eq!(c.re.0, 1u8.wrapping_sub(u8::MAX));

    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    assert_eq!(a - 8.0, Complex::new(a.re - 8.0, a.im));
}

#[test]
fn complex_conjugate() {
    assert_eq!(Complex::new(1, 2).conjugate(), Complex::new(1, -2));
    assert_eq!(Complex::new(1, -2).conjugate(), Complex::new(1, 2));

    assert_eq!(Complex::new(1.0, 2.0).conjugate(), Complex::new(1.0, -2.0));
    assert_eq!(Complex::new(1.0, -2.0).conjugate(), Complex::new(1.0, 2.0));
    assert_eq!(Complex::new(1.0, f32::INFINITY).conjugate(), Complex::new(1.0, f32::NEG_INFINITY));
}

#[test]
fn complex_negation() {
    assert_eq!(-Complex::new(1, 2), Complex::new(-1, -2));
    assert_eq!(-Complex::new(1, -2), Complex::new(-1, 2));

    assert_eq!(-Complex::new(1.0, 2.0), Complex::new(-1.0, -2.0));
    assert_eq!(-Complex::new(1.0, -2.0), Complex::new(-1.0, 2.0));
    assert_eq!(-Complex::new(1.0, f32::INFINITY), Complex::new(-1.0, f32::NEG_INFINITY),);
}

#[test]
fn complex_multiplication() {
    #[cfg(target_has_reliable_f16)]
    assert_eq!(Complex::new(1.0f16, 2.0) * Complex::new(3.0, 4.0), Complex::new(-5.0, 10.0));
    assert_eq!(Complex::new(1.0f32, 2.0) * Complex::new(3.0, 4.0), Complex::new(-5.0, 10.0));
    assert_eq!(Complex::new(1.0f64, 2.0) * Complex::new(3.0, 4.0), Complex::new(-5.0, 10.0));
    #[cfg(target_has_reliable_f128)]
    assert_eq!(Complex::new(1.0f128, 2.0) * Complex::new(3.0, 4.0), Complex::new(-5.0, 10.0));

    // The naive algorithm would return NaN + NaNi for these inputs, but the libcall handles it.
    #[cfg(target_has_reliable_f16)]
    assert_eq!(
        Complex::new(1.0, 0.0) * Complex::new(f16::INFINITY, f16::INFINITY),
        Complex::new(f16::INFINITY, f16::INFINITY)
    );
    assert_eq!(
        Complex::new(1.0, 0.0) * Complex::new(f32::INFINITY, f32::INFINITY),
        Complex::new(f32::INFINITY, f32::INFINITY)
    );
    assert_eq!(
        Complex::new(1.0, 0.0) * Complex::new(f64::INFINITY, f64::INFINITY),
        Complex::new(f64::INFINITY, f64::INFINITY)
    );
    #[cfg(target_has_reliable_f128)]
    assert_eq!(
        Complex::new(1.0, 0.0) * Complex::new(f128::INFINITY, f128::INFINITY),
        Complex::new(f128::INFINITY, f128::INFINITY)
    );
}

#[test]
fn div() {
    #[cfg(target_has_reliable_f16)]
    assert_eq!(Complex::new(2.0f16, 11.0) / Complex::new(2.0, 1.0), Complex::new(3.0, 4.0));
    assert_eq!(Complex::new(2.0f32, 11.0) / Complex::new(2.0, 1.0), Complex::new(3.0, 4.0));
    assert_eq!(Complex::new(2.0f64, 11.0) / Complex::new(2.0, 1.0), Complex::new(3.0, 4.0));
    #[cfg(target_has_reliable_f128)]
    assert_eq!(Complex::new(2.0f128, 11.0) / Complex::new(2.0, 1.0), Complex::new(3.0, 4.0));

    // The naive algorithm would return NaN + NaNi for these inputs, but the libcall handles it.
    #[cfg(target_has_reliable_f16)]
    assert_eq!(
        Complex::new(f16::INFINITY, 0.0) / Complex::new(1.0, 1.0),
        Complex::new(f16::INFINITY, f16::NEG_INFINITY)
    );
    assert_eq!(
        Complex::new(f32::INFINITY, 0.0) / Complex::new(1.0, 1.0),
        Complex::new(f32::INFINITY, f32::NEG_INFINITY)
    );
    assert_eq!(
        Complex::new(f64::INFINITY, 0.0) / Complex::new(1.0, 1.0),
        Complex::new(f64::INFINITY, f64::NEG_INFINITY)
    );
    #[cfg(target_has_reliable_f128)]
    assert_eq!(
        Complex::new(f128::INFINITY, 0.0) / Complex::new(1.0, 1.0),
        Complex::new(f128::INFINITY, f128::NEG_INFINITY)
    );
}
