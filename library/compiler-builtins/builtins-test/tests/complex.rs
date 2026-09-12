#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]
#![feature(complex_numbers)]
#![allow(unused_features)]

mod complex {
    use core::num::Complex;

    use compiler_builtins::support::Float;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Class {
        /// Both components are NaN.
        NaN,
        /// At least one component is infinite.
        Infinite,
        /// Both components are zero.
        Zero,
        /// One component is a "regular" number, the other is NaN.
        NonZeroAndNaN,
        /// Both components are "regular" numbers.
        NonZero,
    }

    fn classify<F: Float>(c: Complex<F>) -> Class {
        if c.re == F::ZERO && c.im == F::ZERO {
            Class::Zero
        } else if c.re.is_infinite() || c.im.is_infinite() {
            Class::Infinite
        } else if c.re.is_nan() && c.im.is_nan() {
            Class::NaN
        } else if c.re.is_nan() {
            if c.im == F::ZERO {
                Class::NaN
            } else {
                Class::NonZeroAndNaN
            }
        } else if c.im.is_nan() {
            if c.re == F::ZERO {
                Class::NaN
            } else {
                Class::NonZeroAndNaN
            }
        } else {
            Class::NonZero
        }
    }

    fn test_mul<F: Float>(p: Complex<F>, q: Complex<F>, actual: Complex<F>, tolerance: F) -> bool {
        let expected = match classify(p) {
            Class::Zero => match classify(q) {
                Class::Zero | Class::NonZero => Class::Zero,
                Class::Infinite | Class::NaN | Class::NonZeroAndNaN => Class::NaN,
            },

            Class::NonZero => match classify(q) {
                Class::Zero => Class::Zero,
                Class::NonZero => {
                    if classify(actual) != Class::NonZero {
                        return true;
                    }

                    let Complex { re: a, im: b } = p;
                    let Complex { re: c, im: d } = q;

                    let z = Complex::new(a * c - b * d, a * d + b * c);
                    let r = actual;

                    let diff_re = r.re - z.re;
                    let diff_im = r.im - z.im;

                    let diff_sq = diff_re * diff_re + diff_im * diff_im;
                    let mag_sq = r.re * r.re + r.im * r.im;

                    if diff_sq > (tolerance * tolerance) * mag_sq {
                        return true;
                    }

                    return false;
                }
                Class::Infinite => Class::Infinite,
                Class::NaN | Class::NonZeroAndNaN => Class::NaN,
            },

            Class::Infinite => match classify(q) {
                Class::Zero | Class::NaN => Class::NaN,
                Class::NonZero | Class::Infinite | Class::NonZeroAndNaN => Class::Infinite,
            },

            Class::NaN => Class::NaN,

            Class::NonZeroAndNaN => match classify(q) {
                Class::Infinite => Class::Infinite,
                Class::Zero | Class::NonZero | Class::NaN | Class::NonZeroAndNaN => Class::NaN,
            },
        };

        classify(actual) != expected
    }

    fn test_div<F: Float>(
        dividend: Complex<F>,
        divisor: Complex<F>,
        actual: Complex<F>,
        tolerance: F,
    ) -> bool {
        let expected = match classify(dividend) {
            Class::Zero => match classify(divisor) {
                Class::Zero => Class::NaN,
                Class::NonZero => Class::Zero,
                Class::Infinite => Class::Zero,
                Class::NaN => Class::NaN,
                Class::NonZeroAndNaN => Class::NaN,
            },

            Class::NonZero => match classify(divisor) {
                Class::Zero => Class::Infinite,
                Class::NonZero => {
                    if classify(actual) != Class::NonZero {
                        return true;
                    }

                    let Complex { re: a, im: b } = dividend;
                    let Complex { re: c, im: d } = divisor;

                    let denominator = c * c + d * d;
                    let z = Complex::new(
                        (a * c + b * d) / denominator, //
                        (b * c - a * d) / denominator,
                    );

                    let r = actual;

                    let diff_re = r.re - z.re;
                    let diff_im = r.im - z.im;

                    let diff_sq = diff_re * diff_re + diff_im * diff_im;
                    let mag_sq = r.re * r.re + r.im * r.im;

                    if diff_sq > (tolerance * tolerance) * mag_sq {
                        return true;
                    }

                    return false;
                }
                Class::Infinite => Class::Zero,
                Class::NaN => Class::NaN,
                Class::NonZeroAndNaN => Class::NaN,
            },

            Class::Infinite => match classify(divisor) {
                Class::Zero | Class::NonZero => Class::Infinite,
                Class::Infinite | Class::NaN | Class::NonZeroAndNaN => Class::NaN,
            },

            Class::NaN => Class::NaN,

            Class::NonZeroAndNaN => match classify(divisor) {
                Class::Zero => Class::Infinite,
                Class::NonZero | Class::Infinite | Class::NaN | Class::NonZeroAndNaN => Class::NaN,
            },
        };

        classify(actual) != expected
    }

    macro_rules! complex_test_data {
        ($f:ty) => {{
            const INFINITY: $f = <$f>::INFINITY;
            const NEG_INFINITY: $f = <$f>::NEG_INFINITY;
            const NAN: $f = <$f>::NAN;
            const SNAN: $f = <$f>::SNAN;

            #[allow(overflowing_literals)]
            let (small, big) = if size_of::<$f>() == 2 {
                (1.0e-2, 1.0e2)
            } else {
                (1.0e-6, 1.0e6)
            };

            [
                Complex::new(small, small),
                Complex::new(-small, small),
                Complex::new(-small, -small),
                Complex::new(small, -small),
                Complex::new(big, small),
                Complex::new(-big, small),
                Complex::new(-big, -small),
                Complex::new(big, -small),
                Complex::new(small, big),
                Complex::new(-small, big),
                Complex::new(-small, -big),
                Complex::new(small, -big),
                Complex::new(big, big),
                Complex::new(-big, big),
                Complex::new(-big, -big),
                Complex::new(big, -big),
                Complex::new(NAN, NAN),
                Complex::new(NEG_INFINITY, NAN),
                Complex::new(-2., NAN),
                Complex::new(-1., NAN),
                Complex::new(-0.5, NAN),
                Complex::new(-0., NAN),
                Complex::new(0., NAN),
                Complex::new(0.5, NAN),
                Complex::new(1., NAN),
                Complex::new(2., NAN),
                Complex::new(INFINITY, NAN),
                Complex::new(NAN, NEG_INFINITY),
                Complex::new(NEG_INFINITY, NEG_INFINITY),
                Complex::new(-2., NEG_INFINITY),
                Complex::new(-1., NEG_INFINITY),
                Complex::new(-0.5, NEG_INFINITY),
                Complex::new(-0., NEG_INFINITY),
                Complex::new(0., NEG_INFINITY),
                Complex::new(0.5, NEG_INFINITY),
                Complex::new(1., NEG_INFINITY),
                Complex::new(2., NEG_INFINITY),
                Complex::new(INFINITY, NEG_INFINITY),
                Complex::new(NAN, -2.),
                Complex::new(NEG_INFINITY, -2.),
                Complex::new(-2., -2.),
                Complex::new(-1., -2.),
                Complex::new(-0.5, -2.),
                Complex::new(-0., -2.),
                Complex::new(0., -2.),
                Complex::new(0.5, -2.),
                Complex::new(1., -2.),
                Complex::new(2., -2.),
                Complex::new(INFINITY, -2.),
                Complex::new(NAN, -1.),
                Complex::new(NEG_INFINITY, -1.),
                Complex::new(-2., -1.),
                Complex::new(-1., -1.),
                Complex::new(-0.5, -1.),
                Complex::new(-0., -1.),
                Complex::new(0., -1.),
                Complex::new(0.5, -1.),
                Complex::new(1., -1.),
                Complex::new(2., -1.),
                Complex::new(INFINITY, -1.),
                Complex::new(NAN, -0.5),
                Complex::new(NEG_INFINITY, -0.5),
                Complex::new(-2., -0.5),
                Complex::new(-1., -0.5),
                Complex::new(-0.5, -0.5),
                Complex::new(-0., -0.5),
                Complex::new(0., -0.5),
                Complex::new(0.5, -0.5),
                Complex::new(1., -0.5),
                Complex::new(2., -0.5),
                Complex::new(INFINITY, -0.5),
                Complex::new(NAN, -0.),
                Complex::new(NEG_INFINITY, -0.),
                Complex::new(-2., -0.),
                Complex::new(-1., -0.),
                Complex::new(-0.5, -0.),
                Complex::new(-0., -0.),
                Complex::new(0., -0.),
                Complex::new(0.5, -0.),
                Complex::new(1., -0.),
                Complex::new(2., -0.),
                Complex::new(INFINITY, -0.),
                Complex::new(NAN, 0.),
                Complex::new(NEG_INFINITY, 0.),
                Complex::new(-2., 0.),
                Complex::new(-1., 0.),
                Complex::new(-0.5, 0.),
                Complex::new(-0., 0.),
                Complex::new(0., 0.),
                Complex::new(0.5, 0.),
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(INFINITY, 0.),
                Complex::new(NAN, 0.5),
                Complex::new(NEG_INFINITY, 0.5),
                Complex::new(-2., 0.5),
                Complex::new(-1., 0.5),
                Complex::new(-0.5, 0.5),
                Complex::new(-0., 0.5),
                Complex::new(0., 0.5),
                Complex::new(0.5, 0.5),
                Complex::new(1., 0.5),
                Complex::new(2., 0.5),
                Complex::new(INFINITY, 0.5),
                Complex::new(NAN, 1.),
                Complex::new(NEG_INFINITY, 1.),
                Complex::new(-2., 1.),
                Complex::new(-1., 1.),
                Complex::new(-0.5, 1.),
                Complex::new(-0., 1.),
                Complex::new(0., 1.),
                Complex::new(0.5, 1.),
                Complex::new(1., 1.),
                Complex::new(2., 1.),
                Complex::new(INFINITY, 1.),
                Complex::new(NAN, 2.),
                Complex::new(NEG_INFINITY, 2.),
                Complex::new(-2., 2.),
                Complex::new(-1., 2.),
                Complex::new(-0.5, 2.),
                Complex::new(-0., 2.),
                Complex::new(0., 2.),
                Complex::new(0.5, 2.),
                Complex::new(1., 2.),
                Complex::new(2., 2.),
                Complex::new(INFINITY, 2.),
                Complex::new(NAN, INFINITY),
                Complex::new(NEG_INFINITY, INFINITY),
                Complex::new(-2., INFINITY),
                Complex::new(-1., INFINITY),
                Complex::new(-0.5, INFINITY),
                Complex::new(-0., INFINITY),
                Complex::new(0., INFINITY),
                Complex::new(0.5, INFINITY),
                Complex::new(1., INFINITY),
                Complex::new(2., INFINITY),
                Complex::new(INFINITY, INFINITY),
                Complex::new(INFINITY, SNAN),
            ]
        }};
    }

    macro_rules! complex_mul {
        ($($f:ty, $fn:ident, $tolerance:literal);*;) => {

            $(
                #[test]
                fn $fn() {
                    use compiler_builtins::float::complex::mul::$fn;

                    let input = complex_test_data!($f);

                    for p in input {
                        for q in input {
                            let Complex{ re: a, im: b } = p;
                            let Complex{ re: c, im: d } = q;

                            let actual = $fn(a, b, c, d);

                            assert!(
                                !test_mul(p, q, actual, $tolerance),
                                "{func}({a:?}, {b:?}, {c:?}, {d:?}): incorrect ({:?}, {:?})",
                                actual.re,
                                actual.im,
                                func = stringify!($fn),
                            );
                        }
                    }
                }
            )*
        };
    }

    macro_rules! complex_div {
        ($($f:ty, $fn:ident, $tolerance:literal);*;) => {
            $(
                #[test]
                fn $fn() {
                    use compiler_builtins::float::complex::div::$fn;

                    let input = complex_test_data!($f);

                    for p in input {
                        for q in input {
                            let Complex{ re: a, im: b } = p;
                            let Complex{ re: c, im: d } = q;

                            let actual = $fn(a, b, c, d);

                            assert!(
                                !test_div(p, q, actual, $tolerance),
                                "{func}({a:?}, {b:?}, {c:?}, {d:?}): incorrect ({:?}, {:?})",
                                actual.re,
                                actual.im,
                                func = stringify!($fn),
                            );
                        }
                    }
                }
            )*
        };
    }

    #[cfg(all(f16_enabled, not(x86_no_sse2)))]
    complex_mul! {
        f16, __rust_mulhc3, 1.0e-3;
    }

    #[cfg(all(f16_enabled, not(x86_no_sse2)))]
    complex_div! {
        f16, __rust_divhc3, 1.0e-3;
    }

    complex_mul! {
        f32, __rust_mulsc3, 1.0e-6;
        f64, __rust_muldc3, 1.0e-9;
    }

    complex_div! {
        f32, __rust_divsc3, 1.0e-6;
        f64, __rust_divdc3, 1.0e-9;
    }

    #[cfg(f128_enabled)]
    cfg_select! {
        any(target_arch = "powerpc", target_arch = "powerpc64") => {
            complex_mul! {
                f128, __rust_mulkc3, 1.0e-12;
            }

            complex_div! {
                f128, __rust_divkc3, 1.0e-12;
            }
        }
        _ => {
            complex_mul! {
                f128, __rust_multc3, 1.0e-12;
            }

            complex_div! {
                f128, __rust_divtc3, 1.0e-12;
            }
        }
    }
}
