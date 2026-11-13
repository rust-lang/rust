use crate::num::imp::libm::complex::*;
use crate::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[unstable(feature = "complex_numbers", issue = "154023")]
#[repr(C)]
#[lang = "complex"]
pub struct Complex<T> {
    /// The real component.
    pub re: T,
    /// The imaginary component.
    pub im: T,
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T> Complex<T> {
    /// Create a new complex number from a real and imaginary component.
    #[must_use]
    pub const fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Self { re: Default::default(), im: Default::default() }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T> Complex<T>
where
    T: Neg<Output = T>,
{
    /// The complex conjugate of a complex number.
    ///
    /// The conjugate of `a + bi` is `a - bi`: the imaginary component is negated.
    /// Geometrically, this is a reflection across the real axis.
    #[must_use]
    pub fn conjugate(self) -> Self {
        Complex { re: self.re, im: -self.im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Neg> Neg for Complex<T> {
    type Output = Complex<T::Output>;

    /// Negate a complex number.
    ///
    /// The negation of `a + bi` is `-a - bi`: both components are negated.
    /// Geometrically this is a rotation of 180°.
    fn neg(self) -> Self::Output {
        Complex::new(-self.re, -self.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add> Add<Self> for Complex<T> {
    type Output = Complex<T::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add<Output = T>> Add<T> for Complex<T> {
    type Output = Complex<T::Output>;

    fn add(self, rhs: T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub> Sub<Self> for Complex<T> {
    type Output = Complex<T::Output>;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub<Output = T>> Sub<T> for Complex<T> {
    type Output = Complex<T::Output>;

    fn sub(self, rhs: T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

macro_rules! impl_complex_mul_div {
    ($ty:ty, $mul:ident, $div:ident) => {
        #[unstable(feature = "complex_numbers", issue = "154023")]
        impl Mul for Complex<$ty> {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                let Complex { re: a, im: b } = self;
                let Complex { re: c, im: d } = rhs;

                let ac = a * c;
                let bd = b * d;
                let ad = a * d;
                let bc = b * c;

                let z = Complex::new(ac - bd, ad + bc);

                // Only call the libcall when both components are NaN.
                //
                // The naive algorithm would return NaN + NaNi for an input like
                // (1 + 0i) * (inf + infi). The libcall instead returns inf + infi.
                //
                // We duplicate the fast path here so that it can be inlined. We use a libcall
                // for the NaN correction to reduce the size of `core`.
                if z.re.is_nan() && z.im.is_nan() {
                    crate::hint::cold_path();
                    $mul(a, b, c, d)
                } else {
                    z
                }
            }
        }

        #[unstable(feature = "complex_numbers", issue = "154023")]
        impl Div for Complex<$ty> {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                $div(self.re, self.im, rhs.re, rhs.im)
            }
        }
    };
}

impl_complex_mul_div!(f16, __rust_mulhc3, __rust_divhc3);
impl_complex_mul_div!(f32, __mulsc3, __divsc3);
impl_complex_mul_div!(f64, __muldc3, __divdc3);
impl_complex_mul_div!(f128, __rust_multc3, __rust_divtc3);
