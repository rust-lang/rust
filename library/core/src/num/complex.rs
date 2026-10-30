use crate::num::imp::libm::complex::*;
use crate::ops::{Add, Div, Mul, Neg, Sub};

/// A [complex number].
///
/// # Basic operations
///
/// A `Complex<T>` supports addition, subtraction, and negation when `T` does. The conjugation
/// requires that `T` supports negation.
///
/// ```
/// #![feature(complex_numbers)]
///
/// use std::num::Complex;
///
/// let x = Complex { re: 5.0, im: 9.0 };
/// let y = Complex::new(2.0, 3.0);
///
/// // Scalar addition and subtraction operate on the real component.
/// assert_eq!(x + 1.0, Complex::new(6.0, 9.0));
/// assert_eq!(x - 1.0, Complex::new(4.0, 9.0));
///
/// // Complex addition and subtraction operate pointwise.
/// assert_eq!(x + y, Complex::new(7.0, 12.0));
/// assert_eq!(x - y, Complex::new(3.0, 6.0));
///
/// // Negation occurs on both components.
/// // Geometrically this is a rotation by 180 degrees.
/// assert_eq!(-x, Complex::new(-5.0, -9.0));
///
/// // The conjugate negates only the imaginary component.
/// // Geometrically this is a reflection across the real axis.
/// assert_eq!(x.conjugate(), Complex::new(5.0, -9.0));
/// ```
///
/// # Complex multiplication and division of floating points
///
/// Multiplication and division follow the textbook formulas, except that they attempt to
/// correct for overflow of intermediate steps of the calculation.
///
/// This behavior mirrors the C standard, although it does not specify exactly when the
/// output should be corrected. Hence, the exact output may vary from platform to platform.
///
/// ```
/// #![feature(complex_numbers)]
///
/// use std::num::Complex;
///
/// assert_eq!(Complex::new(1.0, 2.0) * Complex::new(3.0, 4.0), Complex::new(-5.0, 10.0));
/// assert_eq!(Complex::new(2.0, 11.0) / Complex::new(2.0, 1.0), Complex::new(3.0, 4.0));
///
/// // The textbook algorithm would return NaN + NaNi for these inputs.
/// assert_eq!(
///     Complex::new(1.0, 0.0) * Complex::new(f64::INFINITY, f64::INFINITY),
///     Complex::new(f64::INFINITY, f64::INFINITY)
/// );
/// assert_eq!(
///     Complex::new(1.0, 1.0) / Complex::new(f64::INFINITY, 0.0),
///     Complex::new(0.0, 0.0)
/// );
/// ```
///
/// # FFI compatibility
///
/// The following types are guaranteed to fully match the platform's `_Complex` in terms of layout and ABI:
///
/// - `Complex<`[`c_float`]`>`
/// - `Complex<`[`c_double`]`>`
///
/// For any other `Complex<T>` ABI compatibility is not guaranteed.
///
/// [complex number]: https://en.wikipedia.org/wiki/Complex_number
/// [`c_float`]: crate::ffi::c_float
/// [`c_double`]: crate::ffi::c_double
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
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(Complex::new(3.0, 2.0).conjugate(), Complex::new(3.0, -2.0));
    /// assert_eq!(Complex::new(-3.0, -2.0).conjugate(), Complex::new(-3.0, 2.0));
    /// ```
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
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(-Complex::new(3.0, 2.0), Complex::new(-3.0, -2.0));
    /// ```
    fn neg(self) -> Self::Output {
        Complex::new(-self.re, -self.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add> Add<Self> for Complex<T> {
    type Output = Complex<T::Output>;

    /// Complex addition operates pointwise.
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(Complex::new(2.0, 3.0) + Complex::new(5.0, 9.0), Complex::new(7.0, 12.0));
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add<Output = T>> Add<T> for Complex<T> {
    type Output = Complex<T::Output>;

    /// Scalar addition operates only on the real component.
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(Complex::new(2.0, 3.0) + 10.0, Complex::new(12.0, 3.0));
    /// ```
    fn add(self, rhs: T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub> Sub<Self> for Complex<T> {
    type Output = Complex<T::Output>;

    /// Complex subtraction operates pointwise.
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(Complex::new(2.0, 3.0) - Complex::new(5.0, 9.0), Complex::new(-3.0, -6.0));
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub<Output = T>> Sub<T> for Complex<T> {
    type Output = Complex<T::Output>;

    /// Scalar subtraction operates only on the real component.
    ///
    /// ```
    /// #![feature(complex_numbers)]
    ///
    /// use std::num::Complex;
    ///
    /// assert_eq!(Complex::new(2.0, 3.0) - 10.0, Complex::new(-8.0, 3.0));
    /// ```
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
