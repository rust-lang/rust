use crate::ops::{Add, Neg, Sub};

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
