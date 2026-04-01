use crate::clone::Clone;
use crate::cmp::PartialEq;
use crate::fmt::Debug;
use crate::marker::Copy;
use crate::ops::{Add, Sub};

/// The complex number type in Cartesian form.
/// Supports simple addition and subtraction.
/// ```
/// let x = Complex::new(1.0, 2.0);
/// let y = Complex::new(3.0, 4.0);
/// assert_eq!(x + y, Complex::new(4.0, 6.0);
/// assert_eq!(y - x, Complex::new(2.0, 2.0);
/// ```
#[derive(Debug, PartialEq)]
#[unstable(feature = "complex_numbers", issue = "154023")]
#[repr(C)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T> Complex<T> {
    /// Constructs a new instance of type `Complex`.
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Complex<T::Output>;

    fn add(self, other: Self) -> Self::Output {
        Self::Output { re: self.re + other.re, im: self.im + other.im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Add<Output = T>> Add<T> for Complex<T> {
    type Output = Complex<T::Output>;

    fn add(self, other: T) -> Self::Output {
        Self::Output { re: self.re + other, im: self.im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Complex<T::Output>;

    fn sub(self, other: Self) -> Self::Output {
        Self::Output { re: self.re - other.re, im: self.im - other.im }
    }
}

#[unstable(feature = "complex_numbers", issue = "154023")]
impl<T: Sub<Output = T>> Sub<T> for Complex<T> {
    type Output = Complex<T::Output>;

    fn sub(self, other: T) -> Self::Output {
        Self::Output { re: self.re - other, im: self.im }
    }
}
