/// A complex number.
#[derive(Clone, Copy, Debug, PartialEq)]
#[unstable(feature = "complex_numbers", issue = "154023")]
#[repr(C)]
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
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }
}
