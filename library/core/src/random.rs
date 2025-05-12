//! Random value generation.
//!
//! The [`Random`] trait allows generating a random value for a type using a
//! given [`RandomSource`].

/// A source of randomness.
#[unstable(feature = "random", issue = "130703")]
pub trait RandomSource {
    /// Fills `bytes` with random bytes.
    fn fill_bytes(&mut self, bytes: &mut [u8]);
}

/// A trait for getting a random value for a type.
///
/// **Warning:** Be careful when manipulating random values! The
/// [`random`](Random::random) method on integers samples them with a uniform
/// distribution, so a value of 1 is just as likely as [`i32::MAX`]. By using
/// modulo operations, some of the resulting values can become more likely than
/// others. Use audited crates when in doubt.
#[unstable(feature = "random", issue = "130703")]
pub trait Random: Sized {
    /// Generates a random value.
    fn random(source: &mut (impl RandomSource + ?Sized)) -> Self;
}

impl Random for bool {
    fn random(source: &mut (impl RandomSource + ?Sized)) -> Self {
        u8::random(source) & 1 == 1
    }
}

macro_rules! impl_primitive {
    ($t:ty) => {
        impl Random for $t {
            /// Generates a random value.
            ///
            /// **Warning:** Be careful when manipulating the resulting value! This
            /// method samples according to a uniform distribution, so a value of 1 is
            /// just as likely as [`MAX`](Self::MAX). By using modulo operations, some
            /// values can become more likely than others. Use audited crates when in
            /// doubt.
            fn random(source: &mut (impl RandomSource + ?Sized)) -> Self {
                let mut bytes = (0 as Self).to_ne_bytes();
                source.fill_bytes(&mut bytes);
                Self::from_ne_bytes(bytes)
            }
        }
    };
}

impl_primitive!(u8);
impl_primitive!(i8);
impl_primitive!(u16);
impl_primitive!(i16);
impl_primitive!(u32);
impl_primitive!(i32);
impl_primitive!(u64);
impl_primitive!(i64);
impl_primitive!(u128);
impl_primitive!(i128);
impl_primitive!(usize);
impl_primitive!(isize);
