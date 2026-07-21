//! Random value generation.

use crate::ops::RangeFull;

/// A source of randomness.
#[unstable(feature = "random", issue = "130703")]
pub trait Rng {
    /// Fills `bytes` with random bytes.
    ///
    /// Note that calling `fill_bytes` multiple times is not equivalent to calling `fill_bytes` once
    /// with a larger buffer. An `Rng` is allowed to return different bytes for those two cases. For
    /// instance, this allows an `Rng` to generate a word at a time and throw part of it away if not
    /// needed.
    fn fill_bytes(&mut self, bytes: &mut [u8]);
}

/// Implements `Rng` for mutable references to random number generators by
/// forwarding all methods to the referenced generator.
#[unstable(feature = "random", issue = "130703")]
impl<'a, R: Rng + ?Sized> Rng for &'a mut R {
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        R::fill_bytes(self, bytes);
    }
}

/// A trait representing a distribution of random values for a type.
#[unstable(feature = "random", issue = "130703")]
pub trait Distribution<T> {
    /// Samples a random value from the distribution, using the specified random source.
    fn sample(&self, source: &mut (impl Rng + ?Sized)) -> T;
}

impl<T, DT: Distribution<T>> Distribution<T> for &DT {
    fn sample(&self, source: &mut (impl Rng + ?Sized)) -> T {
        (*self).sample(source)
    }
}

impl Distribution<bool> for RangeFull {
    fn sample(&self, source: &mut (impl Rng + ?Sized)) -> bool {
        let byte: u8 = RangeFull.sample(source);
        byte & 1 == 1
    }
}

macro_rules! impl_primitive {
    ($t:ty) => {
        impl Distribution<$t> for RangeFull {
            fn sample(&self, source: &mut (impl Rng + ?Sized)) -> $t {
                let mut bytes = (0 as $t).to_ne_bytes();
                source.fill_bytes(&mut bytes);
                <$t>::from_ne_bytes(bytes)
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
