#![feature(impl_trait_in_assoc_type)]

//@ check-pass

trait IterBits {
    type BitsIter: Iterator<Item = u8>;
    fn iter_bits(self, n: u8) -> Self::BitsIter;
}

impl<T: Copy, E> IterBits for T
where
    T: std::ops::Shr<Output = T>
        + std::ops::BitAnd<T, Output = T>
        + std::convert::From<u8>
        + std::convert::TryInto<u8, Error = E>,
    E: std::fmt::Debug,
{
    type BitsIter = impl std::iter::Iterator<Item = u8>;
    fn iter_bits(self, n: u8) -> Self::BitsIter {
        (0u8..n).rev().map(move |shift| ((self >> T::from(shift)) & T::from(1)).try_into().unwrap())
    }
}

fn main() {}
