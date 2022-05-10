#![feature(type_alias_impl_trait)]

trait IterBits {
    type BitsIter: Iterator<Item = u8>;
    fn iter_bits(self, n: u8) -> Self::BitsIter;
}

type IterBitsIter<T, E, I> = impl std::iter::Iterator<Item = I>;

impl<T: Copy, E> IterBits for T
where
    T: std::ops::Shr<Output = T>
        + std::ops::BitAnd<T, Output = T>
        + std::convert::From<u8>
        + std::convert::TryInto<u8, Error = E>,
    E: std::fmt::Debug,
{
    type BitsIter = IterBitsIter<T, E, u8>;
    fn iter_bits(self, n: u8) -> Self::BitsIter {
        (0u8..n).rev().map(move |shift| ((self >> T::from(shift)) & T::from(1)).try_into().unwrap())
        //~^ ERROR non-defining opaque type use in defining scope
        //~| ERROR type mismatch resolving
        //~| ERROR type mismatch resolving `<T as TryInto<u8>>::Error == E`
        //~| ERROR no implementation for `T >> T`
        //~| ERROR no implementation for `T & T`
        //~| ERROR the trait bound `T: From<u8>`
        //~| ERROR the trait bound `T: Copy` is not satisfied
        //~| ERROR `E` doesn't implement `Debug`
        //~| ERROR the trait bound `u8: From<T>` is not satisfied
    }
}

fn main() {}
