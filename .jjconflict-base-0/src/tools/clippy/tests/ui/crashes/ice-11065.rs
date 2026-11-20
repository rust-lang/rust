//@ check-pass

#![warn(clippy::useless_conversion)]

use std::option::IntoIter as OptionIter;

fn eq<T: Eq>(a: T, b: T) -> bool {
    a == b
}

macro_rules! tests {
    ($($expr:expr, $ty:ty, ($($test:expr),*);)+) => (pub fn main() {$({
        const C: $ty = $expr;
        assert!(eq(C($($test),*), $expr($($test),*)));
    })+})
}

tests! {
    FromIterator::from_iter, fn(OptionIter<i32>) -> Vec<i32>, (Some(5).into_iter());
}
