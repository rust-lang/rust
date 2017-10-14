
use std::iter::Enumerate;
use std::mem::size_of;

pub fn third<A, B, C>(t: (A, B, C)) -> C { t.2 }

pub fn enumerate<I>(iterable: I) -> Enumerate<I::IntoIter>
    where I: IntoIterator
{
    iterable.into_iter().enumerate()
}

/// return the number of steps from a to b
pub fn ptrdistance<T>(a: *const T, b: *const T) -> usize {
    debug_assert!(a as usize <= b as usize);
    (b as usize - a as usize) / size_of::<T>()
}
