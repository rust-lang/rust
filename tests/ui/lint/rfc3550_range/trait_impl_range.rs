#![crate_type="lib"]

#![deny(trait_impl_range)]
#![allow(dead_code)]

use std::ops::{Range, RangeInclusive, RangeFrom};

// Private trait, no error.
trait Private {}
impl Private for Range<usize> {}

// Public trait.
pub trait Foo {}
impl Foo for Range<usize> {} //~ ERROR public trait impl involving `Range` type
impl Foo for core::ops::RangeInclusive<usize> {} //~ ERROR public trait impl involving `RangeInclusive` type
impl Foo for [Range<usize>; 2] {} //~ ERROR public trait impl involving `Range` type
impl Foo for &[Range<usize>] {} //~ ERROR public trait impl involving `Range` type
impl Foo for *const Range<usize> {} //~ ERROR public trait impl involving `Range` type
impl Foo for &Range<usize> {} //~ ERROR public trait impl involving `Range` type
impl Foo for (u8, Range<usize>) {} //~ ERROR public trait impl involving `Range` type
impl Foo for std::mem::MaybeUninit<Range<usize>> {} //~ ERROR public trait impl involving `Range` type
impl<T> Foo for RangeFrom<T> {} //~ ERROR public trait impl involving `RangeFrom` type

pub struct Thing;
impl std::ops::Index<Range<usize>> for Thing { //~ ERROR public trait impl involving `Range` type
    type Output = ();
    fn index(&self, _: Range<usize>) -> &Self::Output {
        &()
    }
}

// Private adt, no error.
struct Wrapper<T>(T);
impl Foo for Wrapper<Range<usize>> {}

struct PrivateThing;
impl std::ops::Index<Range<usize>> for PrivateThing {
    type Output = ();
    fn index(&self, _: Range<usize>) -> &Self::Output {
        &()
    }
}
