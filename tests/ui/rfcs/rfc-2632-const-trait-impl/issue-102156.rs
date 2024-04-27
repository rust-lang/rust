#![feature(allocator_api)]
#![feature(const_trait_impl)]

use core::convert::{From, TryFrom};
//~^ ERROR
//~| ERROR

use std::pin::Pin;
use std::alloc::Allocator;
impl<T: ?Sized, A: Allocator> const From<Box<T, A>> for Pin<Box<T, A>>
where
    A: 'static,
{}

pub fn main() {}
