// Make sure the `#[expect]` attr on an item is shared with the code derived from it:
// the lint is suppressed there and fulfills the expectation.
//
// See <https://github.com/rust-lang/rust/issues/150553> for rational.

//@ check-pass

#![deny(redundant_lifetimes)]

use std::fmt::Debug;

#[derive(Debug)]
#[expect(redundant_lifetimes)]
pub struct RefWrapper<'a, T>
where
    'a: 'static,
    T: Debug,
{
    pub t_ref: &'a T,
}

fn main() {}
