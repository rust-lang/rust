// Make sure we properly copy the `#[expect]` attr to the derived code and that no
// unfulfilled expectations are trigerred.
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
