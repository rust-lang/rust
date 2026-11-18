// Make sure `#[expect]` on an item with a derive is fulfilled when the lint is only emitted on the
// item itself, and not also on the derived impl, which reuses the lifetime spans of the input.
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
