// known-bug: #89515
//
// The trait solver cannot deal with ambiguous marker trait impls
// if there are lifetimes involved. As we must not special-case any
// regions this does not work, even with 'static
#![feature(marker_trait_attr)]

#[marker]
trait Marker {}

impl Marker for &'static () {}
impl Marker for &'static () {}

fn main() {}
