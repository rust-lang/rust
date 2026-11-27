//! Regression test for <https://github.com/rust-lang/rust/issues/58390>.
//!
//! An unknown `#![feature(..)]` name used to be silently ignored whenever the crate had any
//! other error, because the check only ran during stability checking. Both errors must be
//! reported.

#![feature(this_feature_does_not_exist)] //~ ERROR unknown feature `this_feature_does_not_exist`

struct Foo;

trait Bar {}

impl Bar for Foo {}
impl Bar for Foo {} //~ ERROR conflicting implementations of trait `Bar` for type `Foo`

fn main() {}
