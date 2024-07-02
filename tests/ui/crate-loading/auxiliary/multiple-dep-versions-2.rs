#![crate_name="dependency"]
//@ edition:2021
//@ compile-flags: -C metadata=2 -C extra-filename=-2
pub struct Type(pub i32);
pub trait Trait {}
impl Trait for Type {}
pub fn do_something<X: Trait>(_: X) {}
