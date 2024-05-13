#![crate_name="dependency"]
//@ edition:2021
//@ compile-flags: -C metadata=1 -C extra-filename=-1
pub struct Type;
pub trait Trait {}
impl Trait for Type {}
pub fn do_something<X: Trait>(_: X) { }
