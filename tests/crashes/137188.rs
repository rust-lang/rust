//@ known-bug: #137188
#![feature(min_generic_const_args)]
trait Trait {}
impl Trait for [(); N] {}
fn N<T>() {}
pub fn main() {}
