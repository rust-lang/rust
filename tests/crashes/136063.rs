//@ known-bug: #136063
#![feature(generic_const_exprs)]
trait A<const B: u8 = X> {}
impl A<1> for bool {}
fn bar(arg : &dyn A<x>) { bar(true) }
pub fn main() {}
