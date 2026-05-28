//@ known-bug: #136766
#![feature(generic_const_exprs)]
trait A<const B: bool>{}
impl A<true> for () {}
fn c<const D: usize>(E: [u8; D * D]) where() : A<D>{}
fn main() { c }
