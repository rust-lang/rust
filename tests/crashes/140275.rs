//@ known-bug: #140275
#![feature(generic_const_exprs)]
trait T{}
trait V{}
impl<const N: i32> T for [i32; N::<&mut V>] {}
