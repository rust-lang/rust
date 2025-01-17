//@ check-pass
#![allow(non_camel_case_types)]


trait clam<A> { fn get(self) -> A; }
trait foo<A> {
   fn bar<B,C:clam<A>>(&self, c: C) -> B;
}

pub fn main() { }
