// build-pass (FIXME(62277): could be check-pass?)
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

trait clam<A> { fn get(self) -> A; }
trait foo<A> {
   fn bar<B,C:clam<A>>(&self, c: C) -> B;
}

pub fn main() { }
