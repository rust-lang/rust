//@ known-bug: unknown
//@ run-pass
// This is not correctly parsing the nested use

#![feature(ergonomic_clones)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn work(_: Box<Foo>) {}
fn foo<F:FnOnce()>(_: F) {}

pub fn main() {
  let a = Box::new(Foo);
  foo(use|| { foo(use|| { work(a) }) });
  use || { use || { use || { Foo } } };
}
