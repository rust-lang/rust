//@ run-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn work(_: Box<Foo>) {}
fn foo<F:FnOnce()>(_: F) {}

pub fn main() {
  let a = Box::new(Foo);
  foo(use || { foo(use || { work(a) }) });
  let x = use || { use || { Foo } };
  let _y = x();
}
