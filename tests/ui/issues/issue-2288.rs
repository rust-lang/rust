//@ run-pass
#![allow(non_camel_case_types)]

trait clam<A> {
  fn chowder(&self, y: A);
}

#[derive(Copy, Clone)]
struct foo<A> {
  x: A,
}

impl<A> clam<A> for foo<A> {
  fn chowder(&self, _y: A) {
  }
}

fn foo<A>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn f<A>(x: Box<dyn clam<A>>, a: A) {
  x.chowder(a);
}

pub fn main() {

  let c = foo(42);
  let d: Box<dyn clam<isize>> = Box::new(c) as Box<dyn clam<isize>>;
  f(d, c.x);
}
