#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait Provider {
    type A<'a>;
      //~^ ERROR: missing generics for associated type
}

impl Provider for () {
    type A<'a> = ();
}

struct Holder<B> {
  inner: Box<dyn Provider<A = B>>,
}

fn main() {
    Holder {
        inner: Box::new(()),
    };
}
