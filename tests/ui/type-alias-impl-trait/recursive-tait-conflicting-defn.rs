// issue: 113596

#![feature(type_alias_impl_trait)]

trait Test {}

struct A;

impl Test for A {}

struct B<T> {
  inner: T,
}

impl<T: Test> Test for B<T> {}

type TestImpl = impl Test;

fn test() -> TestImpl {
  A
}

fn make_option() -> Option<TestImpl> {
  Some(test())
}

fn make_option2() -> Option<TestImpl> {
  let inner = make_option().unwrap();

  Some(B { inner })
  //~^ ERROR concrete type differs from previous defining opaque type use
}

fn main() {}
