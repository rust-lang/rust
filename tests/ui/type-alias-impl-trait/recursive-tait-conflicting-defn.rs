// issue: 113596

#![feature(type_alias_impl_trait)]

trait Test {}

struct A;

impl Test for A {}

struct B<T> {
    inner: T,
}

impl<T: Test> Test for B<T> {}

pub type TestImpl = impl Test;

#[define_opaque(TestImpl)]
pub fn test() -> TestImpl {
    A
}

#[define_opaque(TestImpl)]
fn make_option2() -> Option<TestImpl> {
    //~^ ERROR concrete type differs from previous defining opaque type use
    let inner = make_option().unwrap();
    Some(B { inner })
}

fn make_option() -> Option<TestImpl> {
    Some(test())
}

fn main() {}
