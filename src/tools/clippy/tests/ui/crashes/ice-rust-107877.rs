//@ check-pass

#![allow(dead_code)]

struct Foo;

#[allow(clippy::infallible_try_from)]
impl<'a> std::convert::TryFrom<&'a String> for Foo {
    type Error = std::convert::Infallible;

    fn try_from(_: &'a String) -> Result<Self, Self::Error> {
        Ok(Foo)
    }
}

fn find<E>(_: impl std::convert::TryInto<Foo, Error = E>) {}

fn main() {
    find(&String::new());
}
