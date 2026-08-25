//@ check-pass
// Regression test for issue #55099
// Tests that we don't incorrectly consider a lifetime to part
// of the concrete type

#![feature(type_alias_impl_trait)]

trait Future {}

struct AndThen<F>(F);

impl<F> Future for AndThen<F> {}

struct Foo<'a> {
    x: &'a mut (),
}

type F = impl Future;

impl<'a> Foo<'a> {
    #[define_opaque(F)]
    fn reply(&mut self) -> F {
        AndThen(|| ())
    }
}

fn main() {}
