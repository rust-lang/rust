// check-pass
// Regression test for issue #55099
// Tests that we don't incorrectly consider a lifetime to part
// of the concrete type

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Future {
}

struct AndThen<F>(F);

impl<F> Future for AndThen<F> {
}

struct Foo<'a> {
    x: &'a mut (),
}

type F = impl Future;

impl<'a> Foo<'a> {
    fn reply(&mut self) -> F {
        AndThen(|| ())
    }
}

fn main() {}
