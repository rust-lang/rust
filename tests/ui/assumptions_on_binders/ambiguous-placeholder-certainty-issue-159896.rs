// Regression test for issue #159896.
//
// An ambiguous result from eagerly handling placeholders must be preserved
// when returning nested normalization goals, instead of causing an ICE.

//@ check-fail
//@ dont-check-compiler-stderr
//@ dont-require-annotations: ERROR
//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders
//@ edition: 2021

#![feature(type_alias_impl_trait)]

type FooArg<'a> = &'a impl Iterator<Item = FooItem>;
type FooRet = impl Iterator<Item = FooItem>;
type FooItem = Box<dyn Fn(FooArg) -> FooRet>;

struct Bar;

impl Iterator for Bar {
    type Item = FooItem;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn main() {}
