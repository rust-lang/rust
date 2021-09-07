#![feature(generic_const_exprs, adt_const_params, const_trait_impl)]
#![allow(incomplete_features)]

#[derive(PartialEq, Eq)]
struct Foo(u8);

impl const std::ops::Add for Foo {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self
    }
}

struct Evaluatable<const N: Foo>;

fn foo<const N: Foo>(a: Evaluatable<{ N + N }>) {
    bar::<{ std::ops::Add::add(N, N) }>();
}

fn bar<const N: Foo>() {}

fn main() {}
