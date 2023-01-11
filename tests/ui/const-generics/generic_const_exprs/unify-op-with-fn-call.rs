#![feature(generic_const_exprs, adt_const_params, const_trait_impl)]
#![allow(incomplete_features)]

// test `N + N` unifies with explicit function calls for non-builtin-types
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

// test that `N + N` unifies with explicit function calls for builin-types
struct Evaluatable2<const N: usize>;

fn foo2<const N: usize>(a: Evaluatable2<{ N + N }>) {
    bar2::<{ std::ops::Add::add(N, N) }>();
    //~^ error: unconstrained generic constant
    // FIXME(generic_const_exprs) make this not an error
}

fn bar2<const N: usize>() {}

fn main() {}
