//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// A regression test for https://github.com/rust-lang/rust/issues/152789.
// Ensures we do not trigger an ICE when normalization fails for a
// projection on a trait object, even if the projection has the same
// trait id as the object's bound.

pub trait Trait<T> {
    type Assoc;
}

pub trait Foo {
    type FooAssoc;
}

pub struct Wrap<U: Foo>(<dyn Trait<i32, Assoc = i64> as Trait<U::FooAssoc>>::Assoc);
//~^ ERROR: the trait bound `(dyn Trait<i32, Assoc = i64> + 'static): Trait<<U as Foo>::FooAssoc>` is not satisfied

fn main() {}
