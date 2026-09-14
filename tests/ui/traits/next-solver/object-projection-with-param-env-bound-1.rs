//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// A regression test for https://github.com/rust-lang/rust/issues/152789.
// Ensures we do not trigger an ICE or compilation error when normalizing
// a projection on a trait object, where the matching trait predicate
// (with the same trait id as the object bound) originates from another
// source such as the param env, rather than from the object's own bounds.

pub trait Trait<T> {
    type Assoc;
}

pub trait Foo {
    type FooAssoc;
}

pub struct Wrap<U: Foo>(<dyn Trait<i32, Assoc = i64> as Trait<U::FooAssoc>>::Assoc)
where
    dyn Trait<i32, Assoc = i64>: Trait<U::FooAssoc>;

fn main() {}
