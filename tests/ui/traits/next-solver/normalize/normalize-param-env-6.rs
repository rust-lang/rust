//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#246
// Fixed by eager norm and marking param env as rigid.

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
// makes it work:
// #![recursion_limit = "512"]

pub trait Trait {
    type K;
}

// two different types
pub struct T1;
pub struct T2;

// a type that's easy to make really large
pub struct Growing<T>(T);
// for which proving that it implements foo grows with that size
pub trait Foo {
    type Output;
}
impl<T: Foo> Foo for Growing<T> {
    type Output = <T as Foo>::Output;
}
impl Foo for T2 {
    type Output = T1;
}
// a simple way to do this proof
pub type Eval<T> = <T as Foo>::Output;

// a trivial trait bound for one of the types
pub trait Trivial {}
impl Trivial for T1 {}

// and one for which one of the possible impls diverges
pub trait Diverges<T> {}
impl<I> Diverges<T2> for I {}
impl<I, R> Diverges<R> for I
where
    R: Trivial, // move this bound down
    Growing<I>: Diverges<T2>,
{
}

// Our large type
type LargeToEval = Growing<Growing<Growing<Growing<T2>>>>;

impl<K> Trait for K
where
    (): Diverges<T2>,
    T1: Trivial,
    Eval<LargeToEval>: Trivial,
{
    type K = K;
}

fn foo()
where
    Eval<LargeToEval>: Trivial,
    Eval<<LargeToEval as Trait>::K>: Trivial,
{
}

fn main() {
    foo()
}
