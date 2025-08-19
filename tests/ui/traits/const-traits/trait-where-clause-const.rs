//@ compile-flags: -Znext-solver
// Like trait-where-clause.rs, but we are calling from a const context.
// Checking the validity of traits' where clauses happen at a later stage.
// (`rustc_const_eval` instead of `rustc_hir_analysis`) Therefore one file as a
// test is not enough.

#![feature(const_trait_impl)]

#[const_trait]
trait Bar {}

#[const_trait]
trait Foo {
    fn a();
    fn b() where Self: [const] Bar;
    fn c<T: [const] Bar>();
}

const fn test1<T: [const] Foo + Bar>() {
    T::a();
    T::b();
    //~^ ERROR the trait bound
    T::c::<T>();
    //~^ ERROR the trait bound
}

const fn test2<T: [const] Foo + [const] Bar>() {
    T::a();
    T::b();
    T::c::<T>();
}

fn main() {}
