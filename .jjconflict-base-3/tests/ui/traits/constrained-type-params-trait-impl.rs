//! Comprehensive test for type parameter constraints in trait implementations
//!
//! This tests various scenarios of type parameter usage in trait implementations:
//! - Properly constrained parameters through trait bounds
//! - Unconstrained parameters that should cause compilation errors
//! - Complex constraint scenarios with `where` clauses and associated types
//! - Conflicting implementations detection

trait Foo<A> {
    fn get(&self, A: &A) {}
}

trait Bar {
    type Out;
}

impl<T> Foo<T> for [isize; 0] {
    // OK: T is used in the trait bound `Foo<T>`
}

impl<T, U> Foo<T> for [isize; 1] {
    //~^ ERROR the type parameter `U` is not constrained
    // T is constrained by `Foo<T>`, but U is completely unused
}

impl<T, U> Foo<T> for [isize; 2]
where
    T: Bar<Out = U>,
{
    // OK: T is constrained by `Foo<T>`, U is constrained by the where clause
}

impl<T: Bar<Out = U>, U> Foo<T> for [isize; 3] {
    // OK: Same as above but using bound syntax instead of where clause
}

impl<T, U> Foo<T> for U {
    //~^ ERROR conflicting implementations of trait `Foo<_>` for type `[isize; 0]`
    // This conflicts with the first impl when U = [isize; 0]
}

impl<T, U> Bar for T {
    //~^ ERROR the type parameter `U` is not constrained
    type Out = U;
    // Using U only in associated type definition is insufficient for constraint
}

impl<T, U> Bar for T
where
    T: Bar<Out = U>,
{
    //~^^^^ ERROR the type parameter `U` is not constrained by the impl trait, self type, or predicates
    //~| ERROR conflicting implementations of trait `Bar`
    // Self-referential constraint doesn't properly constrain U
}

impl<T, U, V> Foo<T> for T
where
    (T, U): Bar<Out = V>,
{
    //~^^^^ ERROR the type parameter `U` is not constrained
    //~| ERROR the type parameter `V` is not constrained
    //~| ERROR conflicting implementations of trait `Foo<[isize; 0]>` for type `[isize; 0]`
    // V is bound through output type, but U and V are not properly constrained as inputs
}

impl<T, U, V> Foo<(T, U)> for T
where
    (T, U): Bar<Out = V>,
{
    //~^^^^ ERROR conflicting implementations of trait `Foo<([isize; 0], _)>` for type `[isize; 0]`
    // Both T and U are constrained through `Foo<(T, U)>`, but creates conflicting impl
}

fn main() {}
