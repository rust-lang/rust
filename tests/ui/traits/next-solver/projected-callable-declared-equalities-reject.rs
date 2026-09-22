//@ check-fail
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Identity {
    type Output;
}

trait Carrier {
    type Assoc: Identity<Output = Self::Assoc>;
}

impl Carrier for () {
    type Assoc = u32;
    //~^ ERROR the trait bound `u32: Identity` is not satisfied
}

fn wrong_output<C: Carrier<Assoc = T>, T>(value: T) -> u32 {
    let output: <T as Identity>::Output = value;
    output
    //~^ ERROR mismatched types
}

trait Other<U> {
    type Assoc: Identity<Output = U>;
}

fn different_value<C: Other<U, Assoc = T>, T, U>(value: T) -> <T as Identity>::Output {
    value
    //~^ ERROR mismatched types
}

trait MissingEquality {
    type Assoc: Identity;
}

fn missing_equality<C: MissingEquality<Assoc = T>, T>(value: T) -> <T as Identity>::Output {
    value
    //~^ ERROR mismatched types
}

trait ScopedIdentity<'a, U> {
    type Output;
}

trait Scoped<U> {
    type Assoc: for<'a> ScopedIdentity<'a, &'a U, Output = Self::Assoc>;
}

fn missing_scope<C: Scoped<U, Assoc = T>, T, U>(
    value: T,
) -> <T as ScopedIdentity<'static, &'static U>>::Output {
    //~^ ERROR the parameter type `U` may not live long enough
    //~| ERROR the parameter type `U` may not live long enough
    value
}

trait ConditionalIdentity {
    type Output<'a> where Self: 'a;
}

trait Conditional {
    type Assoc: for<'a> ConditionalIdentity<Output<'a> = Self::Assoc>;
}

fn missing_premise<C: Conditional<Assoc = T>, T>(
    value: T,
) -> <T as ConditionalIdentity>::Output<'static> {
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    value
}

trait ForRegion<'a> {
    type Output;
}

trait ScopedPath<'a, U>: ForRegion<'a, Output = Self> {}

trait Paths<U> {
    type Assoc: for<'a> ScopedPath<'a, &'a U>;
}

fn neither_alternative<C, D, T, U, V>(value: T) -> <T as ForRegion<'static>>::Output
//~^ ERROR unable to satisfy outlives constraints
//~| ERROR unable to satisfy outlives constraints
//~| ERROR unable to satisfy outlives constraints
where
    C: Paths<U, Assoc = T>,
    D: Paths<V, Assoc = T>,
{
    value
}

fn main() {}
