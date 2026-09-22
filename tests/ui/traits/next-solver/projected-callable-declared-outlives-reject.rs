//@ check-fail
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Has<'r> {
    type Assoc: 'r;
}

fn unrelated<'a, C, T>(value: &'a T) -> &'static T
where
    C: Has<'static, Assoc = ()>,
{
    value
    //~^ ERROR lifetime may not live long enough
}

fn wrong_region<'a, C, T>(value: T) -> Box<dyn std::any::Any>
where
    C: Has<'a, Assoc = T>,
{
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Conditional {
    type Assoc<'a>: 'a where Self: 'a;
}

fn conditional<T>(value: T) -> Box<dyn std::any::Any>
where
    T: for<'a> Conditional<Assoc<'a> = T>,
{
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait ReferenceInput<'a, T> {
    type Assoc: 'a;
}

fn reference_input<T>(value: T) -> Box<dyn std::any::Any>
where
    for<'a> (): ReferenceInput<'a, &'a T, Assoc = T>,
{
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait ConditionalOn<'r, T> {
    type Assoc: 'r where T: 'r;
}

fn cycle<C, D, T, U>(value: T) -> Box<dyn std::any::Any>
where
    for<'r> C: ConditionalOn<'r, U, Assoc = T>,
    for<'r> D: ConditionalOn<'r, T, Assoc = U>,
{
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Bound<'a, U>: 'a {}
trait ScopedDeclaration<U> {
    type Assoc: for<'a> Bound<'a, &'a U>;
}

fn declaration_premise<C: ScopedDeclaration<U, Assoc = T>, T, U>(
    value: T,
) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Super<U>: for<'a> Bound<'a, &'a U> {}
trait ScopedSuper<U> {
    type Assoc: Super<U>;
}

fn supertrait_premise<C: ScopedSuper<U, Assoc = T>, T, U>(value: T) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

fn main() {}
