//@ check-fail
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Bound<'r, U> {
    type Out: 'r;
}

trait Scoped<U> {
    type Assoc: for<'r> Bound<'r, &'r U, Out = Self::Assoc>;
}

fn missing_scope<C: Scoped<U, Assoc = T>, T, U>(value: T) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Gat {
    type Out<'r>: 'r where Self: 'r;
}

trait Conditional {
    type Assoc: for<'r> Gat<Out<'r> = Self::Assoc>;
}

fn missing_gat_premise<C: Conditional<Assoc = T>, T>(value: T) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Has<'r> {
    type Assoc: Bound<'r, (), Out = Self::Assoc>;
}

fn wrong_region<'r, C: Has<'r, Assoc = T>, T>(value: T) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Other<U> {
    type Assoc: for<'r> Bound<'r, (), Out = U>;
}

fn unrelated<C: Other<U, Assoc = T>, T, U>(value: T) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
}

trait ConditionalBound<'r, U> {
    type Out: 'r where U: 'r;
}

trait ConditionalOn<U> {
    type Assoc: for<'r> ConditionalBound<'r, U, Out = Self::Assoc>;
}

fn cycle<C: ConditionalOn<U, Assoc = T>, D: ConditionalOn<T, Assoc = U>, T, U>(
    value: T,
) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

trait Good<'r>: Bound<'r, (), Out = Self> {}
trait ScopedPath<'r, U>: Good<'r> {}
trait OnlyScoped<U> {
    type Assoc: for<'r> ScopedPath<'r, &'r U>;
}

fn missing_supertrait_premise<C: OnlyScoped<U, Assoc = T>, T, U>(
    value: T,
) -> Box<dyn std::any::Any> {
    Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

fn missing_closure_requirements<'a, 'b, 'r, C, D, T>(
    value: T,
) -> impl FnOnce() -> Box<dyn std::fmt::Debug + 'r>
where
    C: Has<'a, Assoc = T>,
    D: Has<'b, Assoc = T>,
    T: std::fmt::Debug,
{
    move || Box::new(value)
    //~^ ERROR unable to satisfy outlives constraints
}

fn main() {}
