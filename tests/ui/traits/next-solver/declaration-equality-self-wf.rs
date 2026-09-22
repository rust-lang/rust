//@ check-fail
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Bound<'a> {
    type Out: 'a;
}

fn require_static<T: 'static>() {}

fn missing<T, U>()
where
    for<'a> &'a U: Bound<'a, Out = T>,
{
    require_static::<T>();
    //~^ ERROR unable to satisfy outlives constraints
}

fn proven<T, U: 'static>()
where
    for<'a> &'a U: Bound<'a, Out = T>,
{
    require_static::<T>();
}

trait Family {
    type View<'a> where Self: 'a;
}

fn missing_gat<C: Family, T>()
where
    for<'a> C::View<'a>: Bound<'a, Out = T>,
{
    require_static::<T>();
    //~^ ERROR unable to satisfy outlives constraints
}

fn proven_gat<C: Family + 'static, T>()
where
    for<'a> C::View<'a>: Bound<'a, Out = T>,
{
    require_static::<T>();
}

fn main() {}
