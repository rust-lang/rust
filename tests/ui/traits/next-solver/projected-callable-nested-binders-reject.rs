//@ check-fail
//@ compile-flags: -Znext-solver=globally

#![allow(dead_code)]

trait Family {
    type View<'a, 'b>;
}

fn independent<T: Family, F>()
where
    F: for<'a, 'b> Fn(for<'c> fn(T::View<'a, 'c>)) -> for<'d> fn(T::View<'b, 'd>),
    //~^ ERROR binding for associated type `Output` references lifetime `'b`
{
}

fn wrong_scope<T: Family, F>()
where
    F: for<'a> Fn(for<'b> fn(T::View<'a, 'b>)) -> for<'c> fn(T::View<'c, 'a>),
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

fn extra_reference<T: Family, F>()
where
    F: for<'a> Fn(for<'b> fn(T::View<'a, 'b>))
        -> (for<'c> fn(T::View<'a, 'c>), &'a ()),
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

trait TripleFamily {
    type View<'a, 'b, 'c>;
}

fn different_relationship<T: TripleFamily, F>()
where
    F: for<'a> Fn(for<'b> fn(T::View<'a, 'b, 'b>))
        -> for<'c, 'd> fn(T::View<'a, 'c, 'd>),
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

fn main() {}
