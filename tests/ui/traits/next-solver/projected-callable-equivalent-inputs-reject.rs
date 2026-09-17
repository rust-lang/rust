//@ check-fail
//@ compile-flags: -Znext-solver=globally

#![allow(dead_code, type_alias_bounds)]

trait Family {
    type View<'a>;
}

fn only_static<T, F>()
where
    T: Family<View<'static> = &'static ()>,
    F: for<'a> Fn(T::View<'a>) -> &'a (),
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

fn unrelated<T: Family, U: Family, F>()
where
    F: for<'a> Fn(T::View<'a>) -> U::View<'a>,
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

fn wrong_lifetime<T, U, F>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
    F: for<'a, 'b> Fn(T::View<'a>) -> U::View<'b>,
    //~^ ERROR binding for associated type `Output` references lifetime `'b`
{
}

fn independent_reference<T, U, F>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
    F: for<'a> Fn(T::View<'a>) -> (U::View<'a>, &'a ()),
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

type Unused<T: Family> = for<'a> fn(T::View<'a>) -> &'a ();
//~^ ERROR return type references lifetime `'a`

struct Holder<T: Family> {
    pointer: for<'a> fn(T::View<'a>) -> &'a (),
    //~^ ERROR return type references lifetime `'a`
}

fn independent_binder<T: Family, F>()
where
    F: for<'a> Fn(for<'b> fn(T::View<'b>)) -> T::View<'a>,
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
{
}

fn main() {}
