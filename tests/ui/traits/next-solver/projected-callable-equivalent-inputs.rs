//@ check-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Family {
    type View<'a>;
}

// An environment equality can make two differently written projections the
// same complete input and output type.
fn equivalent<T, U, F>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
    F: for<'a> Fn(T::View<'a>) -> U::View<'a>,
{
}

fn equivalent_mut<T, U, F>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
    F: for<'a> FnMut(T::View<'a>) -> Option<U::View<'a>>,
{
}

fn equivalent_once<T, U, F>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
    F: for<'a> FnOnce(T::View<'a>) -> Vec<U::View<'a>>,
{
}

fn pointer<T, U>()
where
    U: Family,
    for<'a> T: Family<View<'a> = U::View<'a>>,
{
    let _: Option<for<'a> fn(T::View<'a>) -> U::View<'a>> = None;
    let _: Option<&dyn for<'a> Fn(T::View<'a>) -> U::View<'a>> = None;
}

fn retain<'a, T, U>(value: T::View<'a>) -> U::View<'a>
where
    U: Family,
    for<'b> T: Family<View<'b> = U::View<'b>>,
{
    value
}

fn reify<T, U>()
where
    U: Family,
    for<'b> T: Family<View<'b> = U::View<'b>>,
{
    let _: for<'a> fn(T::View<'a>) -> U::View<'a> = retain::<T, U>;
}

struct Loan;

impl Family for Loan {
    type View<'a> = &'a u32;
}

struct Erased;

impl Family for Erased {
    type View<'a> = ();
}

fn through_function<T: Family, F>()
where
    F: for<'a> Fn(fn(T::View<'a>)) -> T::View<'a>,
{
}

fn through_nested_binder<T: Family, F>()
where
    F: for<'a> Fn(for<'b> fn(&'b T::View<'a>)) -> T::View<'a>,
{
}

trait Lending {
    type View<'a> where Self: 'a;
}

fn lending<T: Lending, U: Lending, F>()
where
    for<'a> T: Lending<View<'a> = U::View<'a>>,
    F: for<'a> Fn(T::View<'a>) -> U::View<'a>,
{
}

fn lending_pointer<T: Lending, U: Lending>()
where
    for<'a> T: Lending<View<'a> = U::View<'a>>,
{
    let _: Option<for<'a> fn(T::View<'a>) -> U::View<'a>> = None;
    let _: Option<&dyn for<'a> Fn(T::View<'a>) -> U::View<'a>> = None;
}

fn main() {
    reify::<Loan, Loan>();
    reify::<Erased, Erased>();
}
