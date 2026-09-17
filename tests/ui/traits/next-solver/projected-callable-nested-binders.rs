//@ run-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Family {
    type View<'a, 'b>;
}

fn accepts<T: Family, F>(f: F) -> F
where
    F: for<'a> Fn(for<'b> fn(T::View<'a, 'b>)) -> for<'c> fn(T::View<'a, 'c>),
{
    f
}

fn relay<'a, T: Family>(
    callback: for<'b> fn(T::View<'a, 'b>),
) -> for<'c> fn(T::View<'a, 'c>) {
    callback
}

fn pointer<T: Family>() {
    let _: for<'a> fn(for<'b> fn(T::View<'a, 'b>)) -> for<'c> fn(T::View<'a, 'c>) =
        relay::<T>;
}

fn nested<T: Family, F>()
where
    F: for<'a> Fn(Option<for<'b> fn(T::View<'a, 'b>)>) -> for<'c> fn(T::View<'a, 'c>),
{
}

fn object<T: Family, F>()
where
    F: for<'a> Fn(Box<dyn for<'b> Fn(T::View<'a, 'b>)>)
        -> Box<dyn for<'c> Fn(T::View<'a, 'c>)>,
{
}

trait TripleFamily {
    type View<'a, 'b, 'c>;
}

// Binder declaration order does not affect the relationship between occurrences.
fn reordered<T: TripleFamily, F>()
where
    F: for<'a> Fn(for<'b, 'c> fn(T::View<'a, 'c, 'b>))
        -> for<'d, 'e> fn(T::View<'a, 'd, 'e>),
{
}

struct Borrowed;
impl Family for Borrowed {
    type View<'a, 'b> = (&'a u32, &'b u32);
}

struct Erased;
impl Family for Erased {
    type View<'a, 'b> = ();
}

fn main() {
    let f = accepts::<Borrowed, _>(relay::<Borrowed>);
    let check = f(|(a, b)| assert_eq!(a, b));
    check((&17, &17));
    accepts::<Erased, _>(relay::<Erased>)(|()| {})(());
    let _ = accepts::<Borrowed, _>(|callback| callback);
    pointer::<Borrowed>();
    pointer::<Erased>();
}
