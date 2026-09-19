//@ run-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

trait Family {
    type View<'a>: Clone;
}

fn identity<'a, T: Family>(value: T::View<'a>) -> T::View<'a>
where
    T::View<'a>: Sized,
{
    value
}

fn copy<'a, T: Family>(value: T::View<'a>) -> T::View<'a>
where
    T::View<'a>: Clone,
{
    value.clone()
}

fn apply<T: Family>(f: impl for<'a> Fn(T::View<'a>) -> T::View<'a>) {
    drop(f);
}

fn from_reference<'a, T: Family>(value: &'a T::View<'a>) -> T::View<'a>
where
    T::View<'a>: 'a,
{
    value.clone()
}

fn apply_reference<T: Family>(_: impl for<'a> Fn(&'a T::View<'a>) -> T::View<'a>) {}

fn check<T: Family>() {
    apply::<T>(identity::<T>);
    apply::<T>(copy::<T>);
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = identity::<T>;
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = copy::<T>;
    apply_reference::<T>(from_reference::<T>);
    let _: for<'a> fn(&'a T::View<'a>) -> T::View<'a> = from_reference::<T>;
}

struct Borrowed;
impl Family for Borrowed {
    type View<'a> = &'a u32;
}

struct Erased;
impl Family for Erased {
    type View<'a> = ();
}

fn main() {
    check::<Borrowed>();
    check::<Erased>();
    let callback: for<'a> fn(<Borrowed as Family>::View<'a>) -> <Borrowed as Family>::View<'a> =
        copy::<Borrowed>;
    assert_eq!(*callback(&19), 19);
}
