//@ check-fail
//@ compile-flags: -Znext-solver=globally

#![allow(dead_code)]

trait Family {
    type View<'a>;
}

fn restricted<'a: 'static, T: Family>(value: T::View<'a>) -> T::View<'a> {
    value
}

fn copy<'a, T: Family>(value: T::View<'a>) -> T::View<'a>
where
    T::View<'a>: Clone,
{
    value.clone()
}

fn apply<T: Family>(_: impl for<'a> Fn(T::View<'a>) -> T::View<'a>) {}

fn requires_static<T: Family>() {
    apply::<T>(restricted::<T>);
    //~^ ERROR type mismatch resolving
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = restricted::<T>;
    //~^ ERROR mismatched types
}

fn requires_clone<T: Family>()
where
    T::View<'static>: Clone,
{
    apply::<T>(copy::<'static, T>);
    //~^ ERROR type mismatch resolving
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = copy::<'static, T>;
    //~^ ERROR mismatched types
}

fn main() {}
