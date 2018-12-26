// Ensure that OIBIT checks `T` when it encounters a `PhantomData<T>` field, instead of checking
// the `PhantomData<T>` type itself (which almost always implements an auto trait)

#![feature(optin_builtin_traits)]

use std::marker::{PhantomData};

unsafe auto trait Zen {}

unsafe impl<'a, T: 'a> Zen for &'a T where T: Sync {}

struct Guard<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
}

struct Nested<T>(T);

fn is_zen<T: Zen>(_: T) {}

fn not_sync<T>(x: Guard<T>) {
    is_zen(x)
    //~^ ERROR `T` cannot be shared between threads safely [E0277]
}

fn nested_not_sync<T>(x: Nested<Guard<T>>) {
    is_zen(x)
    //~^ ERROR `T` cannot be shared between threads safely [E0277]
}

fn main() {}
