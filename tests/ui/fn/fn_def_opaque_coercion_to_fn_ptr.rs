//! Test that coercing between function items of different functions works,
//! as long as their signatures match. The resulting value is a function pointer.

#![feature(type_alias_impl_trait)]

fn foo<T>(t: T) -> T {
    t
}

fn bar<T>(t: T) -> T {
    t
}

type F = impl Sized;

#[defines(F)]
fn f(a: F) {
    let mut x = bar::<F>;
    x = foo::<()>; //~ ERROR: mismatched types
    x(a);
    x(());
}

type I = impl Sized;

#[defines(I)]
fn i(a: I) {
    let mut x = bar::<()>;
    x = foo::<I>; //~ ERROR: mismatched types
    x(a);
    x(());
}

type J = impl Sized;

#[defines(J)]
fn j(a: J) {
    let x = match true {
        true => bar::<J>,
        false => foo::<()>,
    };
    x(a);
    x(());
}

fn k() -> impl Sized {
    fn bind<T, F: FnOnce(T) -> T>(_: T, f: F) -> F {
        f
    }
    let x = match true {
        true => {
            let f = foo;
            bind(k(), f)
        }
        false => bar::<()>,
    };
    todo!()
}

fn main() {}
