//! Test that coercing between function items of the same function,
//! but with different generic args works.

//@check-pass

#![feature(type_alias_impl_trait)]

fn foo<T>(t: T) -> T {
    t
}

type F = impl Sized;

#[define_opaque(F)]
fn f(a: F) {
    let mut x = foo::<F>;
    x = foo::<()>;
    x(a);
    x(());
}

type G = impl Sized;

#[define_opaque(G)]
fn g(a: G) {
    let x = foo::<()>;
    let _: () = x(a);
}

type H = impl Sized;

#[define_opaque(H)]
fn h(a: H) {
    let x = foo::<H>;
    let _: H = x(());
}

type I = impl Sized;

#[define_opaque(I)]
fn i(a: I) {
    let mut x = foo::<()>;
    x = foo::<I>;
    x(a);
    x(());
}

type J = impl Sized;

#[define_opaque(J)]
fn j(a: J) {
    let x = match true {
        true => foo::<J>,
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
        false => foo::<()>,
    };
    todo!()
}

fn main() {}
