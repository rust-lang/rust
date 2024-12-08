// This is a regression test for issue #114221.
// Check that we compute variances for lazy type aliases.

//@ check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

// [+] `A` is covariant over `'a`.
struct A<'a>(Co<'a>);

// [+] `Co` is covariant over `'a`.
type Co<'a> = &'a ();

fn co<'a>(x: A<'static>) {
    let _: A<'a> = x;
}

// [-] `B` is contravariant over `'a`.
struct B<'a>(Contra<'a>);

// [-] `Contra` is contravariant over `'a`.
type Contra<'a> = fn(&'a ());

fn contra<'a>(x: B<'a>) {
    let _: B<'static> = x;
}

struct C<T, U>(CoContra<T, U>);

// [+, -] `CoContra` is covariant over `T` and contravariant over `U`.
type CoContra<T, U> = Option<(T, fn(U))>;

fn co_contra<'a>(x: C<&'static (), &'a ()>) -> C<&'a (), &'static ()> {
    x
}

fn main() {}
