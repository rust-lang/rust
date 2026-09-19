//! Regression test for <https://github.com/rust-lang/rust/issues/34430>.
//!
//! An associated-type projection under a higher-ranked binder
//! (`for<'a> FnOnce(&'a Foo) -> <T as WithLifetime<'a>>::Type`) failed to normalize,
//! so returning `&'a Foo` from the closure was rejected even though
//! `<FooRef as WithLifetime<'a>>::Type` *is* `&'a Foo`.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![allow(unused)]

trait WithLifetime<'a> {
    type Type;
}

struct Foo;

enum FooRef {}

impl<'a> WithLifetime<'a> for FooRef {
    type Type = &'a Foo;
}

fn wub<T, F>(f: F)
where
    T: for<'a> WithLifetime<'a>,
    F: for<'a> FnOnce(&'a Foo) -> <T as WithLifetime<'a>>::Type,
{
}

fn main() {
    wub::<FooRef, _>(|foo| foo);

    // Annotating the closure's return type used to ICE instead. Both the concrete
    // and the projected spelling are checked, since either could regress alone.
    wub::<FooRef, _>(|foo| -> &Foo { foo });
    wub::<FooRef, _>(|foo| -> <FooRef as WithLifetime>::Type { foo });
}
