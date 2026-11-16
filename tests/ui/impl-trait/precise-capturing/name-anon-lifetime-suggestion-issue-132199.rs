//@ edition: 2021
//@ run-rustfix

// regression test for https://github.com/rust-lang/rust/issues/132199.
// in edition 2024 impl Trait captures every in-scope lifetime by default,
// so none of these shapes error there. edition 2021 is used throughout.

#![allow(dead_code)]

struct T;
struct Foo<'a>(&'a u8);

impl T {
    // two_anon: two elided lifetimes, use<'_> captures only the first so a
    // second E0700 follows with no suggestion.
    fn two_anon(&self, t: &T) -> impl Sized {
        (self, t)
        //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
    }

    // one_anon: one elided lifetime, use<'_> is correct and must stay byte-identical.
    fn one_anon(&self) -> impl Sized {
        self
        //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
    }

    // named: an already-named lifetime, unaffected by this change.
    fn named<'a>(&'a self, t: &'a T) -> impl Sized {
        (self, t)
        //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
    }
}

// second_anon: two elided lifetimes where use<'_> does not resolve and
// produces E0106 on top of the original error.
fn second_anon(_a: &u8, b: &T) -> impl Sized {
    b
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

// path_anon: elision inside a path argument, so the name goes in as Foo<'a>
// and exercises the other arm of hir::LifetimeName matching.
fn path_anon(x: Foo, y: Foo) -> impl Sized {
    (x, y)
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
