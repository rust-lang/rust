//@ compile-flags: -Znext-solver=globally
#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Regression test for https://github.com/rust-lang/rust/issues/153539:

struct S<'a>(&'a ());

impl<X> S<'_> {
    //~^ ERROR the type parameter `X` is not constrained by the impl trait, self type, or predicates
    type P = ();
}

fn ret_ref_local<'e>() -> &'e i32 {
    let f: for<'a> fn(&'a i32) -> S<'a>::P = |x| _ = x;

    f(&1)
    //~^ ERROR mismatched types
}

fn main() {}
