//@ compile-flags: -Znext-solver=globally

// Regression test that ensures clippy fix won't accidently deref the argument of
// a closure when we fix `.find(..).is_some()` into `.any(..)` due to mishandling
// of non-rigid aliases.

#![warn(clippy::search_is_some)]
#![allow(clippy::explicit_auto_deref, clippy::manual_contains)]
#![expect(clippy::useless_vec)]

fn make_arg_no_deref_impl() -> impl Fn(&&u32) -> bool {
    move |x: &&u32| **x == 78
}

fn main() {
    let v = vec![3, 2, 1, 0];
    let arg_no_deref_impl = make_arg_no_deref_impl();

    #[allow(clippy::redundant_closure)]
    let _ = v.iter().find(|x: &&u32| arg_no_deref_impl(x)).is_some();
    //~^ search_is_some
}
