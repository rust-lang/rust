//@ compile-flags: -Znext-solver=globally

#![feature(const_trait_impl)]

const fn with_positive<F: for<'a> [const] Fn(&'a ())>() {}

const _: () = {
    with_positive::<()>(); //~ ERROR expected a `Fn(&'a ())` closure, found `()`
};

fn main() {}
