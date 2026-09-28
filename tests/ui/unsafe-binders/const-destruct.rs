//@ compile-flags: -Znext-solver
//@ check-pass

// Unsafe binders never have drop glue, so they are `const Destruct`.

#![feature(unsafe_binders, const_trait_impl, const_destruct)]
#![allow(incomplete_features)]

use std::marker::Destruct;
use std::unsafe_binder::wrap_binder;

const fn drop_it<T: [const] Destruct>(_: T) {}

const fn generic(b: unsafe<'a> &'a u8) {
    drop_it(b)
}

const _: () = {
    let b: unsafe<> u8 = unsafe { wrap_binder!(0) };
    drop_it(b)
};

fn main() {}
