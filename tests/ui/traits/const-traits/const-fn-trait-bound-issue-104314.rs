//@ check-pass

#![feature(const_trait_impl, const_destruct, const_clone)]

use std::marker::Destruct;

const fn f<T, F: [const] Fn(&T) -> T + [const] Destruct>(_: F) {}

const fn g<T: [const] Clone>() {
    f(<T as Clone>::clone);
}

fn main() {}
