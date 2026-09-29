//@ compile-flags: -Znext-solver
//@ check-pass
//@ known-bug: #130516

// `#[must_use]` looks through unsafe binders.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]
#![deny(unused_must_use)]

use std::unsafe_binder::wrap_binder;

#[must_use]
#[derive(Clone, Copy)]
struct M;

fn bound() -> unsafe<> M {
    unsafe { wrap_binder!(M) }
}

fn main() {
    bound();
}
