//@ compile-flags: -Znext-solver
//@ build-pass
//@ compile-flags: -Csymbol-mangling-version=v0

// v0 symbol mangling works with unsafe binders. See #154367.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

fn generic<T>() {}

fn main() {
    generic::<unsafe<'a> &'a ()>();
}
