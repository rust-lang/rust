//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(unsafe_binders)]

fn main() {
    let x: unsafe<'a> &'a ();
}
