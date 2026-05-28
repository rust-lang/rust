//@ check-pass

#![feature(unsafe_binders)]

fn main() {
    let x: unsafe<'a> &'a ();
}
