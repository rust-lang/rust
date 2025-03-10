//@ check-pass

#![feature(const_closures)]
#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

const fn foo() {
    let cl = const use || {};
}

fn main() {}
