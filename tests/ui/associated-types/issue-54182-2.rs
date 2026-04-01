//@ check-pass

// Before RFC 2532, normalizing a defaulted assoc. type didn't work at all,
// unless the impl in question overrides that type, which makes the default
// pointless.

#![feature(associated_type_defaults)]

trait Tr {
    type Assoc = ();
}

impl Tr for () {}

fn f(thing: <() as Tr>::Assoc) {
    let c: () = thing;
}

fn main() {}
