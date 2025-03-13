//@ check-pass

#![feature(generic_const_items, trivial_bounds)]
#![allow(incomplete_features, dead_code, trivial_bounds)]

const UNUSED: () = ()
where
    String: Copy;

fn main() {}
