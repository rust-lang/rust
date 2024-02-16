//@ check-pass

// Tests the default for the unused_features lint

#![allow(stable_features)]
// FIXME(#44232) we should warn that this isn't used.
#![feature(rust1)]

fn main() {}
