#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {}
struct S;

impl Trait for u8 {
    reuse unresolved::*; //~ ERROR failed to resolve: use of unresolved module or unlinked crate `unresolved`
    reuse S::*; //~ ERROR expected trait, found struct `S`
}

fn main() {}
