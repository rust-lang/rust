#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {}
struct S;

impl Trait for u8 {
    reuse unresolved::*; //~ ERROR cannot find module or crate `unresolved`
    reuse S::*; //~ ERROR expected trait, found struct `S`
}

fn main() {}
