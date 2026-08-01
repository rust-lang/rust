//@ edition:2024
//@ aux-crate:mioou=mioou.rs

use mioou::*;

struct A;

impl Trait1 for A {
//~^ ERROR not all trait items implemented, missing one of: `a1`, `b1`
}

impl Trait2 for A {
//~^ ERROR not all trait items implemented, missing one of: `a2`, `b2`
}

fn main() {}
