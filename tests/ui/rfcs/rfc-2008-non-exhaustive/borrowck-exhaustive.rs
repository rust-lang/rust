// Test that the borrow checker considers checking an exhaustive pattern
// to be an access.

#![allow(dropping_references)]

//@ aux-build:monovariants.rs
extern crate monovariants;

use monovariants::ExhaustiveMonovariant;

enum Local {
    Variant(u32),
}

#[non_exhaustive]
enum LocalNonExhaustive {
    Variant(u32),
}

fn main() {
    let mut x = ExhaustiveMonovariant::Variant(1);
    let y = &mut x;
    match x { //~ ERROR cannot use `x` because it was mutably borrowed
        ExhaustiveMonovariant::Variant(_) => {},
        _ => {},
    }
    drop(y);
    let mut x = Local::Variant(1);
    let y = &mut x;
    match x { //~ ERROR cannot use `x` because it was mutably borrowed
        Local::Variant(_) => {},
        _ => {},
    }
    drop(y);
    let mut x = LocalNonExhaustive::Variant(1);
    let y = &mut x;
    match x { //~ ERROR cannot use `x` because it was mutably borrowed
        LocalNonExhaustive::Variant(_) => {},
        _ => {},
    }
    drop(y);
}
