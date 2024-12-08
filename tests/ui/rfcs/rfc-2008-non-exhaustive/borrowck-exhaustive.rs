// Test that the borrow checker doesn't consider checking an exhaustive pattern
// as an access.

//@ check-pass

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
    match x {
        ExhaustiveMonovariant::Variant(_) => {},
        _ => {},
    }
    drop(y);
    let mut x = Local::Variant(1);
    let y = &mut x;
    match x {
        Local::Variant(_) => {},
        _ => {},
    }
    drop(y);
    let mut x = LocalNonExhaustive::Variant(1);
    let y = &mut x;
    match x {
        LocalNonExhaustive::Variant(_) => {},
        _ => {},
    }
    drop(y);
}
