// Test that the borrow checker considers `#[non_exhaustive]` when checking
// whether a match contains a discriminant read.

//@ aux-build:monovariants.rs
extern crate monovariants;

use monovariants::NonExhaustiveMonovariant;

#[non_exhaustive]
enum LocalNonExhaustive {
    Variant(u32),
}

fn main() {
    let mut x = NonExhaustiveMonovariant::Variant(1);
    let y = &mut x;
    match x {
        //~^ ERROR cannot use `x` because it was mutably borrowed
        NonExhaustiveMonovariant::Variant(_) => {},
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
