//! ICE test #124348
//! We should not be running const eval if the layout has errors.

enum Eek {
    TheConst,
    UnusedByTheConst(Sum),
    //~^ ERROR cannot find type `Sum` in this scope
}

const EEK_ZERO: &[Eek] = &[];

fn main() {}
