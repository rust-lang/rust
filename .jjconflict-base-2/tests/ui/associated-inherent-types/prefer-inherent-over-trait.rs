#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Ensure that prefer inherent associated types over trait associated types
// (assuming the impl headers match and they're accessible).
//@ check-pass

struct Adt;

impl Adt {
    type Ty = ();
}

trait Trait {
    type Ty;
    fn scope();
}

impl Trait for Adt {
    type Ty = i32;
    fn scope() {
        // We prefer the inherent assoc ty `Adt::Ty` (`()`) over the
        // trait assoc ty `<Adt as Trait>::Ty` (`i32`).
        let (): Self::Ty;
    }
}

fn main() {}
