//@ compile-flags: -Znext-solver

// Coherence should handle overflow while normalizing for
// `trait_ref_is_knowable` correctly.

trait Overflow {
    type Assoc;
}
impl<T> Overflow for T {
    type Assoc = <T as Overflow>::Assoc;
    //~^ ERROR: overflow
}


trait Trait {}
impl<T: Copy> Trait for T {}
struct LocalTy;
impl Trait for <LocalTy as Overflow>::Assoc {}
//~^ ERROR conflicting implementations of trait `Trait`

fn main() {}
