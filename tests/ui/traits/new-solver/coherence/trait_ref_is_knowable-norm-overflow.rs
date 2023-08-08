// compile-flags: -Ztrait-solver=next

// Coherence should handle overflow while normalizing for
// `trait_ref_is_knowable` correctly.

// FIXME(-Ztrait-solver=next-coherence): This currently has unstable query results.
// Figure out how to deal with this.

trait Overflow {
    type Assoc;
}
impl<T> Overflow for T {
    type Assoc = <T as Overflow>::Assoc;
}


trait Trait {}
impl<T: Copy> Trait for T {}
struct LocalTy;
impl Trait for <LocalTy as Overflow>::Assoc {}
//~^ ERROR conflicting implementations of trait `Trait` for type `<LocalTy as Overflow>::Assoc`

fn main() {}
