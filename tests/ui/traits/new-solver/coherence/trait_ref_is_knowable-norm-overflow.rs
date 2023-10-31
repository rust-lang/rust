// compile-flags: -Ztrait-solver=next

// Coherence should handle overflow while normalizing for
// `trait_ref_is_knowable` correctly.

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
