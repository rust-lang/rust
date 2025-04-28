// If we treat known inductive cycles as errors, this test compiles
// as normalizing `Overflow::Assoc<Overflow>` fails.
//
// As coherence already uses the new solver on stable, this change
// would require an FCP.

trait Trait {
    type Assoc<T: Trait>;
}

struct Overflow;
impl Trait for Overflow {
    type Assoc<T: Trait> = <T as Trait>::Assoc<Overflow>;
}

trait Overlap<T, WfHack> {}
impl<T: Trait, U: Copy> Overlap<T::Assoc<T>, U> for T {}
impl<U> Overlap<u32, U> for Overflow {}
//~^ ERROR conflicting implementations of trait `Overlap<<Overflow as Trait>::Assoc<Overflow>, _>` for type `Overflow`

fn main() {}
