// A regression test for #129444. This previously ICE'd as
// computing the best obligation for one ambiguous obligation
// added spurious inference constraints which caused another
// candidate to pass.
trait Trait {
    type Assoc;
}

struct W<T: Trait>(*mut T);
impl<T> Trait for W<W<W<T>>> {}
//~^ ERROR the trait bound `W<W<T>>: Trait` is not satisfied
//~| ERROR the trait bound `W<T>: Trait` is not satisfied
//~| ERROR the trait bound `T: Trait` is not satisfied
//~| ERROR not all trait items implemented, missing: `Assoc`

trait NoOverlap {}
impl<T: Trait> NoOverlap for T {}
impl<T: Trait<Assoc = u32>> NoOverlap for W<T> {}
//~^ ERROR conflicting implementations of trait `NoOverlap` for type `W<W<W<W<_>>>>`
fn main() {}
