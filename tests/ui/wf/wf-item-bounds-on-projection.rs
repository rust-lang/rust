//@ check-pass
//@ compile-flags: -Znext-solver -Zstrict-projection-item-bounds

// If the rhs of a projection predicate is generic, then we imply item bounds of the associated
// types on it.

trait Trait {
    type Assoc: Copy;
}

trait Explicit {}
impl<T: Trait<Assoc = U>, U: Copy> Explicit for T {}

fn assert_explicit<T: Explicit>() {}
fn imply_copy<T: Trait<Assoc = U>, U>() {
    assert_explicit::<T>();
}


trait BoundA {}
trait HasBoundA {
    type AssocA: BoundA;
}

trait BoundB {}
trait HasBoundB {
    type AssocB: BoundB;
}

fn imply_both<T, U>()
where
    T: HasBoundA<AssocA = U> + HasBoundB<AssocB = U>,
{
    assert_both::<U>();
}

fn assert_both<T: BoundA + BoundB>() {}

fn imply_for_alias<T, U>()
where
    U: HasBoundA,
    T: HasBoundB<AssocB = U::AssocA>,
{
    assert_bound_b::<U::AssocA>();
}

fn assert_bound_b<T: BoundB>() {}

fn main() {}
