// known-bug

// This should compile but fails with the current solver.
//
// This checks that the new solver uses `Ambiguous` when hitting the
// inductive cycle here when proving `exists<^0, ^1> (): Trait<^0, ^1>`
// which requires proving `Trait<?1, ?0>` but that has the same
// canonical representation.
trait Trait<T, U> {}

impl<T, U> Trait<T, U> for ()
where
    (): Trait<U, T>,
    T: OtherTrait,
{}

trait OtherTrait {}
impl OtherTrait for u32 {}

fn require_trait<T, U>()
where
    (): Trait<T, U>
{}

fn main() {
    require_trait::<_, _>();
    //~^ ERROR overflow evaluating
}
