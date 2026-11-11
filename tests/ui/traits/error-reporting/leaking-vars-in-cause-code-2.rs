//@ compile-flags: -Znext-solver

// The `cause` in `Obligation` is ignored by type folders. So infer vars in cause code is not
// fudged.
// Check the comments of
// `leaking-vars-in-cause-code-1.rs` for more details.
trait Trait<T> {}
struct A<T>(T);
struct B<T>(T);

trait IncompleteGuidance {}

impl<T> Trait<()> for A<T>
where
    T: IncompleteGuidance,
{
}

impl<T, U> Trait<()> for B<T>
//~^ ERROR: the type parameter `U` is not constrained by the impl trait, self type, or predicates
where
    A<T>: Trait<U>,
{
}

fn impls_trait<T: Trait<()>>() {}

fn main() {
    impls_trait::<B<()>>();
    //~^ ERROR: the trait bound `(): IncompleteGuidance` is not satisfied
}
