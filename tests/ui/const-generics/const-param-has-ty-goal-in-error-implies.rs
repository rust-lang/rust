// compile-flags: -Znext-solver

// Test for a weird diagnostics corner case. In the error reporting code, when reporting
// fulfillment errors for goals A and B, we try to see if elaborating A will result in
// another goal that can equate with B. That would signal that B is "implied by" A,
// allowing us to skip reporting it, which is beneficial for cutting down on the number
// of diagnostics we report. In the new trait solver especially, but even in the old trait
// solver through things like defining opaque type usages, this `can_equate` call was not
// properly taking the param-env of the goals, resulting in nested obligations that had
// empty param-envs. If one of these nested obligations was a `ConstParamHasTy` goal, then
// we would ICE, since those goals are particularly strict about the param-env they're
// evaluated in.

// This is morally a fix for <https://github.com/rust-lang/rust/issues/139314>, but that
// repro uses details about how defining usages in the `check_opaque_well_formed` code
// can spring out of type equality, and will likely stop failing soon coincidentally once
// we start using `PostBorrowck` mode in that check.

trait Foo: Baz<()> {}
trait Baz<T> {}

trait IdentityWithConstArgGoal<const N: usize> {
    type Assoc;
}
impl<T, const N: usize> IdentityWithConstArgGoal<N> for T {
    type Assoc = T;
}

fn unsatisfied<T, const N: usize>()
where
    T: Foo,
    T: Baz<<T as IdentityWithConstArgGoal<N>>::Assoc>,
{
}

fn test<const N: usize>() {
    unsatisfied::<(), N>();
    //~^ ERROR the trait bound `(): Foo` is not satisfied
}

fn main() {}
