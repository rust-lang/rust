//@ compile-flags: -Znext-solver
//@ check-pass

// Generalizing a projection containing an inference variable
// which cannot be named by the `root_vid` previously resulted in ambiguity.
//
// Because we do not decrement the universe index when exiting a forall,
// this can cause unexpected failures.
//
// See generalize-proj-new-universe-index-1.rs for more details.

// For this reproduction we need:
// - an inference variable with a lower universe
// - enter a binder to increment the current universe
// - create a new inference variable which is constrained by proving a goal
// - equate a projection containing the new variable with the first variable
// - generalization creates yet another inference variable which is then
//   part of an alias-relate, resulting this to fail with ambiguity.
//
// Because we need to enter the binder in-between the creation of the first
// and second inference variable, this is easiest via
// `assemble_candidates_after_normalizing_self_ty` because eagerly call
// `try_evaluate_added_goals` there before creating the inference variables
// for the impl parameters.
trait Id {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Id for T {
    type Assoc = T;
}

// By adding an higher ranked bound to the impl we currently
// propagate this bound to the caller, forcing us to create a new
// universe.
trait IdHigherRankedBound {
    type Assoc: ?Sized;
}

impl<T: ?Sized> IdHigherRankedBound for T
where
    for<'a> T: 'a,
{
    type Assoc = T;
}

trait WithAssoc<T: ?Sized> {
    type Assoc: ?Sized;
}


struct Leaf;
struct Wrapper<U: ?Sized>(U);
struct Rigid;

impl<U: ?Sized> WithAssoc<U> for Leaf {
    type Assoc = U;
}


impl<Ur: ?Sized> WithAssoc<Wrapper<Ur>> for Rigid
where
    Leaf: WithAssoc<Ur>,
{
    type Assoc = <<Leaf as WithAssoc<Ur>>::Assoc as Id>::Assoc;
}

fn bound<T: ?Sized, U: ?Sized, V: ?Sized>()
where
    T: WithAssoc<U, Assoc = V>,
{
}

fn main() {
    bound::<<Rigid as IdHigherRankedBound>::Assoc, <Wrapper<Leaf> as Id>::Assoc, _>()
}
