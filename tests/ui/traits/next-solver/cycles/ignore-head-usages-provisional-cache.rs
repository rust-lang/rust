//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#232. We've
// previously incorrectly rebased provisional cache entries even if
// the cycle head didn't reach a fixpoint as it did not depend on any
// cycles itself.
//
// Just because the result of a goal does not depend on its own provisional
// result, it does not mean its nested goals don't depend on its result.
struct B;
struct C;
struct D;

pub trait Trait {
    type Output;
}
macro_rules! k {
    ($t:ty) => {
        <$t as Trait>::Output
    };
}

trait CallB<T1, T2> {
    type Output;
    type Return;
}

trait CallC<T1> {
    type Output;
    type Return;
}

trait CallD<T1, T2> {
    type Output;
}

fn foo<X, Y>()
where
    X: Trait,
    Y: Trait,
    D: CallD<k![X], k![Y]>,
    C: CallC<<D as CallD<k![X], k![Y]>>::Output>,
    <C as CallC<<D as CallD<k![X], k![Y]>>::Output>>::Output: Trait,
    B: CallB<
            <C as CallC<<D as CallD<k![X], k![Y]>>::Output>>::Return,
            <C as CallC<<D as CallD<k![X], k![Y]>>::Output>>::Output,
        >,
    <B as CallB<
        <C as CallC<<D as CallD<k![X], k![Y]>>::Output>>::Return,
        <C as CallC<<D as CallD<k![X], k![Y]>>::Output>>::Output,
    >>::Output: Trait<Output = ()>,
{
}
fn main() {}
