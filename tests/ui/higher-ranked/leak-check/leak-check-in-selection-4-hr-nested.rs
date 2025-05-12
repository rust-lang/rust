//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// cc #119820. While the leak check does not consider the binder
// of the current goal, leaks from higher-ranked nested goals are
// considered.
//
// We enter and exit the binder of the nested goal while evaluating
// the candidate.

trait LeakCheckFailure<'a> {}
impl LeakCheckFailure<'static> for () {}

trait Trait<T> {}
impl Trait<u32> for () where for<'a> (): LeakCheckFailure<'a> {}
impl Trait<u16> for () {}
fn impls_trait<T: Trait<U>, U>() {}
fn main() {
    // ok
    //
    // It does not matter whether candidate assembly
    // considers the placeholders from higher-ranked goal.
    //
    // Either `for<'a> (): LeakCheckFailure<'a>` has no applicable
    // candidate or it has a single applicable candidate which then later
    // results in an error. This allows us to infer `U` to `u16`.
    impls_trait::<(), _>()
}
