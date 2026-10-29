// Companion to #157122 / `wf-dyn-in-hrtb-bound-issue-157122.rs`.
//
// That test covers an ICE in `WfPredicates::visit_ty` where a `dyn Trait` nested
// inside a `for<'a>` binder reached `ExistentialTraitRef::with_self_ty` with an
// escaping self type. The fix removes the offending `debug_assert!` rather than
// skipping the block. Skipping would have been a silent regression: the block
// emits the `ConstArgHasType` obligation that type-checks a `dyn`'s const
// argument, so dropping it makes an ill-typed const argument compile.
//
// Here `B` has type `bool` but `HasConst` expects `const N: usize`, and the
// `dyn HasConst<'a, B>` carries the escaping late-bound `'a`. This must still be
// rejected, exactly as it is without the surrounding `for<'a>` binder.

trait HasConst<'a, const N: usize> {}

fn nested<const B: bool>(_f: &dyn for<'a> Fn(&'a (), &'a dyn HasConst<'a, B>)) {}
//~^ ERROR the constant `B` is not of type `usize`

fn main() {}
