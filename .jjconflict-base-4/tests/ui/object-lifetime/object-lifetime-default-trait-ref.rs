// Check that trait refs correctly induce trait object lifetime defaults.
//
// For the longest time, all of the cases below used to get
// rejected as "inderminate" due to an off-by-one error.

//@ check-pass

trait Outer<'a, T: 'a + AbideBy<'a> + ?Sized> { type Ty where Self: Sized; }
impl<'a, T: 'a + AbideBy<'a> + ?Sized> Outer<'a, T> for () { type Ty = &'a T; }

trait Inner {}

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn bound<'r, T>() where T: Outer<'r, dyn Inner> {}

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn dyn_trait<'r>(_: Box<dyn Outer<'r, dyn Inner>>) {}
// Wfck doesn't require the type arguments in dyn trait to meet the bounds declared in the trait.
// Thus we can't rely on `AbideBy` here. Instead, just check if `'r` outlives the implicit bound.
fn check_dyn_trait<'r>(x: Box<dyn Outer<'r, dyn Inner + 'r>>) { dyn_trait(x) }

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn impl_trait<'r>(_: impl Outer<'r, dyn Inner>) {}

// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn assoc_ty_proj<'r>(_: <() as Outer<'r, dyn Inner>>::Ty) {}

// We use this to test that a given trait object lifetime bound is
// *exactly equal* to a given lifetime (not longer, not shorter).
trait AbideBy<'a> {}
impl<'a> AbideBy<'a> for dyn Inner + 'a {}

fn main() {}
