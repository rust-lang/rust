// Check that resolved associated type paths induce the correct
// trait object lifetime default for the self type.

//@ check-pass
//@ revisions: bound clause

// RBV works on the HIR where where-clauses and item bounds of traits aren't merged yet.
// It's therefore wise to check both forms and make sure both are treated the same by RBV.
#[cfg(bound)] trait Outer<'a>: 'a { type Ty; }
#[cfg(clause)] trait Outer<'a> where Self: 'a { type Ty; }
trait Inner {}

impl<'a> Outer<'a> for dyn Inner + 'a { type Ty = &'a (); }

fn f<'r>(x: <dyn Inner + 'r as Outer<'r>>::Ty) { /*check*/ g(x) }
// We deduce `dyn Inner + 'r` from bound `'a` on self ty param of trait `Outer`.
fn g<'r>(x: <dyn Inner as Outer<'r>>::Ty) {}

fn h<'r>(x: <dyn Inner + 'r as Outer<'r>>::Ty) { /*check*/ i(x) }
// Just like the case directly above, we elaborate `dyn Inner` to `dyn Inner + 'r`.
// What's different is the extra segment (`self`) that once threw off RBV in a dev version.
fn i<'r>(x: <dyn Inner as self::Outer<'r>>::Ty) {}

fn main() {}
