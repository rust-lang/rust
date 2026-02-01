// Check that we correctly deduce object lifetime defaults inside self types of qualified paths.

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

fn main() {}
