// Check that we correctly deduce object lifetime defaults inside self types of qualified paths.
//@ check-pass
//@ revisions: bound clause

#[cfg(bound)] trait Outer<'a>: 'a { type Ty; }
#[cfg(clause)] trait Outer<'a> where Self: 'a { type Ty; }
trait Inner {}

impl<'a> Outer<'a> for dyn Inner + 'a { type Ty = &'a (); }

fn f<'r>(x: <dyn Inner + 'r as Outer<'r>>::Ty) { g(x) }
// We deduce `dyn Inner + 'r` from bound `'a` on self ty param of trait `Outer`.
fn g<'r>(x: <dyn Inner as Outer<'r>>::Ty) {}

fn main() {}
