//@ known-bug: unknown

trait Outer<'a>: 'a { type Ty; }
trait Inner {}

impl<'a> Outer<'a> for dyn Inner + 'a { type Ty = &'a (); }

fn f<'r>(x: <dyn Inner + 'r as Outer<'r>>::Ty) { /*check*/ g(x) }
// FIXME: Deduce `dyn Inner + 'r` from bound `'a` on self ty param of trait `Outer`.
fn g<'r>(x: <dyn Inner as Outer<'r>>::Ty) {}

fn main() {}
