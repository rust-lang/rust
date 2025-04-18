// FIXME: Explainer
//@ known-bug: unknown

trait Outer<'a>: 'a { type Ty; }
trait Inner {}

impl<'a> Outer<'a> for dyn Inner + 'a { type Ty = &'a (); }

fn f<'r>(x: <dyn Inner + 'r as Outer<'r>>::Ty) { g(x) }
// FIXME: Should infer `+ 'r`:
fn g<'r>(x: <dyn Inner as Outer<'r>>::Ty) {}

fn main() {}
