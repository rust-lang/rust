// FIXME: Explainer.
//@ known-bug: unknown

trait Outer { type Ty; }
trait Inner {}

impl<'a> Outer for dyn Inner + 'a { type Ty = &'a (); }

fn f<'r>(x: &'r <dyn Inner + 'static as Outer>::Ty) { g(x) }
// FIXME: Should infer `+ 'static`:
fn g<'r>(x: &'r <dyn Inner as Outer>::Ty) {}

fn main() {}
