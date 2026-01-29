//@ known-bug: unknown

trait Outer { type Ty; }
trait Inner {}

impl<'a> Outer for dyn Inner + 'a { type Ty = &'a (); }

// FIXME: Deduce `dyn Inner + 'static` from absence of any bounds on self ty param of trait `Outer`.
fn f<'r>(x: &'r <dyn Inner as Outer>::Ty) { /*check*/ g(x) }
fn g<'r>(x: &'r <dyn Inner + 'static as Outer>::Ty) {}

fn main() {}
