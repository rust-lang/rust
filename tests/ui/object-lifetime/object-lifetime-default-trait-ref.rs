// FIXME: Explainer (used to fail: "inderminate" due to an off by one :^))
//@ check-pass

trait Inner {}
trait Outer<'a, T: 'a + ?Sized> {}

fn bound0<'r, T>() where T: Outer<'r, dyn Inner + 'r> { bound1::<'r, T>() }
fn bound1<'r, T>() where T: Outer<'r, dyn Inner> {}

fn dyn0<'r>(x: Box<dyn Outer<'r, dyn Inner + 'r>>) { dyn1(x) }
fn dyn1<'r>(_: Box<dyn Outer<'r, dyn Inner>>) {}

fn impl0<'r>(x: impl Outer<'r, dyn Inner + 'r>) { impl1(x) }
fn impl1<'r>(_: impl Outer<'r, dyn Inner>) {}

fn main() {}
