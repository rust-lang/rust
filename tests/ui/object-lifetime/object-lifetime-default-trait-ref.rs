// Check that we correctly deduce object lifetime defaults inside *trait refs*!
// For the longest time these examples used to get rejected as "inderminate" due to an off-by-one.
//@ check-pass

trait Inner {}
trait Outer<'a, T: 'a + ?Sized> { type Project where Self: Sized; }

fn bound0<'r, T>() where T: Outer<'r, dyn Inner + 'r> { bound1::<'r, T>() }
// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn bound1<'r, T>() where T: Outer<'r, dyn Inner> {}

fn dyn0<'r>(x: Box<dyn Outer<'r, dyn Inner + 'r>>) { dyn1(x) }
// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn dyn1<'r>(_: Box<dyn Outer<'r, dyn Inner>>) {}

fn impl0<'r>(x: impl Outer<'r, dyn Inner + 'r>) { impl1(x) }
// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn impl1<'r>(_: impl Outer<'r, dyn Inner>) {} // deduce `dyn Inner + 'r`

fn proj<'r>(x: <() as Outer<'r, dyn Inner + 'r>>::Project) { proj1(x) }
// We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of trait `Outer`.
fn proj1<'r>(_: <() as Outer<'r, dyn Inner>>::Project) {}

impl<'a, T: 'a + ?Sized> Outer<'a, T> for () { type Project = &'a T; }

fn main() {}
