// Properly deduce the object lifetime default in generic associated type (GAT) *paths*.
// issue: rust-lang/rust#115379
//@ check-pass

trait Outer {
    type Ty<'a, T: ?Sized + 'a>;
}
impl Outer for () {
    type Ty<'a, T: ?Sized + 'a> = &'a T;
}
trait Inner {}

fn f<'r>(x: <() as Outer>::Ty<'r, dyn Inner + 'r>) { g(x) }
fn g<'r>(_: <() as Outer>::Ty<'r, dyn Inner>) {}

fn main() {}
