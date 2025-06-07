// Properly deduce the object lifetime default in type-relative generic associated type *paths*.
// issue: <https://github.com/rust-lang/rust/issues/115379>
//@ check-pass

trait Outer { type Ty<'a, T: 'a + ?Sized>; }
trait Inner {}

fn f<'r, T: Outer>(x: T::Ty<'r, dyn Inner + 'r>) { g::<T>(x) }
// Deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`
fn g<'r, T: Outer>(x: T::Ty<'r, dyn Inner>) {}

fn main() {}
