// Added in #124532. While `(): Super` is knowable, `(): Sub<?t>` is not.
//
// We therefore elaborate super trait bounds in the implicit negative
// overlap check.

trait Super {}
trait Sub<T>: Super {}

trait Overlap<T> {}
impl<T, U: Sub<T>> Overlap<T> for U {}
impl<T> Overlap<T> for () {}
//~^ ERROR conflicting implementations of trait `Overlap<_>` for type `()`

fn main() {}
