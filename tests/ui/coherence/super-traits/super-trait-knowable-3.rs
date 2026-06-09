// Unlike in `super-trait-knowable-1.rs`, the knowable
// super trait bound is in a nested goal so this would not
// compile if we were to only elaborate root goals.

//@ check-pass

trait Super {}
trait Sub<T>: Super {}

struct W<T>(T);
trait Bound<T> {}
impl<T: Sub<U>, U> Bound<W<U>> for T {}

trait Overlap<T> {}
impl<T, U: Bound<W<T>>> Overlap<T> for U {}
impl<T> Overlap<T> for () {}

fn main() {}
