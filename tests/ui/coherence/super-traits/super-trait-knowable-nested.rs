// Unlike in `super-trait-knowable-1.rs`, the knowable
// super trait bound is in a nested goal and we currently
// only elaborate in the root. This can, and should, be#
// changed in the future.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Super {}
trait Sub<T>: Super {}

trait Bound<T> {}

impl<T: Sub<T>, U> Bound<U> for T {}

trait Overlap<T> {}
impl<T, U: Bound<T>> Overlap<T> for U {}
impl<T> Overlap<T> for () {}
//~^ ERROR conflicting implementations of trait `Overlap<_>` for type `()`

fn main() {}
