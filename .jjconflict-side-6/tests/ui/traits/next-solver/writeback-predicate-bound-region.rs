//@ edition: 2024
//@ check-pass
//@ compile-flags: -Znext-solver

// This previously ICE'd during writeback when resolving
// the stalled coroutine predicate due to its bound lifetime.

trait Trait<'a> {}
impl<'a, T: Send> Trait<'a> for T {}

fn is_trait<T: for<'a> Trait<'a>>(_: T) {}
fn main() {
    is_trait(async {})
}
