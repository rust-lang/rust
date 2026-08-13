//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for github.com/rust-lang/trait-system-refactor-initiative#280
// We force unresolved float infer vars to fallback to `f32` if there're stalled `f32: From<?float>`
// obligations.
// Previously the recursion limit is 3 which is not enough, causing some bevy crates to fail.

trait Trait {}
impl<T: Into<f32>> Trait for T {}

struct W<T>(T);
impl<T: Trait> Trait for W<T> {}

fn impls_trait<T: Trait>(_: T) {}

fn main() {
    impls_trait(W(1.0))
    //~^ WARN: falling back to `f32` as the trait bound `f32: From<f64>` is not satisfied [float_literal_f32_fallback]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
