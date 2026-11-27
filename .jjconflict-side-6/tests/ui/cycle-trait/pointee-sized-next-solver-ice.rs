//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Regression test for https://github.com/rust-lang/rust/issues/151957
//
// When a user-defined trait shares the name `PointeeSized` with
// `core::marker::PointeeSized`, error reporting tries to check whether
// the type implements the lang-item `PointeeSized` trait via
// `predicate_must_hold_modulo_regions`. This creates a `PointeeSized`
// solver obligation which causes an ICE. We avoid this by skipping the
// `PointeeSized` lang item during the "similarly named trait" suggestion.

trait PointeeSized {}

fn require_trait<T: PointeeSized>() {}

fn main() {
    require_trait::<i32>();
    //~^ ERROR the trait bound `i32: PointeeSized` is not satisfied
}
