//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] check-pass
// Test from trait-system-refactor-initiative#241:
// Used to ICE in mir typeck because of ambiguity in the new solver.
// The wrong (first) trait bound was selected.
// This is fixed with new logic for unsizing coercions
// that's independent from that of the old solver, which this test verifies.

trait Super<T> {}
trait Trait<T>: Super<T> + for<'hr> Super<&'hr ()> {}

fn foo<'a>(x: Box<dyn Trait<&'a ()>>) -> Box<dyn Super<&'a ()>> {
    x
    //[next]~^ ERROR type annotations needed: cannot satisfy `dyn Trait<&()>: Unsize<dyn Super<&()>>`
}

fn main() {}
