//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
// Test from trait-system-refactor-initiative#241:
// Used to ICE in mir typeck because of ambiguity in the new solver.
// The wrong (first) trait bound was selected.
// This is fixed with new logic for unsizing coercions
// that's independent from that of the old solver, which this test verifies.

trait Super<'a> {}
trait Trait<'a>: Super<'a> + for<'hr> Super<'hr> {}
//[current]~^ ERROR type annotations needed: cannot satisfy `Self: Super<'a>`

fn foo<'a>(x: Box<dyn Trait<'a>>) -> Box<dyn Super<'a>> {
    x
}

fn main() {}
