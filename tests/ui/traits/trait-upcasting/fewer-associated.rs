//@ check-pass
// issue: 114035
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait A: B {
    type Assoc;
}

trait B {}

fn upcast(a: &dyn A<Assoc = i32>) -> &dyn B {
    a
}

// Make sure that we can drop the existential projection `A::Assoc = i32`
// when upcasting `dyn A<Assoc = i32>` to `dyn B`. Before, we used some
// complicated algorithm which required rebuilding a new object type with
// different bounds in order to test that an upcast was valid, but this
// didn't allow upcasting to t that have fewer associated types
// than the source type.

fn main() {}
