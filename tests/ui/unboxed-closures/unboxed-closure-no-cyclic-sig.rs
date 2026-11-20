//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Test that unboxed closures cannot capture their own type.
//
// Also regression test for issue #21410.

fn g<F>(_: F) where F: FnOnce(Option<F>) {}

fn main() {
    g(|_| {  }); //~ ERROR closure/coroutine type that references itself
}
