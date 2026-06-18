//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/158064>

trait Trait {
    fn bar() -> impl Clone {
        1
    }
}

fn main() {}
