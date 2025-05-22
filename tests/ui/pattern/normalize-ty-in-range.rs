//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/200>.
// Make sure we structurally normalize in range pattern checking in HIR typeck.

trait Foo {
    type Bar;
}

impl Foo for () {
    type Bar = i32;
}

fn main() {
    const X: <() as Foo>::Bar = 0;

    match 0 {
        X..=X => {}
        _ => {}
    }
}
