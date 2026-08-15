//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ check-pass

// Make sure that we support non-defining uses in borrowck.
// Regression test for https://github.com/rust-lang/rust/issues/145748.

pub fn f(_: &()) -> impl Fn() + '_ {
    || {
        let _ = f(&());
    }
}

fn main() {}
