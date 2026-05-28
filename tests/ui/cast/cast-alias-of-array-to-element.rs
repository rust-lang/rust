//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/203>.
// Test that we structually normalize in the hacky `&[T; N] -> *const T` in cast.

trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}

struct W<'a>(&'a <[f32; 0] as Mirror>::Assoc);

fn foo(x: W<'_>) -> *const f32 {
    x.0 as *const f32
}

fn main() {}
