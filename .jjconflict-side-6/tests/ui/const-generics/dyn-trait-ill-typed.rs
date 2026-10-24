//! Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/296>.
//@ compile-flags: -Znext-solver=globally
//@ check-fail

// CHECK PASS TO SHOW IT PASSES, BUT IT SHOULD NOT
// THIS CODE SEGFAULTS, WITH REASON

//~v ERROR: the constant `M` is not of type `usize`
fn foo<const M: u32>() -> Box<dyn Tr<M>> {
    loop {}
}

trait Tr<const N: usize> {}

fn main() {}
