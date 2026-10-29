//@ check-pass
//@ edition: 2024
//@ compile-flags: -Znext-solver

// Regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/283.
//
// `check_transmutes` must make the alias non-rigid when crossing the typing-mode boundary.

async fn get(_r: std::marker::PhantomData<i32>) {}
fn main() {
    let v = get(loop {});
    unsafe { v = std::mem::transmute(0_u8) }
}
