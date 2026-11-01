//! Regression test for <https://github.com/rust-lang/rust/issues/61083>.
//!
//! An associated type projection in a supertrait bound (`Bar<T>: Foo<T::Item>`)
//! failed to normalize when the `Bar` bound was reached through a trait object,
//! so passing the object to a function expecting `Foo<u32>` was rejected.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Foo<T> {}

trait Bar<T: Iterator>: Foo<T::Item> {}

fn a(_x: &(impl Foo<u32> + ?Sized)) {}

// The `dyn` form is the one that used to fail to normalize `T::Item` to `u32`.
fn b(y: &dyn Bar<std::vec::IntoIter<u32>>) {
    a(y)
}

// The equivalent `impl Trait` form always compiled; keep it so both paths stay pinned.
fn c(y: &(impl Bar<std::vec::IntoIter<u32>> + ?Sized)) {
    a(y)
}

fn main() {}
