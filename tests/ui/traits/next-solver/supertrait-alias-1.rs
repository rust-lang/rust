//@ compile-flags: -Znext-solver
//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/171>.
// Tests that we don't try to replace `<V as Super>::Output` when replacing projections in the
// required bounds for `dyn Trait`, b/c `V` is not relevant to the dyn type, which we were
// previously encountering b/c we were walking into the existential projection bounds of the dyn
// type itself.

pub trait Trait: Super {}

pub trait Super {
    type Output;
}

fn bound<T: Trait>() {}

fn visit_simd_operator<V: Super>() {
    bound::<dyn Trait<Output = <V as Super>::Output>>();
}

fn main() {}
