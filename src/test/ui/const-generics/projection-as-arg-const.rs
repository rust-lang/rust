// run-pass

#![allow(incomplete_features)] // This is needed for the `adt_const_params` feature.
#![feature(associated_type_defaults)]

pub trait Identity: Sized {
    type Identity = Self;
    // ^ This is needed because otherwise we get:
    // error: internal compiler error: compiler/rustc_hir_analysis/src/collect/type_of.rs:271:17: \
    // associated type missing default
    //  --> test.rs:6:5
    //   |
    // 6 |     type Identity;
    //   |     ^^^^^^^^^^^^^^
}

impl<T> Identity for T {}

pub fn foo<const X: <i32 as Identity>::Identity>() {
    assert!(X == 12);
}

fn main() {
    foo::<12>();
}
