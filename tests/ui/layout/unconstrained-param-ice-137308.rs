//! Regression test for <https://github.com/rust-lang/rust/issues/137308>.
//!
//! This used to ICE in layout computation, because `<u8 as A>::B` fails to normalize
//! due to the unconstrained param on the impl.

#![feature(rustc_attrs, sized_hierarchy)]
#![crate_type = "lib"]

use std::marker::PointeeSized;

trait A {
    const B: usize;
}

impl<C: PointeeSized> A for u8 { //~ ERROR: the type parameter `C` is not constrained
    const B: usize = 42;
}

#[rustc_layout(debug)]
struct S([u8; <u8 as A>::B]); //~ ERROR: the type has an unknown layout
