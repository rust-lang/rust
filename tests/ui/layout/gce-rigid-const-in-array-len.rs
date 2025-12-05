//! With `feature(generic_const_exprs)`, anon consts (e.g. length in array types) will
//! inherit their parent's predicates. When combined with `feature(trivial_bounds)`, it
//! is possible to have an unevaluated constant that is rigid, but not generic.
//!
//! This is what happens below: `u8: A` does not hold in the global environment, but
//! with trivial bounds + GCE it it possible that `<u8 as A>::B` can appear in an array
//! length without causing a compile error. This constant is *rigid* (i.e. it cannot be
//! normalized further), but it is *not generic* (i.e. it does not depend on any generic
//! parameters).
//!
//! This test ensures that we do not ICE in layout computation when encountering such a
//! constant.

#![feature(rustc_attrs)]
#![feature(generic_const_exprs)] //~ WARNING: the feature `generic_const_exprs` is incomplete
#![feature(trivial_bounds)]

#![crate_type = "lib"]

trait A {
    const B: usize;
}

#[rustc_layout(debug)]
struct S([u8; <u8 as A>::B]) //~ ERROR: the type `[u8; <u8 as A>::B]` has an unknown layout
where
    u8: A;
