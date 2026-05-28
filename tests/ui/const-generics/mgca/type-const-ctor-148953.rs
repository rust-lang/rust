//! Regression test for <https://github.com/rust-lang/rust/issues/148953>
//!
//! Unit struct constructors used as the RHS of a `type const` associated
//! const used to ICE during normalization because they were lowered as
//! `Const::new_unevaluated` with a Ctor def_id. Fixed by adding proper const
//! constructor support that produces a concrete ValTree value instead.

//@ check-pass

#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
struct S;

impl S {
    type const N: S = S;
}

#[derive(ConstParamTy, PartialEq, Eq)]
enum E {
    V,
}

impl E {
    type const M: E = { E::V };
}

fn main() {}
