//@ check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(
    gca_min_const_items,
    generic_const_parameter_types,
    inherent_associated_types,
    min_adt_const_params,
    const_param_ty_trait
)]

use std::gca;

struct ThreeTypes<T1, T2, T3>(T1, T2, T3);

impl<T1, T2, T3: std::marker::ConstParamTy_> ThreeTypes<T1, T2, T3> {
    const INHERENT: [T3; 0] = gca!([]);
}

struct Struct<const O: [u32; 0]>;

fn f() -> Struct<{ gca!(ThreeTypes::<u8, u16, u32>::INHERENT) }> {
    Struct
}

fn main() {}
