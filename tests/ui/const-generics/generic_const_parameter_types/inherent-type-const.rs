//@ check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(
    min_generic_const_args,
    generic_const_parameter_types,
    inherent_associated_types,
    min_adt_const_params,
    const_param_ty_trait
)]

struct ThreeTypes<T1, T2, T3>(T1, T2, T3);

impl<T1, T2, T3: std::marker::ConstParamTy_> ThreeTypes<T1, T2, T3> {
    type const INHERENT: [T3; 0] = [];
}

struct Struct<const O: [u32; 0]>;

fn f() -> Struct<{ core::direct_const_arg!(ThreeTypes::<u8, u16, u32>::INHERENT) }> {
    Struct
}

fn main() {}
