//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: --emit=mir
//@[next] compile-flags: -Znext-solver=globally

#![feature(gca_adts, gca_macroless_items, gca_min_const_items, min_adt_const_params)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct NoDerive(i32);

#[derive(PartialEq, Eq, ConstParamTy)]
struct WrapInline(NoDerive);

const WRAP_DIRECT_INLINE: WrapInline = WrapInline(NoDerive(NoDerive));
//~^ ERROR the constant `NoDerive` is not of type `i32`

impl WrapInline {
    const BAD: WrapInline = WrapInline(NoDerive(NoDerive));
    //~^ ERROR the constant `NoDerive` is not of type `i32`
}

trait Tr {
    #[rustc_always_gca]
    const C: WrapInline;
}

impl Tr for () {
    const C: WrapInline = WrapInline(NoDerive(NoDerive));
    //~^ ERROR the constant `NoDerive` is not of type `i32`
}

const COPIED: WrapInline = const { WRAP_DIRECT_INLINE };

fn main() {
    free_constant();
    inherent_constant();
    trait_constant();
    let _ = COPIED;
}

fn free_constant() {
    match WRAP_DIRECT_INLINE {
        _ => {}
    }
}

fn inherent_constant() {
    match WrapInline::BAD {
        _ => {}
    }
}

fn trait_constant() {
    match <() as Tr>::C {
        _ => {}
    }
}
