//@ compile-flags: -Z unstable-options

#![feature(rustc_attrs)]
#![feature(rustc_private)]
#![deny(rustc::pass_by_value)]
#![allow(unused)]

extern crate rustc_middle;

use rustc_middle::ty::{Ty, TyCtxt};

fn ty_by_ref(
    ty_val: Ty<'_>,
    ty_ref: &Ty<'_>, //~ ERROR passing `Ty<'_>` by reference
    ty_ctxt_val: TyCtxt<'_>,
    ty_ctxt_ref: &TyCtxt<'_>, //~ ERROR passing `TyCtxt<'_>` by reference
) {
}

fn ty_multi_ref(ty_multi: &&Ty<'_>, ty_ctxt_multi: &&&&TyCtxt<'_>) {}
//~^ ERROR passing `Ty<'_>` by reference
//~^^ ERROR passing `TyCtxt<'_>` by reference

trait T {
    fn ty_by_ref_in_trait(
        ty_val: Ty<'_>,
        ty_ref: &Ty<'_>, //~ ERROR passing `Ty<'_>` by reference
        ty_ctxt_val: TyCtxt<'_>,
        ty_ctxt_ref: &TyCtxt<'_>, //~ ERROR passing `TyCtxt<'_>` by reference
    );

    fn ty_multi_ref_in_trait(ty_multi: &&Ty<'_>, ty_ctxt_multi: &&&&TyCtxt<'_>);
    //~^ ERROR passing `Ty<'_>` by reference
    //~^^ ERROR passing `TyCtxt<'_>` by reference
}

struct Foo;

impl T for Foo {
    fn ty_by_ref_in_trait(
        ty_val: Ty<'_>,
        ty_ref: &Ty<'_>,
        ty_ctxt_val: TyCtxt<'_>,
        ty_ctxt_ref: &TyCtxt<'_>,
    ) {
    }

    fn ty_multi_ref_in_trait(ty_multi: &&Ty<'_>, ty_ctxt_multi: &&&&TyCtxt<'_>) {}
}

impl Foo {
    fn ty_by_ref_assoc(
        ty_val: Ty<'_>,
        ty_ref: &Ty<'_>, //~ ERROR passing `Ty<'_>` by reference
        ty_ctxt_val: TyCtxt<'_>,
        ty_ctxt_ref: &TyCtxt<'_>, //~ ERROR passing `TyCtxt<'_>` by reference
    ) {
    }

    fn ty_multi_ref_assoc(ty_multi: &&Ty<'_>, ty_ctxt_multi: &&&&TyCtxt<'_>) {}
    //~^ ERROR passing `Ty<'_>` by reference
    //~^^ ERROR passing `TyCtxt<'_>` by reference
}

#[rustc_pass_by_value]
enum CustomEnum {
    A,
    B,
}

impl CustomEnum {
    fn test(
        value: CustomEnum,
        reference: &CustomEnum, //~ ERROR passing `CustomEnum` by reference
    ) {
    }
}

#[rustc_pass_by_value]
struct CustomStruct {
    s: u8,
}

#[rustc_pass_by_value]
type CustomAlias<'a> = &'a CustomStruct; //~ ERROR passing `CustomStruct` by reference

impl CustomStruct {
    fn test(
        value: CustomStruct,
        reference: &CustomStruct, //~ ERROR passing `CustomStruct` by reference
    ) {
    }

    fn test_alias(
        value: CustomAlias,
        reference: &CustomAlias, //~ ERROR passing `CustomAlias<'_>` by reference
    ) {
    }
}

#[rustc_pass_by_value]
struct WithParameters<T, const N: usize, M = u32> {
    slice: [T; N],
    m: M,
}

impl<T> WithParameters<T, 1> {
    fn test<'a>(
        value: WithParameters<T, 1>,
        reference: &'a WithParameters<T, 1>, //~ ERROR passing `WithParameters<T, 1>` by reference
        reference_with_m: &WithParameters<T, 1, u32>, //~ ERROR passing `WithParameters<T, 1, u32>` by reference
    ) -> &'a WithParameters<T, 1> {
        //~^ ERROR passing `WithParameters<T, 1>` by reference
        reference as &WithParameters<_, 1> //~ ERROR passing `WithParameters<_, 1>` by reference
    }
}

fn main() {}
