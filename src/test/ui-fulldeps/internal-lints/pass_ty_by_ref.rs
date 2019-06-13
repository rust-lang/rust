// compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(ty_pass_by_reference)]
#![allow(unused)]

extern crate rustc;

use rustc::ty::{Ty, TyCtxt};

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

fn main() {}
