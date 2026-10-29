//@ compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc_middle;
extern crate rustc_type_ir;

use rustc_middle::ty::{self, Ty, TyKind};
use rustc_type_ir::{Interner, TyKind as IrTyKind};

#[deny(rustc::usage_of_ty_tykind)]
fn main() {
    let kind = TyKind::Bool; //~ ERROR usage of `ty::TyKind::<kind>`

    match kind {
        TyKind::Bool => {},                 //~ ERROR usage of `ty::TyKind::<kind>`
        _ => {}
    }

    if let ty::Int(int_ty) = kind {}

    if let TyKind::Int(int_ty) = kind {} //~ ERROR usage of `ty::TyKind::<kind>`

    fn ty_kind(ty_bad: TyKind<'_>, ty_good: Ty<'_>) {} //~ ERROR usage of `ty::TyKind`

    fn ir_ty_kind<I: Interner>(bad: IrTyKind<I>) -> IrTyKind<I> {
        //~^ ERROR usage of `ty::TyKind`
        //~| ERROR usage of `ty::TyKind`
        IrTyKind::Bool //~ ERROR usage of `ty::TyKind::<kind>`
    }
}
