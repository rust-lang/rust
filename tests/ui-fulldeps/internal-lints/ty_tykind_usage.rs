// compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc_middle;
extern crate rustc_type_ir;

use rustc_middle::ty::{self, Ty, TyKind};
use rustc_type_ir::{Interner, TyKind as IrTyKind};

#[deny(rustc::usage_of_ty_tykind)]
fn main() {
    let kind = TyKind::Bool; //~ ERROR usage of `ty::TyKind::<kind>`

    match kind {
        TyKind::Bool => (),                 //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Char => (),                 //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Int(..) => (),              //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Uint(..) => (),             //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Float(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Adt(..) => (),              //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Foreign(..) => (),          //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Str => (),                  //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Array(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Slice(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::RawPtr(..) => (),           //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Ref(..) => (),              //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::FnDef(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::FnPtr(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Dynamic(..) => (),          //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Closure(..) => (),          //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Generator(..) => (),        //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::GeneratorWitness(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::GeneratorWitnessMIR(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Never => (),                //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Tuple(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Alias(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Param(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Bound(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Placeholder(..) => (),      //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Infer(..) => (),            //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Error(_) => (),             //~ ERROR usage of `ty::TyKind::<kind>`
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
