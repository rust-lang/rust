// compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc_middle;

use rustc_middle::ty::{self, Ty, TyData};

#[deny(rustc::usage_of_ty_tydata)]
fn main() {
    let data = TyData::Bool; //~ ERROR usage of `ty::TyData::<data>`

    match data {
        TyData::Bool => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Char => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Int(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Uint(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Float(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Adt(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Foreign(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Str => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Array(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Slice(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::RawPtr(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Ref(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::FnDef(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::FnPtr(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Dynamic(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Closure(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Generator(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::GeneratorWitness(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Never => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Tuple(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Projection(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Opaque(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Param(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Bound(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Placeholder(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Infer(..) => (), //~ ERROR usage of `ty::TyData::<data>`
        TyData::Error(_) => (), //~ ERROR usage of `ty::TyData::<data>`
    }

    if let ty::Int(int_ty) = data {}

    if let TyData::Int(int_ty) = data {} //~ ERROR usage of `ty::TyData::<data>`

    fn ty_data(ty_bad: TyData<'_>, ty_good: Ty<'_>) {} //~ ERROR usage of `ty::TyData`
}
