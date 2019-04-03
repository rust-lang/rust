// compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc;

use rustc::ty::{self, Ty, TyKind};

#[deny(usage_of_ty_tykind)]
fn main() {
    let sty = TyKind::Bool; //~ ERROR usage of `ty::TyKind::<kind>`

    match sty {
        TyKind::Bool => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Char => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Int(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Uint(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Float(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Adt(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Foreign(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Str => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Array(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Slice(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::RawPtr(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Ref(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::FnDef(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::FnPtr(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Dynamic(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Closure(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Generator(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::GeneratorWitness(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Never => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Tuple(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Projection(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::UnnormalizedProjection(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Opaque(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Param(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Bound(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Placeholder(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Infer(..) => (), //~ ERROR usage of `ty::TyKind::<kind>`
        TyKind::Error => (), //~ ERROR usage of `ty::TyKind::<kind>`
    }

    if let ty::Int(int_ty) = sty {}

    if let TyKind::Int(int_ty) = sty {} //~ ERROR usage of `ty::TyKind::<kind>`

    fn ty_kind(ty_bad: TyKind<'_>, ty_good: Ty<'_>) {} //~ ERROR usage of `ty::TyKind`
}
