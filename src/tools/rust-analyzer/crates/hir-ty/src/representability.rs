//! Detecting whether a type is infinitely-sized.

use hir_def::{AdtId, VariantId};
use rustc_type_ir::inherent::{AdtDef, IntoKind};

use crate::{
    db::HirDatabase,
    next_solver::{GenericArgKind, GenericArgs, Ty, TyKind},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Representability {
    Representable,
    Infinite,
}

macro_rules! rtry {
    ($e:expr) => {
        match $e {
            e @ Representability::Infinite => return e,
            Representability::Representable => {}
        }
    };
}

#[salsa::tracked(cycle_result = representability_cycle)]
pub(crate) fn representability(db: &dyn HirDatabase, id: AdtId) -> Representability {
    match id {
        AdtId::StructId(id) => variant_representability(db, id.into()),
        AdtId::UnionId(id) => variant_representability(db, id.into()),
        AdtId::EnumId(id) => {
            for &(variant, ..) in &id.enum_variants(db).variants {
                rtry!(variant_representability(db, variant.into()));
            }
            Representability::Representable
        }
    }
}

pub(crate) fn representability_cycle(
    _db: &dyn HirDatabase,
    _: salsa::Id,
    _id: AdtId,
) -> Representability {
    Representability::Infinite
}

fn variant_representability(db: &dyn HirDatabase, id: VariantId) -> Representability {
    for ty in db.field_types(id).values() {
        rtry!(representability_ty(db, ty.get().instantiate_identity()));
    }
    Representability::Representable
}

fn representability_ty<'db>(db: &'db dyn HirDatabase, ty: Ty<'db>) -> Representability {
    match ty.kind() {
        TyKind::Adt(adt_id, args) => representability_adt_ty(db, adt_id.def_id().0, args),
        // FIXME(#11924) allow zero-length arrays?
        TyKind::Array(ty, _) => representability_ty(db, ty),
        TyKind::Tuple(tys) => {
            for ty in tys {
                rtry!(representability_ty(db, ty));
            }
            Representability::Representable
        }
        _ => Representability::Representable,
    }
}

fn representability_adt_ty<'db>(
    db: &'db dyn HirDatabase,
    def_id: AdtId,
    args: GenericArgs<'db>,
) -> Representability {
    rtry!(representability(db, def_id));

    // At this point, we know that the item of the ADT type is representable;
    // but the type parameters may cause a cycle with an upstream type
    let params_in_repr = params_in_repr(db, def_id);
    for (i, arg) in args.iter().enumerate() {
        if let GenericArgKind::Type(ty) = arg.kind()
            && params_in_repr[i]
        {
            rtry!(representability_ty(db, ty));
        }
    }
    Representability::Representable
}

fn params_in_repr(db: &dyn HirDatabase, def_id: AdtId) -> Box<[bool]> {
    let generics = db.generic_params(def_id.into());
    let mut params_in_repr = (0..generics.len_lifetimes() + generics.len_type_or_consts())
        .map(|_| false)
        .collect::<Box<[bool]>>();
    let mut handle_variant = |variant| {
        for field in db.field_types(variant).values() {
            params_in_repr_ty(db, field.get().instantiate_identity(), &mut params_in_repr);
        }
    };
    match def_id {
        AdtId::StructId(def_id) => handle_variant(def_id.into()),
        AdtId::UnionId(def_id) => handle_variant(def_id.into()),
        AdtId::EnumId(def_id) => {
            for &(variant, ..) in &def_id.enum_variants(db).variants {
                handle_variant(variant.into());
            }
        }
    }
    params_in_repr
}

fn params_in_repr_ty<'db>(db: &'db dyn HirDatabase, ty: Ty<'db>, params_in_repr: &mut [bool]) {
    match ty.kind() {
        TyKind::Adt(adt, args) => {
            let inner_params_in_repr = self::params_in_repr(db, adt.def_id().0);
            for (i, arg) in args.iter().enumerate() {
                if let GenericArgKind::Type(ty) = arg.kind()
                    && inner_params_in_repr[i]
                {
                    params_in_repr_ty(db, ty, params_in_repr);
                }
            }
        }
        TyKind::Array(ty, _) => params_in_repr_ty(db, ty, params_in_repr),
        TyKind::Tuple(tys) => tys.iter().for_each(|ty| params_in_repr_ty(db, ty, params_in_repr)),
        TyKind::Param(param) => {
            params_in_repr[param.index as usize] = true;
        }
        _ => {}
    }
}
