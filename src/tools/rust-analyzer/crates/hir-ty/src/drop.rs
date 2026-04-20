//! Utilities for computing drop info about types.

use hir_def::{
    AdtId, ImplId,
    signatures::{StructFlags, StructSignature},
};
use rustc_hash::FxHashSet;
use rustc_type_ir::inherent::{AdtDef, GenericArgs as _, IntoKind};
use stdx::never;

use crate::{
    consteval,
    db::HirDatabase,
    method_resolution::TraitImpls,
    next_solver::{
        DbInterner, ParamEnv, SimplifiedType, Ty, TyKind,
        infer::{InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

#[salsa::tracked]
pub fn destructor(db: &dyn HirDatabase, adt: AdtId) -> Option<ImplId> {
    let module = match adt {
        AdtId::EnumId(id) => db.lookup_intern_enum(id).container,
        AdtId::StructId(id) => db.lookup_intern_struct(id).container,
        AdtId::UnionId(id) => db.lookup_intern_union(id).container,
    };
    let interner = DbInterner::new_with(db, module.krate(db));
    let drop_trait = interner.lang_items().Drop?;
    let impls = match module.block(db) {
        Some(block) => match TraitImpls::for_block(db, block) {
            Some(it) => &**it,
            None => return None,
        },
        None => TraitImpls::for_crate(db, module.krate(db)),
    };
    impls.for_trait_and_self_ty(drop_trait, &SimplifiedType::Adt(adt.into())).0.first().copied()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DropGlue {
    // Order of variants is important.
    None,
    /// May have a drop glue if some type parameter has it.
    ///
    /// For the compiler this is considered as a positive result, IDE distinguishes this from "yes".
    DependOnParams,
    HasDropGlue,
}

pub fn has_drop_glue<'db>(infcx: &InferCtxt<'db>, ty: Ty<'db>, env: ParamEnv<'db>) -> DropGlue {
    has_drop_glue_impl(infcx, ty, env, &mut FxHashSet::default())
}

fn has_drop_glue_impl<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    env: ParamEnv<'db>,
    visited: &mut FxHashSet<Ty<'db>>,
) -> DropGlue {
    let mut ocx = ObligationCtxt::new(infcx);
    let ty = ocx.structurally_normalize_ty(&ObligationCause::dummy(), env, ty).unwrap_or(ty);

    if !visited.insert(ty) {
        // Recursive type.
        return DropGlue::None;
    }

    let db = infcx.interner.db;
    match ty.kind() {
        TyKind::Adt(adt_def, subst) => {
            let adt_id = adt_def.def_id().0;
            if adt_def.destructor(infcx.interner).is_some() {
                return DropGlue::HasDropGlue;
            }
            match adt_id {
                AdtId::StructId(id) => {
                    if StructSignature::of(db, id)
                        .flags
                        .intersects(StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA)
                    {
                        return DropGlue::None;
                    }
                    db.field_types(id.into())
                        .iter()
                        .map(|(_, field_ty)| {
                            has_drop_glue_impl(
                                infcx,
                                field_ty.get().instantiate(infcx.interner, subst),
                                env,
                                visited,
                            )
                        })
                        .max()
                        .unwrap_or(DropGlue::None)
                }
                // Unions cannot have fields with destructors.
                AdtId::UnionId(_) => DropGlue::None,
                AdtId::EnumId(id) => id
                    .enum_variants(db)
                    .variants
                    .iter()
                    .map(|&(variant, _, _)| {
                        db.field_types(variant.into())
                            .iter()
                            .map(|(_, field_ty)| {
                                has_drop_glue_impl(
                                    infcx,
                                    field_ty.get().instantiate(infcx.interner, subst),
                                    env,
                                    visited,
                                )
                            })
                            .max()
                            .unwrap_or(DropGlue::None)
                    })
                    .max()
                    .unwrap_or(DropGlue::None),
            }
        }
        TyKind::Tuple(tys) => tys
            .iter()
            .map(|ty| has_drop_glue_impl(infcx, ty, env, visited))
            .max()
            .unwrap_or(DropGlue::None),
        TyKind::Array(ty, len) => {
            if consteval::try_const_usize(db, len) == Some(0) {
                // Arrays of size 0 don't have drop glue.
                return DropGlue::None;
            }
            has_drop_glue_impl(infcx, ty, env, visited)
        }
        TyKind::Slice(ty) => has_drop_glue_impl(infcx, ty, env, visited),
        TyKind::Closure(_, args) => {
            has_drop_glue_impl(infcx, args.as_closure().tupled_upvars_ty(), env, visited)
        }
        TyKind::Coroutine(_, args) => {
            has_drop_glue_impl(infcx, args.as_coroutine().tupled_upvars_ty(), env, visited)
        }
        TyKind::CoroutineClosure(_, args) => {
            has_drop_glue_impl(infcx, args.as_coroutine_closure().tupled_upvars_ty(), env, visited)
        }
        // FIXME: Coroutine witness.
        TyKind::CoroutineWitness(..) => DropGlue::None,
        TyKind::Ref(..)
        | TyKind::RawPtr(..)
        | TyKind::FnDef(..)
        | TyKind::Str
        | TyKind::Never
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::FnPtr(..)
        | TyKind::Foreign(_)
        | TyKind::Error(_)
        | TyKind::Bound(..)
        | TyKind::Placeholder(..) => DropGlue::None,
        TyKind::Dynamic(..) => DropGlue::HasDropGlue,
        TyKind::Alias(..) => {
            if infcx.type_is_copy_modulo_regions(env, ty) {
                DropGlue::None
            } else {
                DropGlue::HasDropGlue
            }
        }
        TyKind::Param(_) => {
            if infcx.type_is_copy_modulo_regions(env, ty) {
                DropGlue::None
            } else {
                DropGlue::DependOnParams
            }
        }
        TyKind::Infer(..) => unreachable!("inference vars shouldn't exist out of inference"),
        TyKind::Pat(..) | TyKind::UnsafeBinder(..) => {
            never!("we do not handle pattern and unsafe binder types");
            DropGlue::None
        }
    }
}
