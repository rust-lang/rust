//! Utilities for computing drop info about types.

use hir_def::{AdtId, lang_item::LangItem, signatures::StructFlags};
use rustc_hash::FxHashSet;
use rustc_type_ir::inherent::{AdtDef, IntoKind, SliceLike};
use stdx::never;
use triomphe::Arc;

use crate::{
    TraitEnvironment, consteval,
    db::HirDatabase,
    method_resolution::TyFingerprint,
    next_solver::{
        Ty, TyKind,
        infer::{InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

fn has_destructor(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let module = match adt {
        AdtId::EnumId(id) => db.lookup_intern_enum(id).container,
        AdtId::StructId(id) => db.lookup_intern_struct(id).container,
        AdtId::UnionId(id) => db.lookup_intern_union(id).container,
    };
    let Some(drop_trait) = LangItem::Drop.resolve_trait(db, module.krate()) else {
        return false;
    };
    let impls = match module.containing_block() {
        Some(block) => match db.trait_impls_in_block(block) {
            Some(it) => it,
            None => return false,
        },
        None => db.trait_impls_in_crate(module.krate()),
    };
    impls.for_trait_and_self_ty(drop_trait, TyFingerprint::Adt(adt)).next().is_some()
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

pub fn has_drop_glue<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    env: Arc<TraitEnvironment<'db>>,
) -> DropGlue {
    has_drop_glue_impl(infcx, ty, env, &mut FxHashSet::default())
}

fn has_drop_glue_impl<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    env: Arc<TraitEnvironment<'db>>,
    visited: &mut FxHashSet<Ty<'db>>,
) -> DropGlue {
    let mut ocx = ObligationCtxt::new(infcx);
    let ty = ocx.structurally_normalize_ty(&ObligationCause::dummy(), env.env, ty).unwrap_or(ty);

    if !visited.insert(ty) {
        // Recursive type.
        return DropGlue::None;
    }

    let db = infcx.interner.db;
    match ty.kind() {
        TyKind::Adt(adt_def, subst) => {
            let adt_id = adt_def.def_id().0;
            if has_destructor(db, adt_id) {
                return DropGlue::HasDropGlue;
            }
            match adt_id {
                AdtId::StructId(id) => {
                    if db
                        .struct_signature(id)
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
                                field_ty.instantiate(infcx.interner, subst),
                                env.clone(),
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
                                    field_ty.instantiate(infcx.interner, subst),
                                    env.clone(),
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
            .map(|ty| has_drop_glue_impl(infcx, ty, env.clone(), visited))
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
        TyKind::Closure(closure_id, subst) => {
            let owner = db.lookup_intern_closure(closure_id.0).0;
            let infer = db.infer(owner);
            let (captures, _) = infer.closure_info(closure_id.0);
            let env = db.trait_environment_for_body(owner);
            captures
                .iter()
                .map(|capture| {
                    has_drop_glue_impl(infcx, capture.ty(db, subst), env.clone(), visited)
                })
                .max()
                .unwrap_or(DropGlue::None)
        }
        // FIXME: Handle coroutines.
        TyKind::Coroutine(..) | TyKind::CoroutineWitness(..) | TyKind::CoroutineClosure(..) => {
            DropGlue::None
        }
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
            if infcx.type_is_copy_modulo_regions(env.env, ty) {
                DropGlue::None
            } else {
                DropGlue::HasDropGlue
            }
        }
        TyKind::Param(_) => {
            if infcx.type_is_copy_modulo_regions(env.env, ty) {
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
