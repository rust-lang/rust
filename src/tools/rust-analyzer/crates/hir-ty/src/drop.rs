//! Utilities for computing drop info about types.

use chalk_ir::cast::Cast;
use hir_def::AdtId;
use hir_def::lang_item::LangItem;
use hir_def::signatures::StructFlags;
use stdx::never;
use triomphe::Arc;

use crate::{
    AliasTy, Canonical, CanonicalVarKinds, ConcreteConst, ConstScalar, ConstValue, InEnvironment,
    Interner, ProjectionTy, TraitEnvironment, Ty, TyBuilder, TyKind, db::HirDatabase,
    method_resolution::TyFingerprint,
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

pub(crate) fn has_drop_glue(db: &dyn HirDatabase, ty: Ty, env: Arc<TraitEnvironment>) -> DropGlue {
    match ty.kind(Interner) {
        TyKind::Adt(adt, subst) => {
            if has_destructor(db, adt.0) {
                return DropGlue::HasDropGlue;
            }
            match adt.0 {
                AdtId::StructId(id) => {
                    if db.struct_signature(id).flags.contains(StructFlags::IS_MANUALLY_DROP) {
                        return DropGlue::None;
                    }
                    db.field_types(id.into())
                        .iter()
                        .map(|(_, field_ty)| {
                            db.has_drop_glue(
                                field_ty.clone().substitute(Interner, subst),
                                env.clone(),
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
                                db.has_drop_glue(
                                    field_ty.clone().substitute(Interner, subst),
                                    env.clone(),
                                )
                            })
                            .max()
                            .unwrap_or(DropGlue::None)
                    })
                    .max()
                    .unwrap_or(DropGlue::None),
            }
        }
        TyKind::Tuple(_, subst) => subst
            .iter(Interner)
            .map(|ty| ty.assert_ty_ref(Interner))
            .map(|ty| db.has_drop_glue(ty.clone(), env.clone()))
            .max()
            .unwrap_or(DropGlue::None),
        TyKind::Array(ty, len) => {
            if let ConstValue::Concrete(ConcreteConst { interned: ConstScalar::Bytes(len, _) }) =
                &len.data(Interner).value
            {
                match (&**len).try_into() {
                    Ok(len) => {
                        let len = usize::from_le_bytes(len);
                        if len == 0 {
                            // Arrays of size 0 don't have drop glue.
                            return DropGlue::None;
                        }
                    }
                    Err(_) => {
                        never!("const array size with non-usize len");
                    }
                }
            }
            db.has_drop_glue(ty.clone(), env)
        }
        TyKind::Slice(ty) => db.has_drop_glue(ty.clone(), env),
        TyKind::Closure(closure_id, subst) => {
            let owner = db.lookup_intern_closure((*closure_id).into()).0;
            let infer = db.infer(owner);
            let (captures, _) = infer.closure_info(closure_id);
            let env = db.trait_environment_for_body(owner);
            captures
                .iter()
                .map(|capture| db.has_drop_glue(capture.ty(subst), env.clone()))
                .max()
                .unwrap_or(DropGlue::None)
        }
        // FIXME: Handle coroutines.
        TyKind::Coroutine(..) | TyKind::CoroutineWitness(..) => DropGlue::None,
        TyKind::Ref(..)
        | TyKind::Raw(..)
        | TyKind::FnDef(..)
        | TyKind::Str
        | TyKind::Never
        | TyKind::Scalar(_)
        | TyKind::Function(_)
        | TyKind::Foreign(_)
        | TyKind::Error => DropGlue::None,
        TyKind::Dyn(_) => DropGlue::HasDropGlue,
        TyKind::AssociatedType(assoc_type_id, subst) => projection_has_drop_glue(
            db,
            env,
            ProjectionTy { associated_ty_id: *assoc_type_id, substitution: subst.clone() },
            ty,
        ),
        TyKind::Alias(AliasTy::Projection(projection)) => {
            projection_has_drop_glue(db, env, projection.clone(), ty)
        }
        TyKind::OpaqueType(..) | TyKind::Alias(AliasTy::Opaque(_)) => {
            if is_copy(db, ty, env) {
                DropGlue::None
            } else {
                DropGlue::HasDropGlue
            }
        }
        TyKind::Placeholder(_) | TyKind::BoundVar(_) => {
            if is_copy(db, ty, env) {
                DropGlue::None
            } else {
                DropGlue::DependOnParams
            }
        }
        TyKind::InferenceVar(..) => unreachable!("inference vars shouldn't exist out of inference"),
    }
}

fn projection_has_drop_glue(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    projection: ProjectionTy,
    ty: Ty,
) -> DropGlue {
    let normalized = db.normalize_projection(projection, env.clone());
    match normalized.kind(Interner) {
        TyKind::Alias(AliasTy::Projection(_)) | TyKind::AssociatedType(..) => {
            if is_copy(db, ty, env) { DropGlue::None } else { DropGlue::DependOnParams }
        }
        _ => db.has_drop_glue(normalized, env),
    }
}

fn is_copy(db: &dyn HirDatabase, ty: Ty, env: Arc<TraitEnvironment>) -> bool {
    let Some(copy_trait) = LangItem::Copy.resolve_trait(db, env.krate) else {
        return false;
    };
    let trait_ref = TyBuilder::trait_ref(db, copy_trait).push(ty).build();
    let goal = Canonical {
        value: InEnvironment::new(&env.env, trait_ref.cast(Interner)),
        binders: CanonicalVarKinds::empty(Interner),
    };
    db.trait_solve(env.krate, env.block, goal).is_some()
}

pub(crate) fn has_drop_glue_cycle_result(
    _db: &dyn HirDatabase,
    _ty: Ty,
    _env: Arc<TraitEnvironment>,
) -> DropGlue {
    DropGlue::None
}
