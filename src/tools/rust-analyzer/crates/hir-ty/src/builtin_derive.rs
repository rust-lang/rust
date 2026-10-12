//! Implementation of builtin derive impls.

use std::ops::ControlFlow;

use hir_def::{
    AdtId, BuiltinDeriveImplId, BuiltinDeriveImplLoc, HasModule, LocalFieldId, TraitId,
    TypeOrConstParamId, TypeParamId,
    attrs::AttrFlags,
    builtin_derive::BuiltinDeriveImplTrait,
    hir::generics::{GenericParams, TypeOrConstParamData},
};
use itertools::Itertools;
use la_arena::ArenaMap;
use rustc_type_ir::{
    AliasTyKind, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor, Upcast,
    inherent::{GenericArgs as _, IntoKind},
};

use crate::{
    FieldType, GenericPredicates,
    db::HirDatabase,
    next_solver::{
        AliasTy, Clause, Clauses, DbInterner, EarlyBinder, GenericArgs, ParamEnv,
        StoredEarlyBinder, TraitRef, Ty, TyKind, Unnormalized, fold::fold_tys, generics::Generics,
    },
};

fn coerce_pointee_new_type_param(trait_id: TraitId) -> TypeParamId {
    // HACK: Fake the param.
    // We cannot use a dummy param here, because it can leak into the IDE layer and that'll cause panics
    // when e.g. trying to display it. So we use an existing param.
    TypeParamId::from_unchecked(TypeOrConstParamId {
        parent: trait_id.into(),
        local_id: la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(1)),
    })
}

fn trait_args(trait_: BuiltinDeriveImplTrait, self_ty: Ty<'_>) -> GenericArgs<'_> {
    match trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Reborrow
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::Ord => GenericArgs::new_from_slice(&[self_ty.into()]),
        BuiltinDeriveImplTrait::PartialOrd | BuiltinDeriveImplTrait::PartialEq => {
            GenericArgs::new_from_slice(&[self_ty.into(), self_ty.into()])
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            panic!("`CoerceUnsized` and `DispatchFromDyn` have special generics")
        }
    }
}

pub(crate) fn generics_of<'db>(
    interner: DbInterner<'db>,
    id: BuiltinDeriveImplId,
) -> Generics<'db> {
    let db = interner.db;
    let loc = id.loc(db);
    match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Reborrow
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::PartialOrd
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::PartialEq => {
            Generics::from_generic_def(db, loc.adt.into(), false)
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let trait_id = loc
                .trait_
                .get_id(interner.lang_items())
                .expect("we don't pass the impl to the solver if we can't resolve the trait");
            let additional_param = coerce_pointee_new_type_param(trait_id);
            Generics::from_generic_def_plus_one(db, loc.adt.into(), additional_param, false)
        }
    }
}

pub fn generic_params_count(db: &dyn HirDatabase, id: BuiltinDeriveImplId) -> usize {
    let loc = id.loc(db);
    let adt_params = GenericParams::of(db, loc.adt.into());
    let extra_params_count = match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Reborrow
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::PartialOrd
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::PartialEq => 0,
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => 1,
    };
    adt_params.len() + extra_params_count
}

pub fn impl_trait<'db>(
    interner: DbInterner<'db>,
    id: BuiltinDeriveImplId,
) -> EarlyBinder<'db, TraitRef<'db>> {
    let db = interner.db;
    let loc = id.loc(db);
    let trait_id = loc
        .trait_
        .get_id(interner.lang_items())
        .expect("we don't pass the impl to the solver if we can't resolve the trait");
    match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Reborrow
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::PartialOrd
        | BuiltinDeriveImplTrait::PartialEq => {
            let self_ty = Ty::new_adt(
                interner,
                loc.adt,
                GenericArgs::identity_for_item(interner, loc.adt.into()),
            );
            EarlyBinder::bind(TraitRef::new_from_args(
                interner,
                trait_id.into(),
                trait_args(loc.trait_, self_ty),
            ))
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let generic_params = GenericParams::of(db, loc.adt.into());
            let interner = DbInterner::new_no_crate(db);
            let args = GenericArgs::identity_for_item(interner, loc.adt.into());
            let self_ty = Ty::new_adt(interner, loc.adt, args);
            let Some((pointee_param_idx, _, new_param_ty)) =
                coerce_pointee_params(interner, loc, generic_params, trait_id)
            else {
                // Malformed derive.
                return EarlyBinder::bind(TraitRef::new(
                    interner,
                    trait_id.into(),
                    [self_ty, self_ty],
                ));
            };
            let changed_args = replace_pointee(interner, pointee_param_idx, new_param_ty, args);
            let changed_self_ty = Ty::new_adt(interner, loc.adt, changed_args);
            EarlyBinder::bind(TraitRef::new(interner, trait_id.into(), [self_ty, changed_self_ty]))
        }
    }
}

#[inline]
pub fn impl_super_outlives<'db>(
    _interner: DbInterner<'db>,
    _id: BuiltinDeriveImplId,
) -> EarlyBinder<'db, &'db [Clause<'db>]> {
    // Currently no builtin-derived traits have super outlive bounds.
    EarlyBinder::bind(&[])
}

#[salsa::tracked(returns(ref))]
pub fn predicates(db: &dyn HirDatabase, impl_: BuiltinDeriveImplId) -> GenericPredicates {
    let loc = impl_.loc(db);
    let generic_params = GenericParams::of(db, loc.adt.into());
    let interner = DbInterner::new_with(db, loc.module(db).krate(db));
    let adt_predicates = GenericPredicates::query(db, loc.adt.into());
    let trait_id = loc
        .trait_
        .get_id(interner.lang_items())
        .expect("we don't pass the impl to the solver if we can't resolve the trait");
    match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::PartialOrd
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::PartialEq => {
            simple_trait_predicates(interner, loc, generic_params, adt_predicates, trait_id)
        }
        BuiltinDeriveImplTrait::Reborrow => {
            explicit_own_predicates(adt_predicates.own_explicit_predicates())
        }
        BuiltinDeriveImplTrait::Default => {
            if matches!(loc.adt, AdtId::EnumId(_)) {
                // Enums don't have extra bounds.
                explicit_own_predicates(adt_predicates.own_explicit_predicates())
            } else {
                simple_trait_predicates(interner, loc, generic_params, adt_predicates, trait_id)
            }
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let Some((pointee_param_idx, pointee_param_id, new_param_ty)) =
                coerce_pointee_params(interner, loc, generic_params, trait_id)
            else {
                // Malformed derive.
                return GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
                    Clauses::empty().store(),
                ));
            };
            let duplicated_bounds =
                adt_predicates.explicit_predicates().iter_identity().filter_map(|pred| {
                    let pred = pred.skip_norm_wip();
                    let mentions_pointee =
                        pred.visit_with(&mut MentionsPointee { pointee_param_idx }).is_break();
                    if !mentions_pointee {
                        return None;
                    }
                    let transformed =
                        replace_pointee(interner, pointee_param_idx, new_param_ty, pred);
                    Some(transformed)
                });
            let unsize_trait = interner.lang_items().Unsize;
            let unsize_bound = unsize_trait.map(|unsize_trait| {
                let pointee_param_ty = Ty::new_param(interner, pointee_param_id, pointee_param_idx);
                TraitRef::new(interner, unsize_trait.into(), [pointee_param_ty, new_param_ty])
                    .upcast(interner)
            });
            GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
                Clauses::new_from_iter(
                    adt_predicates
                        .explicit_predicates()
                        .iter_identity()
                        .map(Unnormalized::skip_norm_wip)
                        .chain(duplicated_bounds)
                        .chain(unsize_bound),
                )
                .store(),
            ))
        }
    }
}

fn explicit_own_predicates<'db>(
    predicates: EarlyBinder<'db, impl Iterator<Item = Clause<'db>>>,
) -> GenericPredicates {
    GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
        Clauses::new_from_iter(predicates.skip_binder()).store(),
    ))
}

/// Not cached in a query, currently used in `hir` only. If you need this in `hir-ty` consider introducing a query.
pub fn param_env<'db>(interner: DbInterner<'db>, id: BuiltinDeriveImplId) -> ParamEnv<'db> {
    let predicates = predicates(interner.db, id);
    crate::lower::param_env_from_predicates(interner, predicates)
}

struct MentionsPointee {
    pointee_param_idx: u32,
}

impl<'db> TypeVisitor<DbInterner<'db>> for MentionsPointee {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
        if let TyKind::Param(param) = t.kind()
            && param.index == self.pointee_param_idx
        {
            ControlFlow::Break(())
        } else {
            t.super_visit_with(self)
        }
    }
}

fn replace_pointee<'db, T: TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    pointee_param_idx: u32,
    new_param_ty: Ty<'db>,
    t: T,
) -> T {
    fold_tys(interner, t, |ty| match ty.kind() {
        TyKind::Param(param) if param.index == pointee_param_idx => new_param_ty,
        _ => ty,
    })
}

fn simple_trait_predicates<'db>(
    interner: DbInterner<'db>,
    loc: &BuiltinDeriveImplLoc,
    generic_params: &GenericParams,
    adt_predicates: &GenericPredicates,
    trait_id: TraitId,
) -> GenericPredicates {
    let extra_predicates = generic_params
        .iter_type_or_consts()
        .filter(|(_, data)| matches!(data, TypeOrConstParamData::TypeParamData(_)))
        .map(|(param_idx, _)| {
            let param_id = TypeParamId::from_unchecked(TypeOrConstParamId {
                parent: loc.adt.into(),
                local_id: param_idx,
            });
            let param_idx = param_idx.into_raw().into_u32()
                + (generic_params.len_early_bound_lifetimes() as u32);
            let param_ty = Ty::new_param(interner, param_id, param_idx);
            let trait_args = trait_args(loc.trait_, param_ty);
            let trait_ref = TraitRef::new_from_args(interner, trait_id.into(), trait_args);
            trait_ref.upcast(interner)
        });
    let mut assoc_type_bounds = Vec::new();
    match loc.adt {
        AdtId::StructId(id) => extend_assoc_type_bounds(
            interner,
            &mut assoc_type_bounds,
            interner.db.field_types(id.into()),
            trait_id,
            loc.trait_,
        ),
        AdtId::UnionId(id) => extend_assoc_type_bounds(
            interner,
            &mut assoc_type_bounds,
            interner.db.field_types(id.into()),
            trait_id,
            loc.trait_,
        ),
        AdtId::EnumId(id) => {
            for &(variant_id, _) in id.enum_variants(interner.db).variants.values() {
                extend_assoc_type_bounds(
                    interner,
                    &mut assoc_type_bounds,
                    interner.db.field_types(variant_id.into()),
                    trait_id,
                    loc.trait_,
                )
            }
        }
    }
    GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
        Clauses::new_from_iter(
            adt_predicates
                .explicit_predicates()
                .iter_identity()
                .map(Unnormalized::skip_norm_wip)
                .chain(extra_predicates)
                .chain(assoc_type_bounds),
        )
        .store(),
    ))
}

fn extend_assoc_type_bounds<'db>(
    interner: DbInterner<'db>,
    assoc_type_bounds: &mut Vec<Clause<'db>>,
    fields: &ArenaMap<LocalFieldId, FieldType>,
    trait_id: TraitId,
    trait_: BuiltinDeriveImplTrait,
) {
    struct ProjectionFinder<'a, 'db> {
        interner: DbInterner<'db>,
        assoc_type_bounds: &'a mut Vec<Clause<'db>>,
        trait_id: TraitId,
        trait_: BuiltinDeriveImplTrait,
    }

    impl<'db> TypeVisitor<DbInterner<'db>> for ProjectionFinder<'_, 'db> {
        type Result = ();

        fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
            if let TyKind::Alias(AliasTy { kind: AliasTyKind::Projection { .. }, .. }) = t.kind() {
                self.assoc_type_bounds.push(
                    TraitRef::new_from_args(
                        self.interner,
                        self.trait_id.into(),
                        trait_args(self.trait_, t),
                    )
                    .upcast(self.interner),
                );
            }

            t.super_visit_with(self)
        }
    }

    let mut visitor = ProjectionFinder { interner, assoc_type_bounds, trait_id, trait_ };
    for (_, field) in fields.iter() {
        field.ty().instantiate_identity().skip_norm_wip().visit_with(&mut visitor);
    }
}

fn coerce_pointee_params<'db>(
    interner: DbInterner<'db>,
    loc: &BuiltinDeriveImplLoc,
    generic_params: &GenericParams,
    trait_id: TraitId,
) -> Option<(u32, TypeParamId, Ty<'db>)> {
    let pointee_param = {
        if let Ok((pointee_param, _)) = generic_params
            .iter_type_or_consts()
            .filter(|param| matches!(param.1, TypeOrConstParamData::TypeParamData(_)))
            .exactly_one()
        {
            pointee_param
        } else {
            let (_, generic_param_attrs) =
                AttrFlags::query_generic_params(interner.db, loc.adt.into());
            generic_param_attrs
                .iter()
                .find(|param| param.1.contains(AttrFlags::IS_POINTEE))
                .map(|(param, _)| param)
                .or_else(|| {
                    generic_params
                        .iter_type_or_consts()
                        .find(|param| matches!(param.1, TypeOrConstParamData::TypeParamData(_)))
                        .map(|(idx, _)| idx)
                })?
        }
    };
    let pointee_param_id = TypeParamId::from_unchecked(TypeOrConstParamId {
        parent: loc.adt.into(),
        local_id: pointee_param,
    });
    let pointee_param_idx =
        pointee_param.into_raw().into_u32() + (generic_params.len_early_bound_lifetimes() as u32);
    let new_param_idx = generic_params.len() as u32;
    let new_param_id = coerce_pointee_new_type_param(trait_id);
    let new_param_ty = Ty::new_param(interner, new_param_id, new_param_idx);
    Some((pointee_param_idx, pointee_param_id, new_param_ty))
}
