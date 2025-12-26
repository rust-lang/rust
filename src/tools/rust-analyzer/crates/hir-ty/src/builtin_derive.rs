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
    AliasTyKind, Interner, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor, Upcast,
    inherent::{GenericArgs as _, IntoKind},
};

use crate::{
    GenericPredicates,
    db::HirDatabase,
    next_solver::{
        Clause, Clauses, DbInterner, EarlyBinder, GenericArgs, ParamEnv, StoredEarlyBinder,
        StoredTy, TraitRef, Ty, TyKind, fold::fold_tys, generics::Generics,
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

pub(crate) fn generics_of<'db>(interner: DbInterner<'db>, id: BuiltinDeriveImplId) -> Generics {
    let db = interner.db;
    let loc = id.loc(db);
    match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::PartialOrd
        | BuiltinDeriveImplTrait::Eq
        | BuiltinDeriveImplTrait::PartialEq => interner.generics_of(loc.adt.into()),
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let mut generics = interner.generics_of(loc.adt.into());
            let trait_id = loc
                .trait_
                .get_id(interner.lang_items())
                .expect("we don't pass the impl to the solver if we can't resolve the trait");
            generics.push_param(coerce_pointee_new_type_param(trait_id).into());
            generics
        }
    }
}

pub fn generic_params_count(db: &dyn HirDatabase, id: BuiltinDeriveImplId) -> usize {
    let loc = id.loc(db);
    let adt_params = GenericParams::new(db, loc.adt.into());
    let extra_params_count = match loc.trait_ {
        BuiltinDeriveImplTrait::Copy
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
        | BuiltinDeriveImplTrait::Clone
        | BuiltinDeriveImplTrait::Default
        | BuiltinDeriveImplTrait::Debug
        | BuiltinDeriveImplTrait::Hash
        | BuiltinDeriveImplTrait::Ord
        | BuiltinDeriveImplTrait::Eq => {
            let self_ty = Ty::new_adt(
                interner,
                loc.adt,
                GenericArgs::identity_for_item(interner, loc.adt.into()),
            );
            EarlyBinder::bind(TraitRef::new(interner, trait_id.into(), [self_ty]))
        }
        BuiltinDeriveImplTrait::PartialOrd | BuiltinDeriveImplTrait::PartialEq => {
            let self_ty = Ty::new_adt(
                interner,
                loc.adt,
                GenericArgs::identity_for_item(interner, loc.adt.into()),
            );
            EarlyBinder::bind(TraitRef::new(interner, trait_id.into(), [self_ty, self_ty]))
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let generic_params = GenericParams::new(db, loc.adt.into());
            let interner = DbInterner::new_no_crate(db);
            let args = GenericArgs::identity_for_item(interner, loc.adt.into());
            let self_ty = Ty::new_adt(interner, loc.adt, args);
            let Some((pointee_param_idx, _, new_param_ty)) =
                coerce_pointee_params(interner, loc, &generic_params, trait_id)
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

#[salsa::tracked(returns(ref), unsafe(non_update_types))]
pub fn predicates<'db>(db: &'db dyn HirDatabase, impl_: BuiltinDeriveImplId) -> GenericPredicates {
    let loc = impl_.loc(db);
    let generic_params = GenericParams::new(db, loc.adt.into());
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
            simple_trait_predicates(interner, loc, &generic_params, adt_predicates, trait_id)
        }
        BuiltinDeriveImplTrait::Default => {
            if matches!(loc.adt, AdtId::EnumId(_)) {
                // Enums don't have extra bounds.
                GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
                    Clauses::new_from_slice(adt_predicates.explicit_predicates().skip_binder())
                        .store(),
                ))
            } else {
                simple_trait_predicates(interner, loc, &generic_params, adt_predicates, trait_id)
            }
        }
        BuiltinDeriveImplTrait::CoerceUnsized | BuiltinDeriveImplTrait::DispatchFromDyn => {
            let Some((pointee_param_idx, pointee_param_id, new_param_ty)) =
                coerce_pointee_params(interner, loc, &generic_params, trait_id)
            else {
                // Malformed derive.
                return GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
                    Clauses::default().store(),
                ));
            };
            let duplicated_bounds =
                adt_predicates.explicit_predicates().iter_identity_copied().filter_map(|pred| {
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
                    interner,
                    adt_predicates
                        .explicit_predicates()
                        .iter_identity_copied()
                        .chain(duplicated_bounds)
                        .chain(unsize_bound),
                )
                .store(),
            ))
        }
    }
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
            let param_idx =
                param_idx.into_raw().into_u32() + (generic_params.len_lifetimes() as u32);
            let param_ty = Ty::new_param(interner, param_id, param_idx);
            let trait_ref = TraitRef::new(interner, trait_id.into(), [param_ty]);
            trait_ref.upcast(interner)
        });
    let mut assoc_type_bounds = Vec::new();
    match loc.adt {
        AdtId::StructId(id) => extend_assoc_type_bounds(
            interner,
            &mut assoc_type_bounds,
            interner.db.field_types(id.into()),
            trait_id,
        ),
        AdtId::UnionId(id) => extend_assoc_type_bounds(
            interner,
            &mut assoc_type_bounds,
            interner.db.field_types(id.into()),
            trait_id,
        ),
        AdtId::EnumId(id) => {
            for &(variant_id, _, _) in &id.enum_variants(interner.db).variants {
                extend_assoc_type_bounds(
                    interner,
                    &mut assoc_type_bounds,
                    interner.db.field_types(variant_id.into()),
                    trait_id,
                )
            }
        }
    }
    GenericPredicates::from_explicit_own_predicates(StoredEarlyBinder::bind(
        Clauses::new_from_iter(
            interner,
            adt_predicates
                .explicit_predicates()
                .iter_identity_copied()
                .chain(extra_predicates)
                .chain(assoc_type_bounds),
        )
        .store(),
    ))
}

fn extend_assoc_type_bounds<'db>(
    interner: DbInterner<'db>,
    assoc_type_bounds: &mut Vec<Clause<'db>>,
    fields: &ArenaMap<LocalFieldId, StoredEarlyBinder<StoredTy>>,
    trait_: TraitId,
) {
    struct ProjectionFinder<'a, 'db> {
        interner: DbInterner<'db>,
        assoc_type_bounds: &'a mut Vec<Clause<'db>>,
        trait_: TraitId,
    }

    impl<'db> TypeVisitor<DbInterner<'db>> for ProjectionFinder<'_, 'db> {
        type Result = ();

        fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
            if let TyKind::Alias(AliasTyKind::Projection, _) = t.kind() {
                self.assoc_type_bounds.push(
                    TraitRef::new(self.interner, self.trait_.into(), [t]).upcast(self.interner),
                );
            }

            t.super_visit_with(self)
        }
    }

    let mut visitor = ProjectionFinder { interner, assoc_type_bounds, trait_ };
    for (_, field) in fields.iter() {
        field.get().instantiate_identity().visit_with(&mut visitor);
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
        pointee_param.into_raw().into_u32() + (generic_params.len_lifetimes() as u32);
    let new_param_idx = generic_params.len() as u32;
    let new_param_id = coerce_pointee_new_type_param(trait_id);
    let new_param_ty = Ty::new_param(interner, new_param_id, new_param_idx);
    Some((pointee_param_idx, pointee_param_id, new_param_ty))
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir_def::nameres::crate_def_map;
    use itertools::Itertools;
    use stdx::format_to;
    use test_fixture::WithFixture;

    use crate::{builtin_derive::impl_trait, next_solver::DbInterner, test_db::TestDB};

    fn check_trait_refs(#[rust_analyzer::rust_fixture] ra_fixture: &str, expectation: Expect) {
        let db = TestDB::with_files(ra_fixture);
        let def_map = crate_def_map(&db, db.test_crate());

        let interner = DbInterner::new_with(&db, db.test_crate());
        crate::attach_db(&db, || {
            let mut trait_refs = Vec::new();
            for (_, module) in def_map.modules() {
                for derive in module.scope.builtin_derive_impls() {
                    let trait_ref = impl_trait(interner, derive).skip_binder();
                    trait_refs.push(format!("{trait_ref:?}"));
                }
            }

            expectation.assert_eq(&trait_refs.join("\n"));
        });
    }

    fn check_predicates(#[rust_analyzer::rust_fixture] ra_fixture: &str, expectation: Expect) {
        let db = TestDB::with_files(ra_fixture);
        let def_map = crate_def_map(&db, db.test_crate());

        crate::attach_db(&db, || {
            let mut predicates = String::new();
            for (_, module) in def_map.modules() {
                for derive in module.scope.builtin_derive_impls() {
                    let preds = super::predicates(&db, derive).all_predicates().skip_binder();
                    format_to!(
                        predicates,
                        "{}\n\n",
                        preds.iter().format_with("\n", |pred, formatter| formatter(&format_args!(
                            "{pred:?}"
                        ))),
                    );
                }
            }

            expectation.assert_eq(&predicates);
        });
    }

    #[test]
    fn simple_macros_trait_ref() {
        check_trait_refs(
            r#"
//- minicore: derive, clone, copy, eq, ord, hash, fmt

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Simple;

trait Trait {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct WithGenerics<'a, T: Trait, const N: usize>(&'a [T; N]);
        "#,
            expect![[r#"
                Simple: Debug
                Simple: Clone
                Simple: Copy
                Simple: PartialEq<[Simple]>
                Simple: Eq
                Simple: PartialOrd<[Simple]>
                Simple: Ord
                Simple: Hash
                WithGenerics<#0, #1, #2>: Debug
                WithGenerics<#0, #1, #2>: Clone
                WithGenerics<#0, #1, #2>: Copy
                WithGenerics<#0, #1, #2>: PartialEq<[WithGenerics<#0, #1, #2>]>
                WithGenerics<#0, #1, #2>: Eq
                WithGenerics<#0, #1, #2>: PartialOrd<[WithGenerics<#0, #1, #2>]>
                WithGenerics<#0, #1, #2>: Ord
                WithGenerics<#0, #1, #2>: Hash"#]],
        );
    }

    #[test]
    fn coerce_pointee_trait_ref() {
        check_trait_refs(
            r#"
//- minicore: derive, coerce_pointee
use core::marker::CoercePointee;

#[derive(CoercePointee)]
struct Simple<T: ?Sized>(*const T);

#[derive(CoercePointee)]
struct MultiGenericParams<'a, T, #[pointee] U: ?Sized, const N: usize>(*const U);
        "#,
            expect![[r#"
                Simple<#0>: CoerceUnsized<[Simple<#1>]>
                Simple<#0>: DispatchFromDyn<[Simple<#1>]>
                MultiGenericParams<#0, #1, #2, #3>: CoerceUnsized<[MultiGenericParams<#0, #1, #4, #3>]>
                MultiGenericParams<#0, #1, #2, #3>: DispatchFromDyn<[MultiGenericParams<#0, #1, #4, #3>]>"#]],
        );
    }

    #[test]
    fn simple_macros_predicates() {
        check_predicates(
            r#"
//- minicore: derive, clone, copy, eq, ord, hash, fmt

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Simple;

trait Trait {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct WithGenerics<'a, T: Trait, const N: usize>(&'a [T; N]);
        "#,
            expect![[r#"
















                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Debug, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Clone, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Copy, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: PartialEq, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Eq, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: PartialOrd, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Ord, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#2, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Hash, polarity:Positive), bound_vars: [] })

            "#]],
        );
    }

    #[test]
    fn coerce_pointee_predicates() {
        check_predicates(
            r#"
//- minicore: derive, coerce_pointee
use core::marker::CoercePointee;

#[derive(CoercePointee)]
struct Simple<T: ?Sized>(*const T);

trait Trait<T> {}

#[derive(CoercePointee)]
struct MultiGenericParams<'a, T, #[pointee] U: ?Sized, const N: usize>(*const U)
where
    T: Trait<U>,
    U: Trait<U>;
        "#,
            expect![[r#"
                Clause(Binder { value: TraitPredicate(#0: Unsize<[#1]>, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#0: Unsize<[#1]>, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait<[#2]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#2: Trait<[#2]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#3, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Trait<[#4]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#4: Trait<[#4]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#2: Unsize<[#4]>, polarity:Positive), bound_vars: [] })

                Clause(Binder { value: TraitPredicate(#1: Trait<[#2]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#2: Trait<[#2]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: ConstArgHasType(#3, usize), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Sized, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#1: Trait<[#4]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#4: Trait<[#4]>, polarity:Positive), bound_vars: [] })
                Clause(Binder { value: TraitPredicate(#2: Unsize<[#4]>, polarity:Positive), bound_vars: [] })

            "#]],
        );
    }
}
