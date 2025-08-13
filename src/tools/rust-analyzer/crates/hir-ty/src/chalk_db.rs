//! The implementation of `RustIrDatabase` for Chalk, which provides information
//! about the code that Chalk needs.
use std::sync::Arc;

use tracing::debug;

use chalk_ir::{cast::Caster, fold::shift::Shift};
use chalk_solve::rust_ir::{self, WellKnownTrait};

use base_db::Crate;
use hir_def::{
    AssocItemId, CallableDefId, GenericDefId, HasModule, ItemContainerId, Lookup, TypeAliasId,
    VariantId,
    lang_item::LangItem,
    signatures::{ImplFlags, StructFlags, TraitFlags},
};

use crate::{
    AliasEq, AliasTy, DebruijnIndex, Interner, ProjectionTyExt, QuantifiedWhereClause,
    Substitution, TraitRefExt, Ty, TyBuilder, TyExt, TyKind, WhereClause,
    db::HirDatabase,
    from_assoc_type_id, from_chalk_trait_id,
    generics::generics,
    lower::LifetimeElisionKind,
    make_binders,
    mapping::{ToChalk, TypeAliasAsValue, from_chalk},
    to_assoc_type_id, to_chalk_trait_id,
};

pub(crate) type AssociatedTyDatum = chalk_solve::rust_ir::AssociatedTyDatum<Interner>;
pub(crate) type TraitDatum = chalk_solve::rust_ir::TraitDatum<Interner>;
pub(crate) type AdtDatum = chalk_solve::rust_ir::AdtDatum<Interner>;
pub(crate) type ImplDatum = chalk_solve::rust_ir::ImplDatum<Interner>;

pub(crate) type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub(crate) type TraitId = chalk_ir::TraitId<Interner>;
pub(crate) type AdtId = chalk_ir::AdtId<Interner>;
pub(crate) type ImplId = chalk_ir::ImplId<Interner>;
pub(crate) type AssociatedTyValueId = chalk_solve::rust_ir::AssociatedTyValueId<Interner>;
pub(crate) type AssociatedTyValue = chalk_solve::rust_ir::AssociatedTyValue<Interner>;
pub(crate) type FnDefDatum = chalk_solve::rust_ir::FnDefDatum<Interner>;
pub(crate) type Variances = chalk_ir::Variances<Interner>;

impl chalk_ir::UnificationDatabase<Interner> for &dyn HirDatabase {
    fn fn_def_variance(
        &self,
        fn_def_id: chalk_ir::FnDefId<Interner>,
    ) -> chalk_ir::Variances<Interner> {
        HirDatabase::fn_def_variance(*self, from_chalk(*self, fn_def_id))
    }

    fn adt_variance(&self, adt_id: chalk_ir::AdtId<Interner>) -> chalk_ir::Variances<Interner> {
        HirDatabase::adt_variance(*self, adt_id.0)
    }
}

pub(crate) fn associated_ty_data_query(
    db: &dyn HirDatabase,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyDatum> {
    debug!("associated_ty_data {:?}", type_alias);
    let trait_ = match type_alias.lookup(db).container {
        ItemContainerId::TraitId(t) => t,
        _ => panic!("associated type not in trait"),
    };

    // Lower bounds -- we could/should maybe move this to a separate query in `lower`
    let type_alias_data = db.type_alias_signature(type_alias);
    let generic_params = generics(db, type_alias.into());
    let resolver = hir_def::resolver::HasResolver::resolver(type_alias, db);
    let mut ctx = crate::TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        type_alias.into(),
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_type_param_mode(crate::lower::ParamLoweringMode::Variable);

    let trait_subst = TyBuilder::subst_for_def(db, trait_, None)
        .fill_with_bound_vars(crate::DebruijnIndex::INNERMOST, 0)
        .build();
    let pro_ty = TyBuilder::assoc_type_projection(db, type_alias, Some(trait_subst))
        .fill_with_bound_vars(
            crate::DebruijnIndex::INNERMOST,
            generic_params.parent_generics().map_or(0, |it| it.len()),
        )
        .build();
    let self_ty = TyKind::Alias(AliasTy::Projection(pro_ty)).intern(Interner);

    let mut bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, self_ty.clone(), false).for_each(|pred| {
            if let Some(pred) = generic_predicate_to_inline_bound(db, &pred, &self_ty) {
                bounds.push(pred);
            }
        });
    }

    if !ctx.unsized_types.contains(&self_ty) {
        let sized_trait =
            LangItem::Sized.resolve_trait(db, resolver.krate()).map(to_chalk_trait_id);
        let sized_bound = sized_trait.into_iter().map(|sized_trait| {
            let trait_bound =
                rust_ir::TraitBound { trait_id: sized_trait, args_no_self: Default::default() };
            let inline_bound = rust_ir::InlineBound::TraitBound(trait_bound);
            chalk_ir::Binders::empty(Interner, inline_bound)
        });
        bounds.extend(sized_bound);
        bounds.shrink_to_fit();
    }

    // FIXME: Re-enable where clauses on associated types when an upstream chalk bug is fixed.
    //        (rust-analyzer#9052)
    // let where_clauses = convert_where_clauses(db, type_alias.into(), &bound_vars);
    let bound_data = rust_ir::AssociatedTyDatumBound { bounds, where_clauses: vec![] };
    let datum = AssociatedTyDatum {
        trait_id: to_chalk_trait_id(trait_),
        id: to_assoc_type_id(type_alias),
        name: type_alias,
        binders: make_binders(db, &generic_params, bound_data),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    trait_id: TraitId,
) -> Arc<TraitDatum> {
    debug!("trait_datum {:?}", trait_id);
    let trait_ = from_chalk_trait_id(trait_id);
    let trait_data = db.trait_signature(trait_);
    debug!("trait {:?} = {:?}", trait_id, trait_data.name);
    let generic_params = generics(db, trait_.into());
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let flags = rust_ir::TraitFlags {
        auto: trait_data.flags.contains(TraitFlags::AUTO),
        upstream: trait_.lookup(db).container.krate() != krate,
        non_enumerable: true,
        coinductive: false, // only relevant for Chalk testing
        // FIXME: set these flags correctly
        marker: false,
        fundamental: trait_data.flags.contains(TraitFlags::FUNDAMENTAL),
    };
    let where_clauses = convert_where_clauses(db, trait_.into(), &bound_vars);
    let associated_ty_ids =
        trait_.trait_items(db).associated_types().map(to_assoc_type_id).collect();
    let trait_datum_bound = rust_ir::TraitDatumBound { where_clauses };
    let well_known = db.lang_attr(trait_.into()).and_then(well_known_trait_from_lang_item);
    let trait_datum = TraitDatum {
        id: trait_id,
        binders: make_binders(db, &generic_params, trait_datum_bound),
        flags,
        associated_ty_ids,
        well_known,
    };
    Arc::new(trait_datum)
}

fn well_known_trait_from_lang_item(item: LangItem) -> Option<WellKnownTrait> {
    Some(match item {
        LangItem::Clone => WellKnownTrait::Clone,
        LangItem::CoerceUnsized => WellKnownTrait::CoerceUnsized,
        LangItem::Copy => WellKnownTrait::Copy,
        LangItem::DiscriminantKind => WellKnownTrait::DiscriminantKind,
        LangItem::DispatchFromDyn => WellKnownTrait::DispatchFromDyn,
        LangItem::Drop => WellKnownTrait::Drop,
        LangItem::Fn => WellKnownTrait::Fn,
        LangItem::FnMut => WellKnownTrait::FnMut,
        LangItem::FnOnce => WellKnownTrait::FnOnce,
        LangItem::AsyncFn => WellKnownTrait::AsyncFn,
        LangItem::AsyncFnMut => WellKnownTrait::AsyncFnMut,
        LangItem::AsyncFnOnce => WellKnownTrait::AsyncFnOnce,
        LangItem::Coroutine => WellKnownTrait::Coroutine,
        LangItem::Sized => WellKnownTrait::Sized,
        LangItem::Unpin => WellKnownTrait::Unpin,
        LangItem::Unsize => WellKnownTrait::Unsize,
        LangItem::Tuple => WellKnownTrait::Tuple,
        LangItem::PointeeTrait => WellKnownTrait::Pointee,
        LangItem::FnPtrTrait => WellKnownTrait::FnPtr,
        LangItem::Future => WellKnownTrait::Future,
        _ => return None,
    })
}

pub(crate) fn adt_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    chalk_ir::AdtId(adt_id): AdtId,
) -> Arc<AdtDatum> {
    debug!("adt_datum {:?}", adt_id);
    let generic_params = generics(db, adt_id.into());
    let bound_vars_subst = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let where_clauses = convert_where_clauses(db, adt_id.into(), &bound_vars_subst);

    let (fundamental, phantom_data) = match adt_id {
        hir_def::AdtId::StructId(s) => {
            let flags = db.struct_signature(s).flags;
            (flags.contains(StructFlags::FUNDAMENTAL), flags.contains(StructFlags::IS_PHANTOM_DATA))
        }
        // FIXME set fundamental flags correctly
        hir_def::AdtId::UnionId(_) => (false, false),
        hir_def::AdtId::EnumId(_) => (false, false),
    };
    let flags = rust_ir::AdtFlags {
        upstream: adt_id.module(db).krate() != krate,
        fundamental,
        phantom_data,
    };

    // this slows down rust-analyzer by quite a bit unfortunately, so enabling this is currently not worth it
    let _variant_id_to_fields = |id: VariantId| {
        let variant_data = &id.fields(db);
        let fields = if variant_data.fields().is_empty() {
            vec![]
        } else {
            let field_types = db.field_types(id);
            variant_data
                .fields()
                .iter()
                .map(|(idx, _)| field_types[idx].clone().substitute(Interner, &bound_vars_subst))
                .filter(|it| !it.contains_unknown())
                .collect()
        };
        rust_ir::AdtVariantDatum { fields }
    };
    let variant_id_to_fields = |_: VariantId| rust_ir::AdtVariantDatum { fields: vec![] };

    let (kind, variants) = match adt_id {
        hir_def::AdtId::StructId(id) => {
            (rust_ir::AdtKind::Struct, vec![variant_id_to_fields(id.into())])
        }
        hir_def::AdtId::EnumId(id) => {
            let variants = id
                .enum_variants(db)
                .variants
                .iter()
                .map(|&(variant_id, _, _)| variant_id_to_fields(variant_id.into()))
                .collect();
            (rust_ir::AdtKind::Enum, variants)
        }
        hir_def::AdtId::UnionId(id) => {
            (rust_ir::AdtKind::Union, vec![variant_id_to_fields(id.into())])
        }
    };

    let struct_datum_bound = rust_ir::AdtDatumBound { variants, where_clauses };
    let struct_datum = AdtDatum {
        kind,
        id: chalk_ir::AdtId(adt_id),
        binders: make_binders(db, &generic_params, struct_datum_bound),
        flags,
    };
    Arc::new(struct_datum)
}

pub(crate) fn impl_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    impl_id: ImplId,
) -> Arc<ImplDatum> {
    let _p = tracing::info_span!("impl_datum_query").entered();
    debug!("impl_datum {:?}", impl_id);
    let impl_: hir_def::ImplId = from_chalk(db, impl_id);
    impl_def_datum(db, krate, impl_)
}

fn impl_def_datum(db: &dyn HirDatabase, krate: Crate, impl_id: hir_def::ImplId) -> Arc<ImplDatum> {
    let trait_ref = db
        .impl_trait(impl_id)
        // ImplIds for impls where the trait ref can't be resolved should never reach Chalk
        .expect("invalid impl passed to Chalk")
        .into_value_and_skipped_binders()
        .0;
    let impl_data = db.impl_signature(impl_id);

    let generic_params = generics(db, impl_id.into());
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let trait_ = trait_ref.hir_trait_id();
    let impl_type = if impl_id.lookup(db).container.krate() == krate {
        rust_ir::ImplType::Local
    } else {
        rust_ir::ImplType::External
    };
    let where_clauses = convert_where_clauses(db, impl_id.into(), &bound_vars);
    let negative = impl_data.flags.contains(ImplFlags::NEGATIVE);
    let polarity = if negative { rust_ir::Polarity::Negative } else { rust_ir::Polarity::Positive };

    let impl_datum_bound = rust_ir::ImplDatumBound { trait_ref, where_clauses };
    let trait_data = trait_.trait_items(db);
    let associated_ty_value_ids = impl_id
        .impl_items(db)
        .items
        .iter()
        .filter_map(|(_, item)| match item {
            AssocItemId::TypeAliasId(type_alias) => Some(*type_alias),
            _ => None,
        })
        .filter(|&type_alias| {
            // don't include associated types that don't exist in the trait
            let name = &db.type_alias_signature(type_alias).name;
            trait_data.associated_type_by_name(name).is_some()
        })
        .map(|type_alias| TypeAliasAsValue(type_alias).to_chalk(db))
        .collect();
    debug!("impl_datum: {:?}", impl_datum_bound);
    let impl_datum = ImplDatum {
        binders: make_binders(db, &generic_params, impl_datum_bound),
        impl_type,
        polarity,
        associated_ty_value_ids,
    };
    Arc::new(impl_datum)
}

pub(crate) fn associated_ty_value_query(
    db: &dyn HirDatabase,
    krate: Crate,
    id: AssociatedTyValueId,
) -> Arc<AssociatedTyValue> {
    let type_alias: TypeAliasAsValue = from_chalk(db, id);
    type_alias_associated_ty_value(db, krate, type_alias.0)
}

fn type_alias_associated_ty_value(
    db: &dyn HirDatabase,
    _krate: Crate,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyValue> {
    let type_alias_data = db.type_alias_signature(type_alias);
    let impl_id = match type_alias.lookup(db).container {
        ItemContainerId::ImplId(it) => it,
        _ => panic!("assoc ty value should be in impl"),
    };

    let trait_ref = db
        .impl_trait(impl_id)
        .expect("assoc ty value should not exist")
        .into_value_and_skipped_binders()
        .0; // we don't return any assoc ty values if the impl'd trait can't be resolved

    let assoc_ty = trait_ref
        .hir_trait_id()
        .trait_items(db)
        .associated_type_by_name(&type_alias_data.name)
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let (ty, binders) = db.ty(type_alias.into()).into_value_and_skipped_binders();
    let value_bound = rust_ir::AssociatedTyValueBound { ty };
    let value = rust_ir::AssociatedTyValue {
        impl_id: impl_id.to_chalk(db),
        associated_ty_id: to_assoc_type_id(assoc_ty),
        value: chalk_ir::Binders::new(binders, value_bound),
    };
    Arc::new(value)
}

pub(crate) fn fn_def_datum_query(
    db: &dyn HirDatabase,
    callable_def: CallableDefId,
) -> Arc<FnDefDatum> {
    let generic_def = GenericDefId::from_callable(db, callable_def);
    let generic_params = generics(db, generic_def);
    let (sig, binders) = db.callable_item_signature(callable_def).into_value_and_skipped_binders();
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let where_clauses = convert_where_clauses(db, generic_def, &bound_vars);
    let bound = rust_ir::FnDefDatumBound {
        // Note: Chalk doesn't actually use this information yet as far as I am aware, but we provide it anyway
        inputs_and_output: chalk_ir::Binders::empty(
            Interner,
            rust_ir::FnDefInputsAndOutputDatum {
                argument_types: sig.params().to_vec(),
                return_type: sig.ret().clone(),
            }
            .shifted_in(Interner),
        ),
        where_clauses,
    };
    let datum = FnDefDatum {
        id: callable_def.to_chalk(db),
        sig: chalk_ir::FnSig {
            abi: sig.abi,
            safety: chalk_ir::Safety::Safe,
            variadic: sig.is_varargs,
        },
        binders: chalk_ir::Binders::new(binders, bound),
    };
    Arc::new(datum)
}

pub(crate) fn fn_def_variance_query(
    db: &dyn HirDatabase,
    callable_def: CallableDefId,
) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(GenericDefId::from_callable(db, callable_def))
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|v| match v {
                crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
                crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
                crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
                crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
            }),
    )
}

pub(crate) fn adt_variance_query(db: &dyn HirDatabase, adt_id: hir_def::AdtId) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(adt_id.into()).as_deref().unwrap_or_default().iter().map(|v| match v {
            crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
            crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
            crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
            crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
        }),
    )
}

/// Returns instantiated predicates.
pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substitution,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    db.generic_predicates(def)
        .iter()
        .cloned()
        .map(|pred| pred.substitute(Interner, substs))
        .collect()
}

pub(super) fn generic_predicate_to_inline_bound(
    db: &dyn HirDatabase,
    pred: &QuantifiedWhereClause,
    self_ty: &Ty,
) -> Option<chalk_ir::Binders<rust_ir::InlineBound<Interner>>> {
    // An InlineBound is like a GenericPredicate, except the self type is left out.
    // We don't have a special type for this, but Chalk does.
    let self_ty_shifted_in = self_ty.clone().shifted_in_from(Interner, DebruijnIndex::ONE);
    let (pred, binders) = pred.as_ref().into_value_and_skipped_binders();
    match pred {
        WhereClause::Implemented(trait_ref) => {
            if trait_ref.self_type_parameter(Interner) != self_ty_shifted_in {
                // we can only convert predicates back to type bounds if they
                // have the expected self type
                return None;
            }
            let args_no_self = trait_ref.substitution.as_slice(Interner)[1..]
                .iter()
                .cloned()
                .casted(Interner)
                .collect();
            let trait_bound = rust_ir::TraitBound { trait_id: trait_ref.trait_id, args_no_self };
            Some(chalk_ir::Binders::new(binders, rust_ir::InlineBound::TraitBound(trait_bound)))
        }
        WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
            let generics = generics(db, from_assoc_type_id(projection_ty.associated_ty_id).into());
            let parent_len = generics.parent_generics().map_or(0, |g| g.len_self());
            let (trait_args, assoc_args) =
                projection_ty.substitution.as_slice(Interner).split_at(parent_len);
            let (self_ty, args_no_self) =
                trait_args.split_first().expect("projection without trait self type");
            if self_ty.assert_ty_ref(Interner) != &self_ty_shifted_in {
                return None;
            }

            let args_no_self = args_no_self.iter().cloned().casted(Interner).collect();
            let parameters = assoc_args.to_vec();

            let alias_eq_bound = rust_ir::AliasEqBound {
                value: ty.clone(),
                trait_bound: rust_ir::TraitBound {
                    trait_id: to_chalk_trait_id(projection_ty.trait_(db)),
                    args_no_self,
                },
                associated_ty_id: projection_ty.associated_ty_id,
                parameters,
            };
            Some(chalk_ir::Binders::new(
                binders,
                rust_ir::InlineBound::AliasEqBound(alias_eq_bound),
            ))
        }
        _ => None,
    }
}
