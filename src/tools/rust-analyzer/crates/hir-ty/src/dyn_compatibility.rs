//! Compute the dyn-compatibility of a trait

use std::ops::ControlFlow;

use chalk_ir::{
    DebruijnIndex,
    cast::Cast,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
};
use chalk_solve::rust_ir::InlineBound;
use hir_def::{
    AssocItemId, ConstId, CrateRootModuleId, FunctionId, GenericDefId, HasModule, TraitId,
    TypeAliasId, lang_item::LangItem, signatures::TraitFlags,
};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::{
    AliasEq, AliasTy, Binders, BoundVar, CallableSig, GoalData, ImplTraitId, Interner, OpaqueTyId,
    ProjectionTyExt, Solution, Substitution, TraitRef, Ty, TyKind, WhereClause, all_super_traits,
    db::HirDatabase,
    from_assoc_type_id, from_chalk_trait_id,
    generics::{generics, trait_self_param_idx},
    to_chalk_trait_id,
    utils::elaborate_clause_supertraits,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynCompatibilityViolation {
    SizedSelf,
    SelfReferential,
    Method(FunctionId, MethodViolationCode),
    AssocConst(ConstId),
    GAT(TypeAliasId),
    // This doesn't exist in rustc, but added for better visualization
    HasNonCompatibleSuperTrait(TraitId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MethodViolationCode {
    StaticMethod,
    ReferencesSelfInput,
    ReferencesSelfOutput,
    ReferencesImplTraitInTrait,
    AsyncFn,
    WhereClauseReferencesSelf,
    Generic,
    UndispatchableReceiver,
}

pub fn dyn_compatibility(
    db: &dyn HirDatabase,
    trait_: TraitId,
) -> Option<DynCompatibilityViolation> {
    for super_trait in all_super_traits(db, trait_).into_iter().skip(1).rev() {
        if db.dyn_compatibility_of_trait(super_trait).is_some() {
            return Some(DynCompatibilityViolation::HasNonCompatibleSuperTrait(super_trait));
        }
    }

    db.dyn_compatibility_of_trait(trait_)
}

pub fn dyn_compatibility_with_callback<F>(
    db: &dyn HirDatabase,
    trait_: TraitId,
    cb: &mut F,
) -> ControlFlow<()>
where
    F: FnMut(DynCompatibilityViolation) -> ControlFlow<()>,
{
    for super_trait in all_super_traits(db, trait_).into_iter().skip(1).rev() {
        if db.dyn_compatibility_of_trait(super_trait).is_some() {
            cb(DynCompatibilityViolation::HasNonCompatibleSuperTrait(trait_))?;
        }
    }

    dyn_compatibility_of_trait_with_callback(db, trait_, cb)
}

pub fn dyn_compatibility_of_trait_with_callback<F>(
    db: &dyn HirDatabase,
    trait_: TraitId,
    cb: &mut F,
) -> ControlFlow<()>
where
    F: FnMut(DynCompatibilityViolation) -> ControlFlow<()>,
{
    // Check whether this has a `Sized` bound
    if generics_require_sized_self(db, trait_.into()) {
        cb(DynCompatibilityViolation::SizedSelf)?;
    }

    // Check if there exist bounds that referencing self
    if predicates_reference_self(db, trait_) {
        cb(DynCompatibilityViolation::SelfReferential)?;
    }
    if bounds_reference_self(db, trait_) {
        cb(DynCompatibilityViolation::SelfReferential)?;
    }

    // rustc checks for non-lifetime binders here, but we don't support HRTB yet

    let trait_data = db.trait_items(trait_);
    for (_, assoc_item) in &trait_data.items {
        dyn_compatibility_violation_for_assoc_item(db, trait_, *assoc_item, cb)?;
    }

    ControlFlow::Continue(())
}

pub fn dyn_compatibility_of_trait_query(
    db: &dyn HirDatabase,
    trait_: TraitId,
) -> Option<DynCompatibilityViolation> {
    let mut res = None;
    _ = dyn_compatibility_of_trait_with_callback(db, trait_, &mut |osv| {
        res = Some(osv);
        ControlFlow::Break(())
    });

    res
}

pub fn generics_require_sized_self(db: &dyn HirDatabase, def: GenericDefId) -> bool {
    let krate = def.module(db).krate();
    let Some(sized) = LangItem::Sized.resolve_trait(db, krate) else {
        return false;
    };

    let Some(trait_self_param_idx) = trait_self_param_idx(db, def) else {
        return false;
    };

    let predicates = &*db.generic_predicates(def);
    let predicates = predicates.iter().map(|p| p.skip_binders().skip_binders().clone());
    elaborate_clause_supertraits(db, predicates).any(|pred| match pred {
        WhereClause::Implemented(trait_ref) => {
            if from_chalk_trait_id(trait_ref.trait_id) == sized {
                if let TyKind::BoundVar(it) =
                    *trait_ref.self_type_parameter(Interner).kind(Interner)
                {
                    // Since `generic_predicates` is `Binder<Binder<..>>`, the `DebrujinIndex` of
                    // self-parameter is `1`
                    return it
                        .index_if_bound_at(DebruijnIndex::ONE)
                        .is_some_and(|idx| idx == trait_self_param_idx);
                }
            }
            false
        }
        _ => false,
    })
}

// rustc gathers all the spans that references `Self` for error rendering,
// but we don't have good way to render such locations.
// So, just return single boolean value for existence of such `Self` reference
fn predicates_reference_self(db: &dyn HirDatabase, trait_: TraitId) -> bool {
    db.generic_predicates(trait_.into())
        .iter()
        .any(|pred| predicate_references_self(db, trait_, pred, AllowSelfProjection::No))
}

// Same as the above, `predicates_reference_self`
fn bounds_reference_self(db: &dyn HirDatabase, trait_: TraitId) -> bool {
    let trait_data = db.trait_items(trait_);
    trait_data
        .items
        .iter()
        .filter_map(|(_, it)| match *it {
            AssocItemId::TypeAliasId(id) => {
                let assoc_ty_data = db.associated_ty_data(id);
                Some(assoc_ty_data)
            }
            _ => None,
        })
        .any(|assoc_ty_data| {
            assoc_ty_data.binders.skip_binders().bounds.iter().any(|bound| {
                let def = from_assoc_type_id(assoc_ty_data.id).into();
                match bound.skip_binders() {
                    InlineBound::TraitBound(it) => it.args_no_self.iter().any(|arg| {
                        contains_illegal_self_type_reference(
                            db,
                            def,
                            trait_,
                            arg,
                            DebruijnIndex::ONE,
                            AllowSelfProjection::Yes,
                        )
                    }),
                    InlineBound::AliasEqBound(it) => it.parameters.iter().any(|arg| {
                        contains_illegal_self_type_reference(
                            db,
                            def,
                            trait_,
                            arg,
                            DebruijnIndex::ONE,
                            AllowSelfProjection::Yes,
                        )
                    }),
                }
            })
        })
}

#[derive(Clone, Copy)]
enum AllowSelfProjection {
    Yes,
    No,
}

fn predicate_references_self(
    db: &dyn HirDatabase,
    trait_: TraitId,
    predicate: &Binders<Binders<WhereClause>>,
    allow_self_projection: AllowSelfProjection,
) -> bool {
    match predicate.skip_binders().skip_binders() {
        WhereClause::Implemented(trait_ref) => {
            trait_ref.substitution.iter(Interner).skip(1).any(|arg| {
                contains_illegal_self_type_reference(
                    db,
                    trait_.into(),
                    trait_,
                    arg,
                    DebruijnIndex::ONE,
                    allow_self_projection,
                )
            })
        }
        WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(proj), .. }) => {
            proj.substitution.iter(Interner).skip(1).any(|arg| {
                contains_illegal_self_type_reference(
                    db,
                    trait_.into(),
                    trait_,
                    arg,
                    DebruijnIndex::ONE,
                    allow_self_projection,
                )
            })
        }
        _ => false,
    }
}

fn contains_illegal_self_type_reference<T: TypeVisitable<Interner>>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    trait_: TraitId,
    t: &T,
    outer_binder: DebruijnIndex,
    allow_self_projection: AllowSelfProjection,
) -> bool {
    let Some(trait_self_param_idx) = trait_self_param_idx(db, def) else {
        return false;
    };
    struct IllegalSelfTypeVisitor<'a> {
        db: &'a dyn HirDatabase,
        trait_: TraitId,
        super_traits: Option<SmallVec<[TraitId; 4]>>,
        trait_self_param_idx: usize,
        allow_self_projection: AllowSelfProjection,
    }
    impl TypeVisitor<Interner> for IllegalSelfTypeVisitor<'_> {
        type BreakTy = ();

        fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn visit_ty(&mut self, ty: &Ty, outer_binder: DebruijnIndex) -> ControlFlow<Self::BreakTy> {
            match ty.kind(Interner) {
                TyKind::BoundVar(BoundVar { debruijn, index }) => {
                    if *debruijn == outer_binder && *index == self.trait_self_param_idx {
                        ControlFlow::Break(())
                    } else {
                        ty.super_visit_with(self.as_dyn(), outer_binder)
                    }
                }
                TyKind::Alias(AliasTy::Projection(proj)) => match self.allow_self_projection {
                    AllowSelfProjection::Yes => {
                        let trait_ = proj.trait_(self.db);
                        if self.super_traits.is_none() {
                            self.super_traits = Some(all_super_traits(self.db, self.trait_));
                        }
                        if self.super_traits.as_ref().is_some_and(|s| s.contains(&trait_)) {
                            ControlFlow::Continue(())
                        } else {
                            ty.super_visit_with(self.as_dyn(), outer_binder)
                        }
                    }
                    AllowSelfProjection::No => ty.super_visit_with(self.as_dyn(), outer_binder),
                },
                _ => ty.super_visit_with(self.as_dyn(), outer_binder),
            }
        }

        fn visit_const(
            &mut self,
            constant: &chalk_ir::Const<Interner>,
            outer_binder: DebruijnIndex,
        ) -> std::ops::ControlFlow<Self::BreakTy> {
            constant.data(Interner).ty.super_visit_with(self.as_dyn(), outer_binder)
        }
    }

    let mut visitor = IllegalSelfTypeVisitor {
        db,
        trait_,
        super_traits: None,
        trait_self_param_idx,
        allow_self_projection,
    };
    t.visit_with(visitor.as_dyn(), outer_binder).is_break()
}

fn dyn_compatibility_violation_for_assoc_item<F>(
    db: &dyn HirDatabase,
    trait_: TraitId,
    item: AssocItemId,
    cb: &mut F,
) -> ControlFlow<()>
where
    F: FnMut(DynCompatibilityViolation) -> ControlFlow<()>,
{
    // Any item that has a `Self : Sized` requisite is otherwise
    // exempt from the regulations.
    if generics_require_sized_self(db, item.into()) {
        return ControlFlow::Continue(());
    }

    match item {
        AssocItemId::ConstId(it) => cb(DynCompatibilityViolation::AssocConst(it)),
        AssocItemId::FunctionId(it) => {
            virtual_call_violations_for_method(db, trait_, it, &mut |mvc| {
                cb(DynCompatibilityViolation::Method(it, mvc))
            })
        }
        AssocItemId::TypeAliasId(it) => {
            let def_map = CrateRootModuleId::from(trait_.krate(db)).def_map(db);
            if def_map.is_unstable_feature_enabled(&intern::sym::generic_associated_type_extended) {
                ControlFlow::Continue(())
            } else {
                let generic_params = db.generic_params(item.into());
                if !generic_params.is_empty() {
                    cb(DynCompatibilityViolation::GAT(it))
                } else {
                    ControlFlow::Continue(())
                }
            }
        }
    }
}

fn virtual_call_violations_for_method<F>(
    db: &dyn HirDatabase,
    trait_: TraitId,
    func: FunctionId,
    cb: &mut F,
) -> ControlFlow<()>
where
    F: FnMut(MethodViolationCode) -> ControlFlow<()>,
{
    let func_data = db.function_signature(func);
    if !func_data.has_self_param() {
        cb(MethodViolationCode::StaticMethod)?;
    }

    if func_data.is_async() {
        cb(MethodViolationCode::AsyncFn)?;
    }

    let sig = db.callable_item_signature(func.into());
    if sig.skip_binders().params().iter().skip(1).any(|ty| {
        contains_illegal_self_type_reference(
            db,
            func.into(),
            trait_,
            ty,
            DebruijnIndex::INNERMOST,
            AllowSelfProjection::Yes,
        )
    }) {
        cb(MethodViolationCode::ReferencesSelfInput)?;
    }

    if contains_illegal_self_type_reference(
        db,
        func.into(),
        trait_,
        sig.skip_binders().ret(),
        DebruijnIndex::INNERMOST,
        AllowSelfProjection::Yes,
    ) {
        cb(MethodViolationCode::ReferencesSelfOutput)?;
    }

    if !func_data.is_async() {
        if let Some(mvc) = contains_illegal_impl_trait_in_trait(db, &sig) {
            cb(mvc)?;
        }
    }

    let generic_params = db.generic_params(func.into());
    if generic_params.len_type_or_consts() > 0 {
        cb(MethodViolationCode::Generic)?;
    }

    if func_data.has_self_param() && !receiver_is_dispatchable(db, trait_, func, &sig) {
        cb(MethodViolationCode::UndispatchableReceiver)?;
    }

    let predicates = &*db.generic_predicates_without_parent(func.into());
    let trait_self_idx = trait_self_param_idx(db, func.into());
    for pred in predicates {
        let pred = pred.skip_binders().skip_binders();

        if matches!(pred, WhereClause::TypeOutlives(_)) {
            continue;
        }

        // Allow `impl AutoTrait` predicates
        if let WhereClause::Implemented(TraitRef { trait_id, substitution }) = pred {
            let trait_data = db.trait_signature(from_chalk_trait_id(*trait_id));
            if trait_data.flags.contains(TraitFlags::AUTO)
                && substitution
                    .as_slice(Interner)
                    .first()
                    .and_then(|arg| arg.ty(Interner))
                    .and_then(|ty| ty.bound_var(Interner))
                    .is_some_and(|b| {
                        b.debruijn == DebruijnIndex::ONE && Some(b.index) == trait_self_idx
                    })
            {
                continue;
            }
        }

        if contains_illegal_self_type_reference(
            db,
            func.into(),
            trait_,
            pred,
            DebruijnIndex::ONE,
            AllowSelfProjection::Yes,
        ) {
            cb(MethodViolationCode::WhereClauseReferencesSelf)?;
            break;
        }
    }

    ControlFlow::Continue(())
}

fn receiver_is_dispatchable(
    db: &dyn HirDatabase,
    trait_: TraitId,
    func: FunctionId,
    sig: &Binders<CallableSig>,
) -> bool {
    let Some(trait_self_idx) = trait_self_param_idx(db, func.into()) else {
        return false;
    };

    // `self: Self` can't be dispatched on, but this is already considered dyn-compatible
    // See rustc's comment on https://github.com/rust-lang/rust/blob/3f121b9461cce02a703a0e7e450568849dfaa074/compiler/rustc_trait_selection/src/traits/object_safety.rs#L433-L437
    if sig
        .skip_binders()
        .params()
        .first()
        .and_then(|receiver| receiver.bound_var(Interner))
        .is_some_and(|b| {
            b == BoundVar { debruijn: DebruijnIndex::INNERMOST, index: trait_self_idx }
        })
    {
        return true;
    }

    let placeholder_subst = generics(db, func.into()).placeholder_subst(db);

    let substituted_sig = sig.clone().substitute(Interner, &placeholder_subst);
    let Some(receiver_ty) = substituted_sig.params().first() else {
        return false;
    };

    let krate = func.module(db).krate();
    let traits = (
        LangItem::Unsize.resolve_trait(db, krate),
        LangItem::DispatchFromDyn.resolve_trait(db, krate),
    );
    let (Some(unsize_did), Some(dispatch_from_dyn_did)) = traits else {
        return false;
    };

    // Type `U`
    let unsized_self_ty =
        TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::U32)).intern(Interner);
    // `Receiver[Self => U]`
    let Some(unsized_receiver_ty) = receiver_for_self_ty(db, func, unsized_self_ty.clone()) else {
        return false;
    };

    let self_ty = placeholder_subst.as_slice(Interner)[trait_self_idx].assert_ty_ref(Interner);
    let unsized_predicate = WhereClause::Implemented(TraitRef {
        trait_id: to_chalk_trait_id(unsize_did),
        substitution: Substitution::from_iter(Interner, [self_ty.clone(), unsized_self_ty.clone()]),
    });
    let trait_predicate = WhereClause::Implemented(TraitRef {
        trait_id: to_chalk_trait_id(trait_),
        substitution: Substitution::from_iter(
            Interner,
            std::iter::once(unsized_self_ty.cast(Interner))
                .chain(placeholder_subst.iter(Interner).skip(1).cloned()),
        ),
    });

    let generic_predicates = &*db.generic_predicates(func.into());

    let clauses = std::iter::once(unsized_predicate)
        .chain(std::iter::once(trait_predicate))
        .chain(generic_predicates.iter().map(|pred| {
            pred.clone().substitute(Interner, &placeholder_subst).into_value_and_skipped_binders().0
        }))
        .map(|pred| {
            pred.cast::<chalk_ir::ProgramClause<Interner>>(Interner).into_from_env_clause(Interner)
        });
    let env = chalk_ir::Environment::new(Interner).add_clauses(Interner, clauses);

    let obligation = WhereClause::Implemented(TraitRef {
        trait_id: to_chalk_trait_id(dispatch_from_dyn_did),
        substitution: Substitution::from_iter(Interner, [receiver_ty.clone(), unsized_receiver_ty]),
    });
    let goal = GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(obligation)).intern(Interner);

    let in_env = chalk_ir::InEnvironment::new(&env, goal);

    let mut table = chalk_solve::infer::InferenceTable::<Interner>::new();
    let canonicalized = table.canonicalize(Interner, in_env);
    let solution = db.trait_solve(krate, None, canonicalized.quantified);

    matches!(solution, Some(Solution::Unique(_)))
}

fn receiver_for_self_ty(db: &dyn HirDatabase, func: FunctionId, ty: Ty) -> Option<Ty> {
    let generics = generics(db, func.into());
    let trait_self_idx = trait_self_param_idx(db, func.into())?;
    let subst = generics.placeholder_subst(db);
    let subst = Substitution::from_iter(
        Interner,
        subst.iter(Interner).enumerate().map(|(idx, arg)| {
            if idx == trait_self_idx { ty.clone().cast(Interner) } else { arg.clone() }
        }),
    );
    let sig = db.callable_item_signature(func.into());
    let sig = sig.substitute(Interner, &subst);
    sig.params_and_return.first().cloned()
}

fn contains_illegal_impl_trait_in_trait(
    db: &dyn HirDatabase,
    sig: &Binders<CallableSig>,
) -> Option<MethodViolationCode> {
    struct OpaqueTypeCollector(FxHashSet<OpaqueTyId>);

    impl TypeVisitor<Interner> for OpaqueTypeCollector {
        type BreakTy = ();

        fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn visit_ty(&mut self, ty: &Ty, outer_binder: DebruijnIndex) -> ControlFlow<Self::BreakTy> {
            if let TyKind::OpaqueType(opaque_ty_id, _) = ty.kind(Interner) {
                self.0.insert(*opaque_ty_id);
            }
            ty.super_visit_with(self.as_dyn(), outer_binder)
        }
    }

    let ret = sig.skip_binders().ret();
    let mut visitor = OpaqueTypeCollector(FxHashSet::default());
    _ = ret.visit_with(visitor.as_dyn(), DebruijnIndex::INNERMOST);

    // Since we haven't implemented RPITIT in proper way like rustc yet,
    // just check whether `ret` contains RPIT for now
    for opaque_ty in visitor.0 {
        let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty.into());
        if matches!(impl_trait_id, ImplTraitId::ReturnTypeImplTrait(..)) {
            return Some(MethodViolationCode::ReferencesImplTraitInTrait);
        }
    }

    None
}

#[cfg(test)]
mod tests;
