//! Compute the dyn-compatibility of a trait

use std::ops::ControlFlow;

use chalk_ir::{
    DebruijnIndex,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
};
use chalk_solve::rust_ir::InlineBound;
use hir_def::{
    AssocItemId, ConstId, CrateRootModuleId, FunctionId, GenericDefId, HasModule, TraitId,
    TypeAliasId, lang_item::LangItem, signatures::TraitFlags,
};
use intern::Symbol;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    AliasTyKind, ClauseKind, PredicatePolarity, TypeSuperVisitable as _, TypeVisitable as _,
    Upcast,
    inherent::{IntoKind, SliceLike},
};
use smallvec::SmallVec;

use crate::{
    AliasEq, AliasTy, Binders, BoundVar, ImplTraitId, Interner, ProjectionTyExt, Ty, TyKind,
    WhereClause, all_super_traits,
    db::{HirDatabase, InternedOpaqueTyId},
    from_assoc_type_id, from_chalk_trait_id,
    generics::trait_self_param_idx,
    next_solver::{
        Clauses, DbInterner, GenericArgs, ParamEnv, SolverDefId, TraitPredicate, TypingMode,
        infer::DbInternerInferExt, mk_param,
    },
    traits::next_trait_solve_in_ctxt,
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

    let trait_data = trait_.trait_items(db);
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
            if from_chalk_trait_id(trait_ref.trait_id) == sized
                && let TyKind::BoundVar(it) =
                    *trait_ref.self_type_parameter(Interner).kind(Interner)
            {
                // Since `generic_predicates` is `Binder<Binder<..>>`, the `DebrujinIndex` of
                // self-parameter is `1`
                return it
                    .index_if_bound_at(DebruijnIndex::ONE)
                    .is_some_and(|idx| idx == trait_self_param_idx);
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
    let trait_data = trait_.trait_items(db);
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

fn contains_illegal_self_type_reference_ns<
    'db,
    T: rustc_type_ir::TypeVisitable<DbInterner<'db>>,
>(
    db: &'db dyn HirDatabase,
    trait_: TraitId,
    t: &T,
    allow_self_projection: AllowSelfProjection,
) -> bool {
    struct IllegalSelfTypeVisitor<'db> {
        db: &'db dyn HirDatabase,
        trait_: TraitId,
        super_traits: Option<SmallVec<[TraitId; 4]>>,
        allow_self_projection: AllowSelfProjection,
    }
    impl<'db> rustc_type_ir::TypeVisitor<DbInterner<'db>> for IllegalSelfTypeVisitor<'db> {
        type Result = ControlFlow<()>;

        fn visit_ty(
            &mut self,
            ty: <DbInterner<'db> as rustc_type_ir::Interner>::Ty,
        ) -> Self::Result {
            match ty.kind() {
                rustc_type_ir::TyKind::Param(param) if param.index == 0 => ControlFlow::Break(()),
                rustc_type_ir::TyKind::Param(_) => ControlFlow::Continue(()),
                rustc_type_ir::TyKind::Alias(AliasTyKind::Projection, proj) => match self
                    .allow_self_projection
                {
                    AllowSelfProjection::Yes => {
                        let trait_ = proj.trait_def_id(DbInterner::new_with(self.db, None, None));
                        let trait_ = match trait_ {
                            SolverDefId::TraitId(id) => id,
                            _ => unreachable!(),
                        };
                        if self.super_traits.is_none() {
                            self.super_traits = Some(all_super_traits(self.db, self.trait_));
                        }
                        if self.super_traits.as_ref().is_some_and(|s| s.contains(&trait_)) {
                            ControlFlow::Continue(())
                        } else {
                            ty.super_visit_with(self)
                        }
                    }
                    AllowSelfProjection::No => ty.super_visit_with(self),
                },
                _ => ty.super_visit_with(self),
            }
        }
    }

    let mut visitor =
        IllegalSelfTypeVisitor { db, trait_, super_traits: None, allow_self_projection };
    t.visit_with(&mut visitor).is_break()
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

    let sig = db.callable_item_signature_ns(func.into());
    if sig.skip_binder().inputs().iter().skip(1).any(|ty| {
        contains_illegal_self_type_reference_ns(db, trait_, &ty, AllowSelfProjection::Yes)
    }) {
        cb(MethodViolationCode::ReferencesSelfInput)?;
    }

    if contains_illegal_self_type_reference_ns(
        db,
        trait_,
        &sig.skip_binder().output(),
        AllowSelfProjection::Yes,
    ) {
        cb(MethodViolationCode::ReferencesSelfOutput)?;
    }

    if !func_data.is_async()
        && let Some(mvc) = contains_illegal_impl_trait_in_trait(db, &sig)
    {
        cb(mvc)?;
    }

    let generic_params = db.generic_params(func.into());
    if generic_params.len_type_or_consts() > 0 {
        cb(MethodViolationCode::Generic)?;
    }

    if func_data.has_self_param() && !receiver_is_dispatchable(db, trait_, func, &sig) {
        cb(MethodViolationCode::UndispatchableReceiver)?;
    }

    let predicates = &*db.generic_predicates_without_parent_ns(func.into());
    for pred in predicates {
        let pred = pred.kind().skip_binder();

        if matches!(pred, ClauseKind::TypeOutlives(_)) {
            continue;
        }

        // Allow `impl AutoTrait` predicates
        let interner = DbInterner::new_with(db, Some(trait_.krate(db)), None);
        if let ClauseKind::Trait(TraitPredicate {
            trait_ref: pred_trait_ref,
            polarity: PredicatePolarity::Positive,
        }) = pred
            && let SolverDefId::TraitId(trait_id) = pred_trait_ref.def_id
            && let trait_data = db.trait_signature(trait_id)
            && trait_data.flags.contains(TraitFlags::AUTO)
            && pred_trait_ref.self_ty()
                == crate::next_solver::Ty::new(
                    interner,
                    rustc_type_ir::TyKind::Param(crate::next_solver::ParamTy { index: 0 }),
                )
        {
            continue;
        }

        if contains_illegal_self_type_reference_ns(db, trait_, &pred, AllowSelfProjection::Yes) {
            cb(MethodViolationCode::WhereClauseReferencesSelf)?;
            break;
        }
    }

    ControlFlow::Continue(())
}

fn receiver_is_dispatchable<'db>(
    db: &dyn HirDatabase,
    trait_: TraitId,
    func: FunctionId,
    sig: &crate::next_solver::EarlyBinder<
        'db,
        crate::next_solver::Binder<'db, rustc_type_ir::FnSig<DbInterner<'db>>>,
    >,
) -> bool {
    let sig = sig.instantiate_identity();

    let interner: DbInterner<'_> = DbInterner::new_with(db, Some(trait_.krate(db)), None);
    let self_param_ty = crate::next_solver::Ty::new(
        interner,
        rustc_type_ir::TyKind::Param(crate::next_solver::ParamTy { index: 0 }),
    );

    // `self: Self` can't be dispatched on, but this is already considered dyn-compatible
    // See rustc's comment on https://github.com/rust-lang/rust/blob/3f121b9461cce02a703a0e7e450568849dfaa074/compiler/rustc_trait_selection/src/traits/object_safety.rs#L433-L437
    if sig.inputs().iter().next().is_some_and(|p| p.skip_binder() == self_param_ty) {
        return true;
    }

    let Some(&receiver_ty) = sig.inputs().skip_binder().as_slice().first() else {
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

    let meta_sized_did = LangItem::MetaSized.resolve_trait(db, krate);
    let Some(meta_sized_did) = meta_sized_did else {
        return false;
    };

    // Type `U`
    let unsized_self_ty = crate::next_solver::Ty::new_param(interner, u32::MAX, Symbol::empty());
    // `Receiver[Self => U]`
    let unsized_receiver_ty = receiver_for_self_ty(interner, func, receiver_ty, unsized_self_ty);

    let param_env = {
        let generic_predicates = &*db.generic_predicates_ns(func.into());

        // Self: Unsize<U>
        let unsize_predicate = crate::next_solver::TraitRef::new(
            interner,
            SolverDefId::TraitId(unsize_did),
            [self_param_ty, unsized_self_ty],
        );

        // U: Trait<Arg1, ..., ArgN>
        let trait_def_id = SolverDefId::TraitId(trait_);
        let args = GenericArgs::for_item(interner, trait_def_id, |name, index, kind, _| {
            if index == 0 { unsized_self_ty.into() } else { mk_param(interner, index, name, kind) }
        });
        let trait_predicate =
            crate::next_solver::TraitRef::new_from_args(interner, trait_def_id, args);

        let meta_sized_predicate = crate::next_solver::TraitRef::new(
            interner,
            SolverDefId::TraitId(meta_sized_did),
            [unsized_self_ty],
        );

        ParamEnv {
            clauses: Clauses::new_from_iter(
                interner,
                generic_predicates.iter().copied().chain([
                    unsize_predicate.upcast(interner),
                    trait_predicate.upcast(interner),
                    meta_sized_predicate.upcast(interner),
                ]),
            ),
        }
    };

    // Receiver: DispatchFromDyn<Receiver[Self => U]>
    let predicate = crate::next_solver::TraitRef::new(
        interner,
        SolverDefId::TraitId(dispatch_from_dyn_did),
        [receiver_ty, unsized_receiver_ty],
    );
    let goal = crate::next_solver::Goal::new(interner, param_env, predicate);

    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());
    // the receiver is dispatchable iff the obligation holds
    let res = next_trait_solve_in_ctxt(&infcx, goal);
    res.map_or(false, |res| matches!(res.1, rustc_type_ir::solve::Certainty::Yes))
}

fn receiver_for_self_ty<'db>(
    interner: DbInterner<'db>,
    func: FunctionId,
    receiver_ty: crate::next_solver::Ty<'db>,
    self_ty: crate::next_solver::Ty<'db>,
) -> crate::next_solver::Ty<'db> {
    let args = crate::next_solver::GenericArgs::for_item(
        interner,
        SolverDefId::FunctionId(func),
        |name, index, kind, _| {
            if index == 0 { self_ty.into() } else { mk_param(interner, index, name, kind) }
        },
    );

    crate::next_solver::EarlyBinder::bind(receiver_ty).instantiate(interner, args)
}

fn contains_illegal_impl_trait_in_trait<'db>(
    db: &'db dyn HirDatabase,
    sig: &crate::next_solver::EarlyBinder<
        'db,
        crate::next_solver::Binder<'db, rustc_type_ir::FnSig<DbInterner<'db>>>,
    >,
) -> Option<MethodViolationCode> {
    struct OpaqueTypeCollector(FxHashSet<InternedOpaqueTyId>);

    impl<'db> rustc_type_ir::TypeVisitor<DbInterner<'db>> for OpaqueTypeCollector {
        type Result = ControlFlow<()>;

        fn visit_ty(
            &mut self,
            ty: <DbInterner<'db> as rustc_type_ir::Interner>::Ty,
        ) -> Self::Result {
            if let rustc_type_ir::TyKind::Alias(AliasTyKind::Opaque, op) = ty.kind() {
                let id = match op.def_id {
                    SolverDefId::InternedOpaqueTyId(id) => id,
                    _ => unreachable!(),
                };
                self.0.insert(id);
            }
            ty.super_visit_with(self)
        }
    }

    let ret = sig.skip_binder().output();
    let mut visitor = OpaqueTypeCollector(FxHashSet::default());
    _ = ret.visit_with(&mut visitor);

    // Since we haven't implemented RPITIT in proper way like rustc yet,
    // just check whether `ret` contains RPIT for now
    for opaque_ty in visitor.0 {
        let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty);
        if matches!(impl_trait_id, ImplTraitId::ReturnTypeImplTrait(..)) {
            return Some(MethodViolationCode::ReferencesImplTraitInTrait);
        }
    }

    None
}

#[cfg(test)]
mod tests;
