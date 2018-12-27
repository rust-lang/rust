use rustc::traits::{
    WellFormed,
    FromEnv,
    DomainGoal,
    GoalKind,
    Clause,
    Clauses,
    ProgramClause,
    ProgramClauseCategory,
    Environment,
};
use rustc::ty;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc_target::spec::abi;
use super::ChalkInferenceContext;
use crate::lowering::Lower;
use std::iter;

fn assemble_clauses_from_impls<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    trait_def_id: DefId,
    clauses: &mut Vec<Clause<'tcx>>
) {
    tcx.for_each_impl(trait_def_id, |impl_def_id| {
        clauses.extend(
            tcx.program_clauses_for(impl_def_id)
                .into_iter()
                .cloned()
        );
    });
}

fn assemble_clauses_from_assoc_ty_values<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    trait_def_id: DefId,
    clauses: &mut Vec<Clause<'tcx>>
) {
    tcx.for_each_impl(trait_def_id, |impl_def_id| {
        for def_id in tcx.associated_item_def_ids(impl_def_id).iter() {
            clauses.extend(
                tcx.program_clauses_for(*def_id)
                    .into_iter()
                    .cloned()
            );
        }
    });
}

fn program_clauses_for_raw_ptr<'tcx>(tcx: ty::TyCtxt<'_, '_, 'tcx>) -> Clauses<'tcx> {
    let ty = ty::Bound(
        ty::INNERMOST,
        ty::BoundVar::from_u32(0).into()
    );
    let ty = tcx.mk_ty(ty);

    let ptr_ty = tcx.mk_ptr(ty::TypeAndMut {
        ty,
        mutbl: hir::Mutability::MutImmutable,
    });

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(ptr_ty)),
        hypotheses: ty::List::empty(),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<T> { WellFormed(*const T). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn program_clauses_for_fn_ptr<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    arity_and_output: usize,
    variadic: bool,
    unsafety: hir::Unsafety,
    abi: abi::Abi
) -> Clauses<'tcx> {
    let inputs_and_output = tcx.mk_type_list(
        (0..arity_and_output).into_iter()
            .map(|i| ty::BoundVar::from(i))
            // DebruijnIndex(1) because we are going to inject these in a `PolyFnSig`
            .map(|var| tcx.mk_ty(ty::Bound(ty::DebruijnIndex::from(1usize), var.into())))
    );

    let fn_sig = ty::Binder::bind(ty::FnSig {
        inputs_and_output,
        variadic,
        unsafety,
        abi,
    });
    let fn_ptr = tcx.mk_fn_ptr(fn_sig);

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(fn_ptr)),
        hypotheses: ty::List::empty(),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall <T1, ..., Tn+1> { WellFormed(for<> fn(T1, ..., Tn) -> Tn+1). }`
    // where `n + 1` == `arity_and_output`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn program_clauses_for_slice<'tcx>(tcx: ty::TyCtxt<'_, '_, 'tcx>) -> Clauses<'tcx> {
    let ty = ty::Bound(
        ty::INNERMOST,
        ty::BoundVar::from_u32(0).into()
    );
    let ty = tcx.mk_ty(ty);

    let slice_ty = tcx.mk_slice(ty);

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };
    let sized_implemented = ty::TraitRef {
        def_id: sized_trait,
        substs: tcx.mk_substs_trait(ty, ty::List::empty()),
    };
    let sized_implemented: DomainGoal = ty::TraitPredicate {
        trait_ref: sized_implemented
    }.lower();

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(slice_ty)),
        hypotheses: tcx.mk_goals(
            iter::once(tcx.mk_goal(GoalKind::DomainGoal(sized_implemented)))
        ),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<T> { WellFormed([T]) :- Implemented(T: Sized). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn program_clauses_for_array<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    length: &'tcx ty::Const<'tcx>
) -> Clauses<'tcx> {
    let ty = ty::Bound(
        ty::INNERMOST,
        ty::BoundVar::from_u32(0).into()
    );
    let ty = tcx.mk_ty(ty);

    let array_ty = tcx.mk_ty(ty::Array(ty, length));

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };
    let sized_implemented = ty::TraitRef {
        def_id: sized_trait,
        substs: tcx.mk_substs_trait(ty, ty::List::empty()),
    };
    let sized_implemented: DomainGoal = ty::TraitPredicate {
        trait_ref: sized_implemented
    }.lower();

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(array_ty)),
        hypotheses: tcx.mk_goals(
            iter::once(tcx.mk_goal(GoalKind::DomainGoal(sized_implemented)))
        ),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<T> { WellFormed([T; length]) :- Implemented(T: Sized). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn program_clauses_for_tuple<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    arity: usize
) -> Clauses<'tcx> {
    let type_list = tcx.mk_type_list(
        (0..arity).into_iter()
            .map(|i| ty::BoundVar::from(i))
            .map(|var| tcx.mk_ty(ty::Bound(ty::INNERMOST, var.into())))
    );

    let tuple_ty = tcx.mk_ty(ty::Tuple(type_list));

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };
    let sized_implemented = type_list[0..arity - 1].iter()
        .map(|ty| ty::TraitRef {
            def_id: sized_trait,
            substs: tcx.mk_substs_trait(*ty, ty::List::empty()),
        })
        .map(|trait_ref| ty::TraitPredicate { trait_ref })
        .map(|predicate| predicate.lower());

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(tuple_ty)),
        hypotheses: tcx.mk_goals(
            sized_implemented.map(|domain_goal| {
                tcx.mk_goal(GoalKind::DomainGoal(domain_goal))
            })
        ),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // ```
    // forall<T1, ..., Tn-1, Tn> {
    //     WellFormed((T1, ..., Tn)) :-
    //         Implemented(T1: Sized),
    //         ...
    //         Implemented(Tn-1: Sized).
    // }
    // ```
    tcx.mk_clauses(iter::once(wf_clause))
}

fn program_clauses_for_ref<'tcx>(tcx: ty::TyCtxt<'_, '_, 'tcx>) -> Clauses<'tcx> {
    let region = tcx.mk_region(
        ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(0))
    );
    let ty = tcx.mk_ty(
        ty::Bound(ty::INNERMOST, ty::BoundVar::from_u32(1).into())
    );

    let ref_ty = tcx.mk_ref(region, ty::TypeAndMut {
        ty,
        mutbl: hir::Mutability::MutImmutable,
    });

    let outlives: DomainGoal = ty::OutlivesPredicate(ty, region).lower();
    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(ref_ty)),
        hypotheses: tcx.mk_goals(
            iter::once(tcx.mk_goal(outlives.into_goal()))
        ),
        category: ProgramClauseCategory::ImpliedBound,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<'a, T> { WellFormed(&'a T) :- Outlives(T: 'a). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

impl ChalkInferenceContext<'cx, 'gcx, 'tcx> {
    pub(super) fn program_clauses_impl(
        &self,
        environment: &Environment<'tcx>,
        goal: &DomainGoal<'tcx>,
    ) -> Vec<Clause<'tcx>> {
        use rustc::traits::WhereClause::*;

        let mut clauses = match goal {
            DomainGoal::Holds(Implemented(trait_predicate)) => {
                // These come from:
                // * implementations of the trait itself (rule `Implemented-From-Impl`)
                // * the trait decl (rule `Implemented-From-Env`)

                let mut clauses = vec![];
                assemble_clauses_from_impls(
                    self.infcx.tcx,
                    trait_predicate.def_id(),
                    &mut clauses
                );

                // FIXME: we need to add special rules for builtin impls:
                // * `Copy` / `Clone`
                // * `Sized`
                // * `Unsize`
                // * `Generator`
                // * `FnOnce` / `FnMut` / `Fn`
                // * trait objects
                // * auto traits

                // Rule `Implemented-From-Env` will be computed from the environment.
                clauses
            }

            DomainGoal::Holds(ProjectionEq(projection_predicate)) => {
                // These come from:
                // * the assoc type definition (rule `ProjectionEq-Placeholder`)
                // * normalization of the assoc ty values (rule `ProjectionEq-Normalize`)
                // * implied bounds from trait definitions (rule `Implied-Bound-From-Trait`)
                // * implied bounds from type definitions (rule `Implied-Bound-From-Type`)

                let clauses = self.infcx.tcx.program_clauses_for(
                    projection_predicate.projection_ty.item_def_id
                ).into_iter()

                    // only select `ProjectionEq-Placeholder` and `ProjectionEq-Normalize`
                    .filter(|clause| clause.category() == ProgramClauseCategory::Other)

                    .cloned()
                    .collect::<Vec<_>>();

                // Rules `Implied-Bound-From-Trait` and `Implied-Bound-From-Type` will be computed
                // from the environment.
                clauses
            }

            DomainGoal::Holds(RegionOutlives(..)) => {
                // These come from:
                // * implied bounds from trait definitions (rule `Implied-Bound-From-Trait`)
                // * implied bounds from type definitions (rule `Implied-Bound-From-Type`)

                // All of these rules are computed in the environment.
                vec![]
            }

            DomainGoal::Holds(TypeOutlives(..)) => {
                // These come from:
                // * implied bounds from trait definitions (rule `Implied-Bound-From-Trait`)
                // * implied bounds from type definitions (rule `Implied-Bound-From-Type`)

                // All of these rules are computed in the environment.
                vec![]
            }

            DomainGoal::WellFormed(WellFormed::Trait(trait_predicate)) => {
                // These come from -- the trait decl (rule `WellFormed-TraitRef`).
                self.infcx.tcx.program_clauses_for(trait_predicate.def_id())
                    .into_iter()

                    // only select `WellFormed-TraitRef`
                    .filter(|clause| clause.category() == ProgramClauseCategory::WellFormed)

                    .cloned()
                    .collect()
            }

            DomainGoal::WellFormed(WellFormed::Ty(ty)) => {
                // These come from:
                // * the associated type definition if `ty` refers to an unnormalized
                //   associated type (rule `WellFormed-AssocTy`)
                // * custom rules for built-in types
                // * the type definition otherwise (rule `WellFormed-Type`)
                let clauses = match ty.sty {
                    ty::Projection(data) => {
                        self.infcx.tcx.program_clauses_for(data.item_def_id)
                    }

                    // These types are always WF and non-parametric.
                    ty::Bool |
                    ty::Char |
                    ty::Int(..) |
                    ty::Uint(..) |
                    ty::Float(..) |
                    ty::Str |
                    ty::Never => {
                        let wf_clause = ProgramClause {
                            goal: DomainGoal::WellFormed(WellFormed::Ty(ty)),
                            hypotheses: ty::List::empty(),
                            category: ProgramClauseCategory::WellFormed,
                        };
                        let wf_clause = Clause::ForAll(ty::Binder::dummy(wf_clause));

                        self.infcx.tcx.mk_clauses(iter::once(wf_clause))
                    }

                    // Always WF (recall that we do not check for parameters to be WF).
                    ty::RawPtr(..) => program_clauses_for_raw_ptr(self.infcx.tcx),

                    // Always WF (recall that we do not check for parameters to be WF).
                    ty::FnPtr(fn_ptr) => {
                        let fn_ptr = fn_ptr.skip_binder();
                        program_clauses_for_fn_ptr(
                            self.infcx.tcx,
                            fn_ptr.inputs_and_output.len(),
                            fn_ptr.variadic,
                            fn_ptr.unsafety,
                            fn_ptr.abi
                        )
                    }

                    // WF if inner type is `Sized`.
                    ty::Slice(..) => program_clauses_for_slice(self.infcx.tcx),

                    // WF if inner type is `Sized`.
                    ty::Array(_, length) => program_clauses_for_array(self.infcx.tcx, length),

                    // WF if all types but the last one are `Sized`.
                    ty::Tuple(types) => program_clauses_for_tuple(
                        self.infcx.tcx,
                        types.len()
                    ),

                    // WF if `sub_ty` outlives `region`.
                    ty::Ref(..) => program_clauses_for_ref(self.infcx.tcx),

                    ty::Dynamic(..) => {
                        // FIXME: no rules yet for trait objects
                        ty::List::empty()
                    }

                    ty::Adt(def, ..) => {
                        self.infcx.tcx.program_clauses_for(def.did)
                    }

                    ty::Foreign(def_id) |
                    ty::FnDef(def_id, ..) |
                    ty::Closure(def_id, ..) |
                    ty::Generator(def_id, ..) |
                    ty::Opaque(def_id, ..) => {
                        self.infcx.tcx.program_clauses_for(def_id)
                    }

                    ty::GeneratorWitness(..) |
                    ty::Placeholder(..) |
                    ty::UnnormalizedProjection(..) |
                    ty::Infer(..) |
                    ty::Bound(..) |
                    ty::Param(..) |
                    ty::Error => {
                        bug!("unexpected type {:?}", ty)
                    }
                };

                clauses.into_iter()
                    .filter(|clause| clause.category() == ProgramClauseCategory::WellFormed)
                    .cloned()
                    .collect()
            }

            DomainGoal::FromEnv(FromEnv::Trait(..)) => {
                // These come from:
                // * implied bounds from trait definitions (rule `Implied-Bound-From-Trait`)
                // * implied bounds from type definitions (rule `Implied-Bound-From-Type`)
                // * implied bounds from assoc type defs (rules `Implied-Trait-From-AssocTy`,
                //   `Implied-Bound-From-AssocTy` and `Implied-WC-From-AssocTy`)

                // All of these rules are computed in the environment.
                vec![]
            }

            DomainGoal::FromEnv(FromEnv::Ty(..)) => {
                // There are no `FromEnv::Ty(..) :- ...` rules (this predicate only
                // comes from the environment).
                vec![]
            }

            DomainGoal::Normalize(projection_predicate) => {
                // These come from -- assoc ty values (rule `Normalize-From-Impl`).
                let mut clauses = vec![];

                assemble_clauses_from_assoc_ty_values(
                    self.infcx.tcx,
                    projection_predicate.projection_ty.trait_ref(self.infcx.tcx).def_id,
                    &mut clauses
                );

                clauses
            }
        };

        let environment = self.infcx.tcx.lift_to_global(environment)
            .expect("environment is not global");
        clauses.extend(
            self.infcx.tcx.program_clauses_for_env(environment)
                .into_iter()
                .cloned()
        );
        clauses
    }
}
