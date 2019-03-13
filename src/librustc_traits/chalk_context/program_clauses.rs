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
use rustc::ty::subst::{InternalSubsts, Subst};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc_target::spec::abi;
use super::ChalkInferenceContext;
use crate::lowering::Lower;
use crate::generic_types;
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

fn assemble_builtin_sized_impls<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    sized_def_id: DefId,
    ty: ty::Ty<'tcx>,
    clauses: &mut Vec<Clause<'tcx>>
) {
    let mut push_builtin_impl = |ty: ty::Ty<'tcx>, nested: &[ty::Ty<'tcx>]| {
        let clause = ProgramClause {
            goal: ty::TraitPredicate {
                trait_ref: ty::TraitRef {
                    def_id: sized_def_id,
                    substs: tcx.mk_substs_trait(ty, &[]),
                },
            }.lower(),
            hypotheses: tcx.mk_goals(
                nested.iter()
                    .cloned()
                    .map(|nested_ty| ty::TraitRef {
                        def_id: sized_def_id,
                        substs: tcx.mk_substs_trait(nested_ty, &[]),
                    })
                    .map(|trait_ref| ty::TraitPredicate { trait_ref })
                    .map(|pred| GoalKind::DomainGoal(pred.lower()))
                    .map(|goal_kind| tcx.mk_goal(goal_kind))
            ),
            category: ProgramClauseCategory::Other,
        };
        // Bind innermost bound vars that may exist in `ty` and `nested`.
        clauses.push(Clause::ForAll(ty::Binder::bind(clause)));
    };

    match &ty.sty {
        // Non parametric primitive types.
        ty::Bool |
        ty::Char |
        ty::Int(..) |
        ty::Uint(..) |
        ty::Float(..) |
        ty::Error |
        ty::Never => push_builtin_impl(ty, &[]),

        // These ones are always `Sized`.
        &ty::Array(_, length) => {
            push_builtin_impl(tcx.mk_ty(ty::Array(generic_types::bound(tcx, 0), length)), &[]);
        }
        ty::RawPtr(ptr) => {
            push_builtin_impl(generic_types::raw_ptr(tcx, ptr.mutbl), &[]);
        }
        &ty::Ref(_, _, mutbl) => {
            push_builtin_impl(generic_types::ref_ty(tcx, mutbl), &[]);
        }
        ty::FnPtr(fn_ptr) => {
            let fn_ptr = fn_ptr.skip_binder();
            let fn_ptr = generic_types::fn_ptr(
                tcx,
                fn_ptr.inputs_and_output.len(),
                fn_ptr.c_variadic,
                fn_ptr.unsafety,
                fn_ptr.abi
            );
            push_builtin_impl(fn_ptr, &[]);
        }
        &ty::FnDef(def_id, ..) => {
            push_builtin_impl(generic_types::fn_def(tcx, def_id), &[]);
        }
        &ty::Closure(def_id, ..) => {
            push_builtin_impl(generic_types::closure(tcx, def_id), &[]);
        }
        &ty::Generator(def_id, ..) => {
            push_builtin_impl(generic_types::generator(tcx, def_id), &[]);
        }

        // `Sized` if the last type is `Sized` (because else we will get a WF error anyway).
        &ty::Tuple(type_list) => {
            let type_list = generic_types::type_list(tcx, type_list.len());
            push_builtin_impl(tcx.mk_ty(ty::Tuple(type_list)), &**type_list);
        }

        // Struct def
        ty::Adt(adt_def, _) => {
            let substs = InternalSubsts::bound_vars_for_item(tcx, adt_def.did);
            let adt = tcx.mk_ty(ty::Adt(adt_def, substs));
            let sized_constraint = adt_def.sized_constraint(tcx)
                .iter()
                .map(|ty| ty.subst(tcx, substs))
                .collect::<Vec<_>>();
            push_builtin_impl(adt, &sized_constraint);
        }

        // Artificially trigger an ambiguity.
        ty::Infer(..) => {
            // Everybody can find at least two types to unify against:
            // general ty vars, int vars and float vars.
            push_builtin_impl(tcx.types.i32, &[]);
            push_builtin_impl(tcx.types.u32, &[]);
            push_builtin_impl(tcx.types.f32, &[]);
            push_builtin_impl(tcx.types.f64, &[]);
        }

        ty::Projection(_projection_ty) => {
            // FIXME: add builtin impls from the associated type values found in
            // trait impls of `projection_ty.trait_ref(tcx)`.
        }

        // The `Sized` bound can only come from the environment.
        ty::Param(..) |
        ty::Placeholder(..) |
        ty::UnnormalizedProjection(..) => (),

        // Definitely not `Sized`.
        ty::Foreign(..) |
        ty::Str |
        ty::Slice(..) |
        ty::Dynamic(..) |
        ty::Opaque(..) => (),

        ty::Bound(..) |
        ty::GeneratorWitness(..) => bug!("unexpected type {:?}", ty),
    }
}

fn wf_clause_for_raw_ptr<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    mutbl: hir::Mutability
) -> Clauses<'tcx> {
    let ptr_ty = generic_types::raw_ptr(tcx, mutbl);

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(ptr_ty)),
        hypotheses: ty::List::empty(),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::Implies(wf_clause);

    // `forall<T> { WellFormed(*const T). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn wf_clause_for_fn_ptr<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    arity_and_output: usize,
    c_variadic: bool,
    unsafety: hir::Unsafety,
    abi: abi::Abi
) -> Clauses<'tcx> {
    let fn_ptr = generic_types::fn_ptr(tcx, arity_and_output, c_variadic, unsafety, abi);

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

fn wf_clause_for_slice<'tcx>(tcx: ty::TyCtxt<'_, '_, 'tcx>) -> Clauses<'tcx> {
    let ty = generic_types::bound(tcx, 0);
    let slice_ty = tcx.mk_slice(ty);

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };
    let sized_implemented = ty::TraitRef {
        def_id: sized_trait,
        substs: tcx.mk_substs_trait(ty, ty::List::empty()),
    };
    let sized_implemented: DomainGoal<'_> = ty::TraitPredicate {
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

fn wf_clause_for_array<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    length: &'tcx ty::LazyConst<'tcx>
) -> Clauses<'tcx> {
    let ty = generic_types::bound(tcx, 0);
    let array_ty = tcx.mk_ty(ty::Array(ty, length));

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };
    let sized_implemented = ty::TraitRef {
        def_id: sized_trait,
        substs: tcx.mk_substs_trait(ty, ty::List::empty()),
    };
    let sized_implemented: DomainGoal<'_> = ty::TraitPredicate {
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

fn wf_clause_for_tuple<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    arity: usize
) -> Clauses<'tcx> {
    let type_list = generic_types::type_list(tcx, arity);
    let tuple_ty = tcx.mk_ty(ty::Tuple(type_list));

    let sized_trait = match tcx.lang_items().sized_trait() {
        Some(def_id) => def_id,
        None => return ty::List::empty(),
    };

    // If `arity == 0` (i.e. the unit type) or `arity == 1`, this list of
    // hypotheses is actually empty.
    let sized_implemented = type_list[0 .. std::cmp::max(arity, 1) - 1].iter()
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

fn wf_clause_for_ref<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    mutbl: hir::Mutability
) -> Clauses<'tcx> {
    let region = tcx.mk_region(
        ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(0))
    );
    let ty = generic_types::bound(tcx, 1);
    let ref_ty = tcx.mk_ref(region, ty::TypeAndMut {
        ty,
        mutbl,
    });

    let _outlives: DomainGoal<'_> = ty::OutlivesPredicate(ty, region).lower();
    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(ref_ty)),
        hypotheses: ty::List::empty(),

        // FIXME: restore this later once we get better at handling regions
        // hypotheses: tcx.mk_goals(
        //     iter::once(tcx.mk_goal(outlives.into_goal()))
        // ),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<'a, T> { WellFormed(&'a T) :- Outlives(T: 'a). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

fn wf_clause_for_fn_def<'tcx>(
    tcx: ty::TyCtxt<'_, '_, 'tcx>,
    def_id: DefId
) -> Clauses<'tcx> {
    let fn_def = generic_types::fn_def(tcx, def_id);

    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(fn_def)),
        hypotheses: ty::List::empty(),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall <T1, ..., Tn+1> { WellFormed(fn some_fn(T1, ..., Tn) -> Tn+1). }`
    // where `def_id` maps to the `some_fn` function definition
    tcx.mk_clauses(iter::once(wf_clause))
}

impl ChalkInferenceContext<'cx, 'gcx, 'tcx> {
    pub(super) fn program_clauses_impl(
        &self,
        environment: &Environment<'tcx>,
        goal: &DomainGoal<'tcx>,
    ) -> Vec<Clause<'tcx>> {
        use rustc::traits::WhereClause::*;
        use rustc::infer::canonical::OriginalQueryValues;

        let goal = self.infcx.resolve_type_vars_if_possible(goal);

        debug!("program_clauses(goal = {:?})", goal);

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

                if Some(trait_predicate.def_id()) == self.infcx.tcx.lang_items().sized_trait() {
                    assemble_builtin_sized_impls(
                        self.infcx.tcx,
                        trait_predicate.def_id(),
                        trait_predicate.self_ty(),
                        &mut clauses
                    );
                }

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

                    // These types are always WF.
                    ty::Bool |
                    ty::Char |
                    ty::Int(..) |
                    ty::Uint(..) |
                    ty::Float(..) |
                    ty::Str |
                    ty::Param(..) |
                    ty::Placeholder(..) |
                    ty::Error |
                    ty::Never => {
                        let wf_clause = ProgramClause {
                            goal: DomainGoal::WellFormed(WellFormed::Ty(ty)),
                            hypotheses: ty::List::empty(),
                            category: ProgramClauseCategory::WellFormed,
                        };
                        let wf_clause = Clause::Implies(wf_clause);

                        self.infcx.tcx.mk_clauses(iter::once(wf_clause))
                    }

                    // Always WF (recall that we do not check for parameters to be WF).
                    ty::RawPtr(ptr) => wf_clause_for_raw_ptr(self.infcx.tcx, ptr.mutbl),

                    // Always WF (recall that we do not check for parameters to be WF).
                    ty::FnPtr(fn_ptr) => {
                        let fn_ptr = fn_ptr.skip_binder();
                        wf_clause_for_fn_ptr(
                            self.infcx.tcx,
                            fn_ptr.inputs_and_output.len(),
                            fn_ptr.c_variadic,
                            fn_ptr.unsafety,
                            fn_ptr.abi
                        )
                    }

                    // WF if inner type is `Sized`.
                    ty::Slice(..) => wf_clause_for_slice(self.infcx.tcx),

                    // WF if inner type is `Sized`.
                    ty::Array(_, length) => wf_clause_for_array(self.infcx.tcx, length),

                    // WF if all types but the last one are `Sized`.
                    ty::Tuple(types) => wf_clause_for_tuple(
                        self.infcx.tcx,
                        types.len()
                    ),

                    // WF if `sub_ty` outlives `region`.
                    ty::Ref(_, _, mutbl) => wf_clause_for_ref(self.infcx.tcx, mutbl),

                    ty::FnDef(def_id, ..) => wf_clause_for_fn_def(self.infcx.tcx, def_id),

                    ty::Dynamic(..) => {
                        // FIXME: no rules yet for trait objects
                        ty::List::empty()
                    }

                    ty::Adt(def, ..) => {
                        self.infcx.tcx.program_clauses_for(def.did)
                    }

                    // FIXME: these are probably wrong
                    ty::Foreign(def_id) |
                    ty::Closure(def_id, ..) |
                    ty::Generator(def_id, ..) |
                    ty::Opaque(def_id, ..) => {
                        self.infcx.tcx.program_clauses_for(def_id)
                    }

                    // Artificially trigger an ambiguity.
                    ty::Infer(..) => {
                        let tcx = self.infcx.tcx;
                        let types = [tcx.types.i32, tcx.types.u32, tcx.types.f32, tcx.types.f64];
                        let clauses = types.iter()
                            .cloned()
                            .map(|ty| ProgramClause {
                                goal: DomainGoal::WellFormed(WellFormed::Ty(ty)),
                                hypotheses: ty::List::empty(),
                                category: ProgramClauseCategory::WellFormed,
                            })
                            .map(|clause| Clause::Implies(clause));
                        tcx.mk_clauses(clauses)
                    }

                    ty::GeneratorWitness(..) |
                    ty::UnnormalizedProjection(..) |
                    ty::Bound(..) => {
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

        debug!("program_clauses: clauses = {:?}", clauses);
        debug!("program_clauses: adding clauses from environment = {:?}", environment);

        let mut _orig_query_values = OriginalQueryValues::default();
        let canonical_environment = self.infcx.canonicalize_query(
            environment,
            &mut _orig_query_values
        ).value;
        let env_clauses = self.infcx.tcx.program_clauses_for_env(canonical_environment);

        debug!("program_clauses: env_clauses = {:?}", env_clauses);

        clauses.extend(env_clauses.into_iter().cloned());
        clauses.extend(environment.clauses.iter().cloned());
        clauses
    }
}
