mod builtin;
mod primitive;

use rustc::traits::{
    WellFormed,
    FromEnv,
    DomainGoal,
    Clause,
    ProgramClause,
    ProgramClauseCategory,
    Environment,
};
use rustc::ty::{self, TyCtxt};
use rustc::hir::def_id::DefId;
use super::ChalkInferenceContext;
use std::iter;

use self::primitive::*;
use self::builtin::*;

fn assemble_clauses_from_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    clauses: &mut Vec<Clause<'tcx>>,
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
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    clauses: &mut Vec<Clause<'tcx>>,
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

impl ChalkInferenceContext<'cx, 'tcx> {
    pub(super) fn program_clauses_impl(
        &self,
        environment: &Environment<'tcx>,
        goal: &DomainGoal<'tcx>,
    ) -> Vec<Clause<'tcx>> {
        use rustc::traits::WhereClause::*;
        use rustc::infer::canonical::OriginalQueryValues;

        let goal = self.infcx.resolve_vars_if_possible(goal);

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

                if Some(trait_predicate.def_id()) == self.infcx.tcx.lang_items().unsize_trait() {
                    let source = trait_predicate.self_ty();
                    let target = trait_predicate.trait_ref.substs.type_at(1);
                    assemble_builtin_unsize_impls(
                        self.infcx.tcx,
                        trait_predicate.def_id(),
                        source,
                        target,
                        &mut clauses
                    );
                }

                if Some(trait_predicate.def_id()) == self.infcx.tcx.lang_items().copy_trait() {
                    assemble_builtin_copy_clone_impls(
                        self.infcx.tcx,
                        trait_predicate.def_id(),
                        trait_predicate.self_ty(),
                        &mut clauses
                    );
                }

                if Some(trait_predicate.def_id()) == self.infcx.tcx.lang_items().clone_trait() {
                    // For all builtin impls, the conditions for `Copy` and
                    // `Clone` are the same.
                    assemble_builtin_copy_clone_impls(
                        self.infcx.tcx,
                        trait_predicate.def_id(),
                        trait_predicate.self_ty(),
                        &mut clauses
                    );
                }

                // FIXME: we need to add special rules for other builtin impls:
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

            // For outlive requirements, just assume they hold. `ResolventOps::resolvent_clause`
            // will register them as actual region constraints later.
            DomainGoal::Holds(RegionOutlives(..)) | DomainGoal::Holds(TypeOutlives(..)) => {
                vec![Clause::Implies(ProgramClause {
                    goal,
                    hypotheses: ty::List::empty(),
                    category: ProgramClauseCategory::Other,
                })]
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
                            goal,
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
