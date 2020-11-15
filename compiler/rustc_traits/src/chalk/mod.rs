//! Calls `chalk-solve` to solve a `ty::Predicate`
//!
//! In order to call `chalk-solve`, this file must convert a `CanonicalChalkEnvironmentAndGoal` into
//! a Chalk uncanonical goal. It then calls Chalk, and converts the answer back into rustc solution.

crate mod db;
crate mod lowering;

use rustc_data_structures::fx::FxHashMap;

use rustc_index::vec::IndexVec;

use rustc_middle::infer::canonical::{CanonicalTyVarKind, CanonicalVarKind};
use rustc_middle::traits::ChalkRustInterner;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::GenericArg;
use rustc_middle::ty::{self, BoundVar, ParamTy, TyCtxt, TypeFoldable};

use rustc_infer::infer::canonical::{
    Canonical, CanonicalVarValues, Certainty, QueryRegionConstraints, QueryResponse,
};
use rustc_infer::traits::{self, CanonicalChalkEnvironmentAndGoal};

use crate::chalk::db::RustIrDatabase as ChalkRustIrDatabase;
use crate::chalk::lowering::{
    LowerInto, ParamsSubstitutor, PlaceholdersCollector, RegionsSubstitutor,
};

use chalk_solve::Solution;

crate fn provide(p: &mut Providers) {
    *p = Providers { evaluate_goal, ..*p };
}

crate fn evaluate_goal<'tcx>(
    tcx: TyCtxt<'tcx>,
    obligation: CanonicalChalkEnvironmentAndGoal<'tcx>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, traits::query::NoSolution> {
    let interner = ChalkRustInterner { tcx };

    // Chalk doesn't have a notion of `Params`, so instead we use placeholders.
    let mut placeholders_collector = PlaceholdersCollector::new();
    obligation.visit_with(&mut placeholders_collector);

    let reempty_placeholder = tcx.mk_region(ty::RegionKind::RePlaceholder(ty::Placeholder {
        universe: ty::UniverseIndex::ROOT,
        name: ty::BoundRegion::BrAnon(placeholders_collector.next_anon_region_placeholder + 1),
    }));

    let mut params_substitutor =
        ParamsSubstitutor::new(tcx, placeholders_collector.next_ty_placeholder);
    let obligation = obligation.fold_with(&mut params_substitutor);
    // FIXME(chalk): we really should be substituting these back in the solution
    let _params: FxHashMap<usize, ParamTy> = params_substitutor.params;

    let mut regions_substitutor = RegionsSubstitutor::new(tcx, reempty_placeholder);
    let obligation = obligation.fold_with(&mut regions_substitutor);

    let max_universe = obligation.max_universe.index();

    let lowered_goal: chalk_ir::UCanonical<
        chalk_ir::InEnvironment<chalk_ir::Goal<ChalkRustInterner<'tcx>>>,
    > = chalk_ir::UCanonical {
        canonical: chalk_ir::Canonical {
            binders: chalk_ir::CanonicalVarKinds::from_iter(
                &interner,
                obligation.variables.iter().map(|v| match v.kind {
                    CanonicalVarKind::PlaceholderTy(_ty) => unimplemented!(),
                    CanonicalVarKind::PlaceholderRegion(_ui) => unimplemented!(),
                    CanonicalVarKind::Ty(ty) => match ty {
                        CanonicalTyVarKind::General(ui) => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
                            chalk_ir::UniverseIndex { counter: ui.index() },
                        ),
                        CanonicalTyVarKind::Int => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::Integer),
                            chalk_ir::UniverseIndex::root(),
                        ),
                        CanonicalTyVarKind::Float => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::Float),
                            chalk_ir::UniverseIndex::root(),
                        ),
                    },
                    CanonicalVarKind::Region(ui) => chalk_ir::WithKind::new(
                        chalk_ir::VariableKind::Lifetime,
                        chalk_ir::UniverseIndex { counter: ui.index() },
                    ),
                    CanonicalVarKind::Const(_ui) => unimplemented!(),
                    CanonicalVarKind::PlaceholderConst(_pc) => unimplemented!(),
                }),
            ),
            value: obligation.value.lower_into(&interner),
        },
        universes: max_universe + 1,
    };

    use chalk_solve::Solver;
    let mut solver = chalk_engine::solve::SLGSolver::new(32, None);
    let db = ChalkRustIrDatabase { interner, reempty_placeholder };
    let solution = solver.solve(&db, &lowered_goal);
    debug!(?obligation, ?solution, "evaluatate goal");

    // Ideally, the code to convert *back* to rustc types would live close to
    // the code to convert *from* rustc types. Right now though, we don't
    // really need this and so it's really minimal.
    // Right now, we also treat a `Unique` solution the same as
    // `Ambig(Definite)`. This really isn't right.
    let make_solution = |subst: chalk_ir::Substitution<_>| {
        let mut var_values: IndexVec<BoundVar, GenericArg<'tcx>> = IndexVec::new();
        subst.as_slice(&interner).iter().for_each(|p| {
            var_values.push(p.lower_into(&interner));
        });
        let sol = Canonical {
            max_universe: ty::UniverseIndex::from_usize(0),
            variables: obligation.variables.clone(),
            value: QueryResponse {
                var_values: CanonicalVarValues { var_values },
                region_constraints: QueryRegionConstraints::default(),
                certainty: Certainty::Proven,
                value: (),
            },
        };
        tcx.arena.alloc(sol)
    };
    solution
        .map(|s| match s {
            Solution::Unique(subst) => {
                // FIXME(chalk): handle constraints
                make_solution(subst.value.subst)
            }
            Solution::Ambig(guidance) => {
                match guidance {
                    chalk_solve::Guidance::Definite(subst) => make_solution(subst.value),
                    chalk_solve::Guidance::Suggested(_) => unimplemented!(),
                    chalk_solve::Guidance::Unknown => {
                        // chalk_fulfill doesn't use the var_values here, so
                        // let's just ignore that
                        let sol = Canonical {
                            max_universe: ty::UniverseIndex::from_usize(0),
                            variables: obligation.variables.clone(),
                            value: QueryResponse {
                                var_values: CanonicalVarValues { var_values: IndexVec::new() }
                                    .make_identity(tcx),
                                region_constraints: QueryRegionConstraints::default(),
                                certainty: Certainty::Ambiguous,
                                value: (),
                            },
                        };
                        &*tcx.arena.alloc(sol)
                    }
                }
            }
        })
        .ok_or(traits::query::NoSolution)
}
