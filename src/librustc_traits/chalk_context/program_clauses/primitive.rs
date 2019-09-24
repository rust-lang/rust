use rustc::traits::{
    WellFormed,
    DomainGoal,
    GoalKind,
    Clause,
    Clauses,
    ProgramClause,
    ProgramClauseCategory,
};
use rustc::ty::{self, TyCtxt};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc_target::spec::abi;
use crate::lowering::Lower;
use crate::generic_types;
use std::iter;

crate fn wf_clause_for_raw_ptr(tcx: TyCtxt<'_>, mutbl: hir::Mutability) -> Clauses<'_> {
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

crate fn wf_clause_for_fn_ptr(
    tcx: TyCtxt<'_>,
    arity_and_output: usize,
    variadic: bool,
    unsafety: hir::Unsafety,
    abi: abi::Abi,
) -> Clauses<'_> {
    let fn_ptr = generic_types::fn_ptr(tcx, arity_and_output, variadic, unsafety, abi);

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

crate fn wf_clause_for_slice(tcx: TyCtxt<'_>) -> Clauses<'_> {
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

crate fn wf_clause_for_array<'tcx>(
    tcx: TyCtxt<'tcx>,
    length: &'tcx ty::Const<'tcx>,
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

crate fn wf_clause_for_tuple(tcx: TyCtxt<'_>, arity: usize) -> Clauses<'_> {
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
            substs: tcx.mk_substs_trait(ty.expect_ty(), ty::List::empty()),
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

crate fn wf_clause_for_ref(tcx: TyCtxt<'_>, mutbl: hir::Mutability) -> Clauses<'_> {
    let region = tcx.mk_region(
        ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(0))
    );
    let ty = generic_types::bound(tcx, 1);
    let ref_ty = tcx.mk_ref(region, ty::TypeAndMut {
        ty,
        mutbl,
    });

    let outlives: DomainGoal<'_> = ty::OutlivesPredicate(ty, region).lower();
    let wf_clause = ProgramClause {
        goal: DomainGoal::WellFormed(WellFormed::Ty(ref_ty)),
        hypotheses: tcx.mk_goals(
            iter::once(tcx.mk_goal(outlives.into_goal()))
        ),
        category: ProgramClauseCategory::WellFormed,
    };
    let wf_clause = Clause::ForAll(ty::Binder::bind(wf_clause));

    // `forall<'a, T> { WellFormed(&'a T) :- Outlives(T: 'a). }`
    tcx.mk_clauses(iter::once(wf_clause))
}

crate fn wf_clause_for_fn_def(tcx: TyCtxt<'_>, def_id: DefId) -> Clauses<'_> {
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
