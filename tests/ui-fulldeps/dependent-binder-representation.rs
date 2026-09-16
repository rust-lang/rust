//@ edition: 2021
//@ run-pass
// ignore-tidy-linelength
//@ run-flags: --sysroot {{sysroot-base}} {{src-base}}/auxiliary/dependent-binder-representation-input.rs
//@ ignore-cross-compile
//@ ignore-remote
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;
extern crate rustc_type_ir;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;

use rustc_driver::{Callbacks, Compilation};
use rustc_hir::def_id::DefId;
use rustc_interface::interface::Compiler;
use rustc_middle::traits::solve::{
    CandidateEvidence, CandidateEvidenceSource, TraitEvidence, TraitEvidenceKind,
};
use rustc_middle::ty::{
    self, FallibleTypeFolder, TyCtxt, TypeFoldable, TypeVisitable, TypeVisitableExt, Upcast,
};
use rustc_type_ir::relate::{self, Relate, RelateResult, TypeRelation, VarianceDiagInfo};

fn main() -> ExitCode {
    rustc_driver::catch_with_exit_code(|| {
        rustc_driver::run_compiler(&std::env::args().collect::<Vec<_>>(), &mut Check);
    })
}

struct Check;

impl Callbacks for Check {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.sess.dcx().abort_if_errors();
        let family = find_item(tcx, "Family");
        let item = associated_item(tcx, family);
        let other_item = associated_item(tcx, find_item(tcx, "Other"));
        let seven = ty::Const::from_target_usize(tcx, 7);
        let trait_ref = family_ref(tcx, family, tcx.types.unit, seven);
        let impl_def_id = tcx
            .all_impls(family)
            .find(|&impl_def_id| {
                tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip() == tcx.types.unit
            })
            .unwrap();
        let recipe = CandidateEvidence::new(
            trait_ref,
            CandidateEvidenceSource::Impl { impl_def_id, args: tcx.mk_args(&[seven.into()]) },
            [],
        );
        let evidence = tcx.mk_trait_evidence(recipe.clone());

        test_typed_const(tcx);
        test_telescope_projection(tcx, item, evidence);
        test_supertrait_telescope(tcx, item, evidence);
        test_nested_evidence(tcx, trait_ref);
        test_dependent_nested_evidence(tcx, evidence);
        test_contract_substitution(tcx, item, evidence);
        test_malformed_telescopes(tcx, trait_ref);
        test_unused_telescope(tcx, trait_ref);
        test_contract_flags(tcx, trait_ref);
        test_outlives_components(tcx, item, trait_ref);
        test_const_evidence_relation(tcx);
        test_projection_own_variance(tcx);
        test_proof_relation(tcx, recipe.clone());
        test_constructor_validation(tcx, other_item, recipe, evidence);
        Compilation::Stop
    }
}

fn find_item(tcx: TyCtxt<'_>, name: &str) -> DefId {
    tcx.hir_crate_items(())
        .free_items()
        .map(|item| item.owner_id.to_def_id())
        .find(|&def_id| tcx.opt_item_name(def_id).is_some_and(|item| item.as_str() == name))
        .unwrap()
}

fn associated_item(tcx: TyCtxt<'_>, trait_id: DefId) -> DefId {
    tcx.associated_items(trait_id).in_definition_order().next().unwrap().def_id
}

fn bound_ty(tcx: TyCtxt<'_>, depth: u32, index: u32) -> ty::Ty<'_> {
    ty::Ty::new_bound(
        tcx,
        ty::DebruijnIndex::from_u32(depth),
        ty::BoundTy { var: ty::BoundVar::from_u32(index), kind: ty::BoundTyKind::Anon },
    )
}

fn bound_const(tcx: TyCtxt<'_>, depth: u32, index: u32) -> ty::Const<'_> {
    ty::Const::new_bound(
        tcx,
        ty::DebruijnIndex::from_u32(depth),
        ty::BoundConst::new(ty::BoundVar::from_u32(index)),
    )
}

fn family_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    family: DefId,
    self_ty: ty::Ty<'tcx>,
    n: ty::Const<'tcx>,
) -> ty::TraitRef<'tcx> {
    ty::TraitRef::new_from_args(tcx, family, tcx.mk_args(&[self_ty.into(), n.into()]))
}

fn trait_clause(trait_ref: ty::TraitRef<'_>) -> ty::ClauseKind<'_> {
    ty::ClauseKind::Trait(ty::TraitClause { trait_ref, polarity: ty::ClausePolarity::Positive })
}

fn evidence_entry(trait_ref: ty::TraitRef<'_>) -> ty::BoundVariableKind<'_> {
    ty::BoundVariableKind::Evidence(rustc_type_ir::EvidenceVariable::principal(trait_clause(
        trait_ref,
    )))
}

fn bound_evidence<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    depth: u32,
    index: u32,
) -> TraitEvidence<'tcx> {
    tcx.mk_trait_evidence_kind(
        trait_ref,
        TraitEvidenceKind::Bound(
            ty::BoundVarIndexKind::Bound(ty::DebruijnIndex::from_u32(depth)),
            ty::BoundEvidence::new(ty::BoundVar::from_u32(index)),
        ),
    )
}

fn test_typed_const(tcx: TyCtxt<'_>) {
    let t = bound_ty(tcx, 0, 0);
    let c = bound_const(tcx, 0, 1);
    let vars = tcx.mk_bound_variable_kinds(&[
        ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon),
        ty::BoundVariableKind::Const(Some(t)),
    ]);
    let value = ty::Const::from_bool(tcx, true);
    let args = tcx.mk_args(&[tcx.types.bool.into(), value.into()]);
    let binder = ty::Binder::bind_with_vars(c, vars);
    let (result, clauses) = binder.instantiate_with_args_and_telescope_clauses(tcx, args);
    assert_eq!(result, value);
    assert_eq!(clauses.len(), 1);
    assert_eq!(clauses[0].index, 1);
    assert_eq!(
        clauses[0].clause.kind(),
        ty::Binder::dummy(ty::ClauseKind::ConstArgHasType(value, tcx.types.bool)),
    );
    assert_eq!(
        clauses[0].identity.kind(),
        ty::Binder::bind_with_vars(ty::ClauseKind::ConstArgHasType(c, t), vars),
    );
    assert_eq!(clauses[0].instantiation, args);
    assert!(clauses[0].required_contract.is_none());

    // A type used only by an inner declaration still belongs to the outer binder.
    let inner = ty::Binder::bind_with_vars(
        bound_const(tcx, 0, 0),
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Const(Some(bound_ty(tcx, 1, 0)))]),
    );
    let outer = ty::Binder::bind_with_vars(
        inner,
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
    );
    let result = outer.instantiate_with_args(tcx, tcx.mk_args(&[tcx.types.bool.into()]));
    assert_eq!(result.bound_vars()[0].const_ty(), Some(tcx.types.bool));
    assert_eq!(result.skip_binder(), bound_const(tcx, 0, 0));
}

fn test_telescope_projection<'tcx>(tcx: TyCtxt<'tcx>, item: DefId, evidence: TraitEvidence<'tcx>) {
    let t = bound_ty(tcx, 0, 0);
    let c = bound_const(tcx, 0, 1);
    let trait_ref = family_ref(tcx, evidence.trait_ref.def_id, t, c);
    let vars = tcx.mk_bound_variable_kinds(&[
        ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon),
        ty::BoundVariableKind::Const(Some(tcx.types.usize)),
        evidence_entry(trait_ref),
    ]);
    let projection = tcx.mk_evidence_projection(ty::EvidenceProjectionData {
        item_def_id: item,
        evidence: bound_evidence(tcx, trait_ref, 0, 2),
    });
    let alias =
        ty::AliasTy::new_from_args(tcx, ty::EvidenceProjection { projection }, tcx.mk_args(&[]));
    let binder = ty::Binder::bind_with_vars(alias, vars);
    let args = evidence.trait_ref.args;
    let (alias, clauses) = binder.instantiate_with_args_and_evidence_and_telescope_clauses(
        tcx,
        args,
        tcx.mk_trait_evidences(&[evidence]),
    );

    // The evidence slot is outside GenericArgs; the projection recovers Self and N from it.
    assert_eq!(binder.ordinary_bound_var_count(), 2);
    assert!(alias.args.is_empty());
    assert_eq!(alias.full_args(tcx), args);
    let walked: Vec<_> = ty::Ty::new_alias(tcx, ty::IsRigid::Yes, alias).walk().collect();
    assert!(walked.contains(&args.type_at(0).into()));
    assert!(walked.contains(&args.const_at(1).into()));
    assert_eq!(alias.trait_ref(tcx), evidence.trait_ref);
    let ty::EvidenceProjection { projection } = alias.kind else { panic!() };
    assert_eq!(projection.evidence, evidence);
    assert_eq!(projection.item_def_id, item);
    assert_eq!(tcx.trait_of_assoc(item), Some(evidence.trait_ref.def_id));
    assert_eq!(clauses.iter().map(|clause| clause.index).collect::<Vec<_>>(), [1, 2]);
    let expected = [
        ty::ClauseKind::ConstArgHasType(args.const_at(1), tcx.types.usize),
        trait_clause(evidence.trait_ref),
    ];
    let identities = [ty::ClauseKind::ConstArgHasType(c, tcx.types.usize), trait_clause(trait_ref)];
    for ((clause, expected), identity) in clauses.iter().zip(expected).zip(identities) {
        assert_eq!(clause.clause.kind(), ty::Binder::dummy(expected));
        assert_eq!(clause.identity.kind(), ty::Binder::bind_with_vars(identity, vars));
        assert_eq!(clause.instantiation, args);
        assert!(clause.required_contract.is_none());
    }
}

fn test_supertrait_telescope<'tcx>(tcx: TyCtxt<'tcx>, item: DefId, evidence: TraitEvidence<'tcx>) {
    let family = evidence.trait_ref.def_id;
    let n = evidence.trait_ref.args.const_at(1);
    let identity_args = ty::GenericArgs::identity_for_item(tcx, family);
    let ordinary = ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon);
    let project = |evidence: TraitEvidence<'tcx>| {
        let projection =
            tcx.mk_evidence_projection(ty::EvidenceProjectionData { item_def_id: item, evidence });
        ty::AliasTy::new_from_args(tcx, ty::EvidenceProjection { projection }, tcx.mk_args(&[]))
    };
    let projection_ty = |evidence| ty::Ty::new_alias(tcx, ty::IsRigid::No, project(evidence));

    let pred_ref = family_ref(tcx, family, bound_ty(tcx, 0, 0), identity_args.const_at(1));
    let pred_vars = tcx.mk_bound_variable_kinds(&[ordinary, evidence_entry(pred_ref)]);
    let pred: ty::Clause<'tcx> = ty::Binder::bind_with_vars(
        ty::ClauseKind::Projection(ty::ProjectionClause {
            projection_term: project(bound_evidence(tcx, pred_ref, 0, 1)).into(),
            term: identity_args.type_at(0).into(),
        }),
        pred_vars,
    )
    .upcast(tcx);

    let bool_ref = family_ref(tcx, family, tcx.types.bool, n);
    let bool_impl = tcx
        .all_impls(family)
        .find(|&impl_def_id| {
            tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip() == tcx.types.bool
        })
        .unwrap();
    let bool_evidence = tcx.mk_trait_evidence(CandidateEvidence::new(
        bool_ref,
        CandidateEvidenceSource::Impl { impl_def_id: bool_impl, args: tcx.mk_args(&[n.into()]) },
        [],
    ));
    let trait_evidence_ref = family_ref(tcx, family, bound_ty(tcx, 0, 0), n);
    let merged_pred_ref = family_ref(tcx, family, bound_ty(tcx, 0, 1), n);

    // Exercise both the ordinary supertrait path and the merge of two evidence suffixes.
    for trait_has_evidence in [false, true] {
        let mut trait_vars = vec![ordinary];
        let trait_self = if trait_has_evidence {
            trait_vars.push(evidence_entry(trait_evidence_ref));
            projection_ty(bound_evidence(tcx, trait_evidence_ref, 0, 1))
        } else {
            bound_ty(tcx, 0, 0)
        };
        let trait_ref = ty::Binder::bind_with_vars(
            family_ref(tcx, family, trait_self, n),
            tcx.mk_bound_variable_kinds(&trait_vars),
        );
        let merged = pred.instantiate_supertrait(tcx, trait_ref).as_projection_clause().unwrap();

        // [T, TE] and [P, PE] become [T, P, TE, PE], not [T, TE, P, PE].
        // The predicate's N is an early parameter and must also be substituted in PE.
        let mut expected_vars = vec![ordinary, ordinary];
        let mut evidence_args = vec![];
        let mut expected_clauses = vec![];
        let (merged_term, instantiated_term) = if trait_has_evidence {
            expected_vars.push(evidence_entry(trait_evidence_ref));
            evidence_args.push(evidence);
            expected_clauses.push(trait_clause(evidence.trait_ref));
            (projection_ty(bound_evidence(tcx, trait_evidence_ref, 0, 2)), projection_ty(evidence))
        } else {
            (bound_ty(tcx, 0, 0), tcx.types.unit)
        };
        let pred_evidence_index = if trait_has_evidence { 3 } else { 2 };
        expected_vars.push(evidence_entry(merged_pred_ref));
        evidence_args.push(bool_evidence);
        expected_clauses.push(trait_clause(bool_ref));
        assert_eq!(merged.bound_vars(), tcx.mk_bound_variable_kinds(&expected_vars));
        assert_eq!(merged.ordinary_bound_var_count(), 2);
        assert_eq!(
            merged.skip_binder(),
            ty::ProjectionClause {
                projection_term: project(bound_evidence(
                    tcx,
                    merged_pred_ref,
                    0,
                    pred_evidence_index,
                ))
                .into(),
                term: merged_term.into(),
            },
        );

        let args = tcx.mk_args(&[tcx.types.unit.into(), tcx.types.bool.into()]);
        let (instantiated, clauses) = merged
            .instantiate_with_args_and_evidence_and_telescope_clauses(
                tcx,
                args,
                tcx.mk_trait_evidences(&evidence_args),
            );
        assert_eq!(
            instantiated,
            ty::ProjectionClause {
                projection_term: project(bool_evidence).into(),
                term: instantiated_term.into(),
            },
        );
        assert_eq!(clauses.len(), expected_clauses.len());
        for (index, (clause, expected)) in clauses.iter().zip(expected_clauses).enumerate() {
            assert_eq!(clause.index, index as u32 + 2);
            assert_eq!(clause.clause.kind(), ty::Binder::dummy(expected));
            assert_eq!(clause.instantiation, args);
        }
    }
}

fn test_nested_evidence<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>) {
    let vars = tcx.mk_bound_variable_kinds(&[evidence_entry(trait_ref)]);
    let at = |depth| bound_evidence(tcx, trait_ref, depth, 0);
    let inner = ty::Binder::bind_with_vars((at(0), at(1), at(2)), vars);
    let shifted = ty::shift_vars(tcx, inner, 1);
    assert_eq!(shifted.bound_vars(), vars);
    assert_eq!(shifted.skip_binder(), (at(0), at(2), at(3)));

    // Removing the outer binder shifts the replacement beneath the surviving inner binder.
    let outer = ty::Binder::bind_with_vars(inner, vars);
    let result = outer.instantiate_with_args_and_evidence(
        tcx,
        tcx.mk_args(&[]),
        tcx.mk_trait_evidences(&[at(1)]),
    );
    assert_eq!(result.bound_vars(), vars);
    assert_eq!(result.skip_binder(), (at(0), at(2), at(1)));
    let result = result.instantiate_with_args_and_evidence(
        tcx,
        tcx.mk_args(&[]),
        tcx.mk_trait_evidences(&[at(0)]),
    );
    assert_eq!(result, (at(0), at(1), at(0)));
}

fn test_dependent_nested_evidence<'tcx>(tcx: TyCtxt<'tcx>, evidence: TraitEvidence<'tcx>) {
    let trait_ref = family_ref(
        tcx,
        evidence.trait_ref.def_id,
        bound_ty(tcx, 0, 0),
        evidence.trait_ref.args.const_at(1),
    );
    let vars = tcx.mk_bound_variable_kinds(&[
        ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon),
        evidence_entry(trait_ref),
    ]);
    let inner = ty::Binder::bind_with_vars(
        bound_evidence(tcx, ty::shift_vars(tcx, trait_ref, 1), 1, 1),
        tcx.mk_bound_variable_kinds(&[]),
    );
    let _ = inner.visit_with(&mut ty::ValidateBoundVars::new(vars));
    let outer = ty::Binder::bind_with_vars(inner, vars);
    let result = outer.instantiate_with_args_and_evidence(
        tcx,
        tcx.mk_args(&[tcx.types.unit.into()]),
        tcx.mk_trait_evidences(&[evidence]),
    );
    assert_eq!(result.skip_binder(), evidence);

    let wrong_ref =
        family_ref(tcx, trait_ref.def_id, tcx.types.bool, evidence.trait_ref.args.const_at(1));
    assert_rejected(|| {
        let wrong_inner = ty::Binder::bind_with_vars(
            bound_evidence(tcx, wrong_ref, 1, 1),
            tcx.mk_bound_variable_kinds(&[]),
        );
        ty::Binder::bind_with_vars(wrong_inner, vars).instantiate_with_args_and_evidence(
            tcx,
            tcx.mk_args(&[tcx.types.unit.into()]),
            tcx.mk_trait_evidences(&[evidence]),
        );
    });
}

fn test_contract_substitution<'tcx>(tcx: TyCtxt<'tcx>, item: DefId, evidence: TraitEvidence<'tcx>) {
    let ordinary_args = tcx.mk_args(&[tcx.types.unit.into()]);
    let trait_ref = family_ref(
        tcx,
        evidence.trait_ref.def_id,
        bound_ty(tcx, 0, 0),
        evidence.trait_ref.args.const_at(1),
    );
    let principal: ty::Clause<'tcx> = ty::Binder::bind_with_vars(
        trait_clause(ty::shift_vars(tcx, trait_ref, 1)),
        tcx.mk_bound_variable_kinds(&[]),
    )
    .upcast(tcx);
    let equality: ty::Clause<'tcx> = ty::Binder::bind_with_vars(
        ty::ClauseKind::Projection(ty::ProjectionClause {
            projection_term: ty::AliasTy::new_from_args(
                tcx,
                ty::Projection { def_id: item },
                ty::shift_vars(tcx, trait_ref.args, 1),
            )
            .into(),
            term: tcx.types.bool.into(),
        }),
        tcx.mk_bound_variable_kinds(&[]),
    )
    .upcast(tcx);
    let data = ty::BoundRequiredContractData {
        identity: ty::solve::InstantiatedItemContract {
            key: ty::solve::ItemContractKey { owner: trait_ref.def_id, hir_local_id: 1 },
            complete_early_args: tcx.mk_args(&[bound_ty(tcx, 0, 0).into()]),
        },
        clauses: tcx.mk_clauses(&[principal, equality]),
        principal_index: 0,
        ordinary_args: None,
    };
    let make_binder = |data| {
        ty::Binder::bind_with_vars(
            bound_evidence(tcx, trait_ref, 0, 1),
            tcx.mk_bound_variable_kinds(&[
                ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon),
                ty::BoundVariableKind::Evidence(ty::EvidenceVariable {
                    clause: trait_clause(trait_ref),
                    required_contract: Some(tcx.mk_bound_required_contract(data)),
                }),
            ]),
        )
    };
    let binder = make_binder(data);
    let evidence_args = tcx.mk_trait_evidences(&[evidence]);
    assert_rejected(|| {
        binder.instantiate_with_args_and_evidence(tcx, ordinary_args, evidence_args);
    });
    let (result, clauses) = binder.instantiate_with_args_and_evidence_and_telescope_clauses(
        tcx,
        ordinary_args,
        evidence_args,
    );
    assert_eq!(result, evidence);
    assert_eq!(clauses.len(), 1);
    let contract = clauses[0].required_contract.unwrap();
    assert_eq!(contract.principal_clause(), clauses[0].clause);
    assert_eq!(contract.clauses.len(), 2);
    assert_eq!(contract.identity.complete_early_args, ordinary_args);
    assert_eq!(contract.ordinary_args, Some(ordinary_args));
    let equality = contract.clauses[1].as_projection_clause().unwrap().no_bound_vars().unwrap();
    assert_eq!(equality.projection_term.full_args(tcx), evidence.trait_ref.args);
    assert_eq!(equality.term, tcx.types.bool.into());

    for malformed in [
        ty::BoundRequiredContractData { clauses: tcx.mk_clauses(&[]), ..data },
        ty::BoundRequiredContractData { principal_index: 2, ..data },
        ty::BoundRequiredContractData { principal_index: 1, ..data },
    ] {
        assert_rejected(|| {
            tcx.mk_bound_required_contract(malformed);
        });
    }
    let wrong_principal: ty::Clause<'tcx> =
        ty::Binder::dummy(trait_clause(evidence.trait_ref)).upcast(tcx);
    assert_rejected(|| {
        make_binder(ty::BoundRequiredContractData {
            clauses: tcx.mk_clauses(&[wrong_principal]),
            ..data
        })
        .instantiate_with_args_and_evidence_and_telescope_clauses(
            tcx,
            ordinary_args,
            evidence_args,
        );
    });
    let contract = tcx.mk_bound_required_contract(data);
    assert_eq!(contract.try_fold_with(&mut RejectBool(tcx)), Err(()));
}

struct RejectBool<'tcx>(TyCtxt<'tcx>);

impl<'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for RejectBool<'tcx> {
    type Error = ();

    fn cx(&self) -> TyCtxt<'tcx> {
        self.0
    }

    fn try_fold_ty(&mut self, ty: ty::Ty<'tcx>) -> Result<ty::Ty<'tcx>, ()> {
        if ty.is_bool() { Err(()) } else { Ok(ty) }
    }
}

fn test_malformed_telescopes<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>) {
    for (entries, args) in [
        (
            vec![evidence_entry(trait_ref), ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)],
            tcx.mk_args(&[]),
        ),
        (
            vec![
                ty::BoundVariableKind::Const(Some(bound_ty(tcx, 0, 1))),
                ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon),
            ],
            tcx.mk_args(&[ty::Const::from_bool(tcx, true).into(), tcx.types.bool.into()]),
        ),
    ] {
        assert_rejected(|| {
            ty::Binder::bind_with_vars(tcx.types.unit, tcx.mk_bound_variable_kinds(&entries))
                .instantiate_with_args_and_telescope_clauses(tcx, args);
        });
    }
}

fn test_proof_relation<'tcx>(tcx: TyCtxt<'tcx>, recipe: CandidateEvidence<TyCtxt<'tcx>>) {
    let trait_ref = recipe.root_node().trait_ref;
    let original = tcx.mk_trait_evidence(recipe.clone());
    let mut shortened = recipe.clone();
    let CandidateEvidenceSource::Impl { impl_def_id, args } = shortened.root_source() else {
        panic!()
    };
    assert!(!args.is_empty());
    shortened.nodes[0].source = CandidateEvidenceSource::Impl {
        impl_def_id, args: tcx.mk_args(&[]),
    };
    assert!(EraseRegions::new(tcx).evidences(tcx.mk_trait_evidence(shortened), original).is_err());
    let key = ty::solve::CoherenceKey { trait_ref };
    let different = CandidateEvidence::new(
        trait_ref,
        CandidateEvidenceSource::ParamEnv {
            source: ty::solve::ParamEnvSource::NonGlobal,
            origin: ty::solve::ParamEnvAssumption::CallerBound { index: 0 },
        },
        [],
    );
    let a = tcx.mk_trait_evidence(recipe.into_unique(key));
    let b = tcx.mk_trait_evidence(different.clone().into_unique(key));
    assert!(EraseRegions::new(tcx).evidences(a, b).is_err());
    assert!(EraseRegions::new(tcx).evidences(b, tcx.mk_trait_evidence(different)).is_err());
    let object_bound: ty::Clause<'tcx> = ty::Binder::bind_with_vars(
        trait_clause(trait_ref),
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon)]),
    )
    .upcast(tcx);
    let dyn_evidence = |instantiation| {
        tcx.mk_trait_evidence(CandidateEvidence::new(
            trait_ref,
            CandidateEvidenceSource::Dyn {
                object_bound,
                instantiation,
                vtable_slot: None,
                operation: None,
            },
            [],
        ))
    };
    let uninstantiated = dyn_evidence(None);
    let instantiated = dyn_evidence(Some(tcx.mk_args(&[tcx.lifetimes.re_static.into()])));
    assert!(EraseRegions::new(tcx).evidences(uninstantiated, instantiated).is_err());
}

fn test_unused_telescope<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>) {
    for entry in [ty::BoundVariableKind::Const(Some(tcx.types.bool)), evidence_entry(trait_ref)] {
        let binder =
            ty::Binder::bind_with_vars(tcx.types.unit, tcx.mk_bound_variable_kinds(&[entry]));
        assert!(binder.no_bound_vars().is_none());
        assert_rejected(|| {
            tcx.instantiate_bound_regions_with_erased(binder);
        });
        assert_rejected(|| {
            tcx.replace_bound_vars_uncached(
                binder,
                ty::FnMutDelegate {
                    regions: &mut |_| unreachable!(),
                    types: &mut |_| unreachable!(),
                    consts: &mut |_| unreachable!(),
                },
            );
        });
    }
    let plain = ty::Binder::bind_with_vars(
        tcx.types.unit,
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Const(None)]),
    );
    assert_eq!(plain.no_bound_vars(), Some(tcx.types.unit));
}

fn test_contract_flags<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>) {
    let clause: ty::Clause<'tcx> = ty::Binder::dummy(trait_clause(trait_ref)).upcast(tcx);
    let contract = tcx.mk_bound_required_contract(ty::BoundRequiredContractData {
        identity: ty::solve::InstantiatedItemContract {
            key: ty::solve::ItemContractKey { owner: trait_ref.def_id, hir_local_id: 0 },
            complete_early_args: tcx.mk_args(&[]),
        },
        clauses: tcx.mk_clauses(&[clause]),
        principal_index: 0,
        ordinary_args: Some(tcx.mk_args(&[bound_ty(tcx, 1, 0).into()])),
    });
    let inner = ty::Binder::bind_with_vars(
        tcx.types.unit,
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Evidence(ty::EvidenceVariable {
            clause: trait_clause(trait_ref),
            required_contract: Some(contract),
        })]),
    );
    // The only outer reference is in contract metadata, behind an interned type's flags.
    let inner_ty = ty::Ty::new_unsafe_binder(tcx, inner);
    assert!(inner_ty.has_escaping_bound_vars());
    let outer = ty::Binder::bind_with_vars(
        inner_ty,
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
    );
    let result = outer.instantiate_with_args(tcx, tcx.mk_args(&[tcx.types.bool.into()]));
    assert!(!result.has_escaping_bound_vars());
    let ty::UnsafeBinder(inner) = *result.kind() else { panic!() };
    let ty::BoundVariableKind::Evidence(entry) = inner.bound_vars()[0] else { panic!() };
    assert_eq!(entry.required_contract.unwrap().ordinary_args.unwrap().type_at(0), tcx.types.bool);
}

fn test_outlives_components<'tcx>(tcx: TyCtxt<'tcx>, item: DefId, trait_ref: ty::TraitRef<'tcx>) {
    use rustc_type_ir::outlives::{Component, push_outlives_components};

    let projection = tcx.mk_evidence_projection(ty::EvidenceProjectionData {
        item_def_id: item,
        evidence: bound_evidence(tcx, trait_ref, 0, 0),
    });
    let alias =
        ty::AliasTy::new_from_args(tcx, ty::EvidenceProjection { projection }, tcx.mk_args(&[]));
    assert!(alias.has_escaping_bound_vars());
    assert!(!alias.full_args(tcx).has_escaping_bound_vars());
    let projected_ty = ty::Ty::new_alias(tcx, ty::IsRigid::Yes, alias);
    let mut components = Default::default();
    push_outlives_components(tcx, projected_ty, &mut components);
    assert!(matches!(components.as_slice(), [Component::EscapingAlias(_)]));
}

fn test_const_evidence_relation<'tcx>(tcx: TyCtxt<'tcx>) {
    let borrowing = find_item(tcx, "Borrowing");
    let item = associated_item(tcx, borrowing);
    let impl_def_id = tcx.all_impls(borrowing).next().unwrap();
    let alias = |region: ty::Region<'tcx>| {
        let trait_ref = ty::TraitRef::new_from_args(
            tcx,
            borrowing,
            tcx.mk_args(&[tcx.types.unit.into(), region.into()]),
        );
        let evidence = tcx.mk_trait_evidence(CandidateEvidence::new(
            trait_ref,
            CandidateEvidenceSource::Impl { impl_def_id, args: tcx.mk_args(&[region.into()]) },
            [],
        ));
        let projection =
            tcx.mk_evidence_projection(ty::EvidenceProjectionData { item_def_id: item, evidence });
        ty::AliasConst::new(
            tcx,
            ty::AliasConstKind::EvidenceProjection { projection },
            tcx.mk_args(&[]),
        )
    };
    let a = alias(tcx.lifetimes.re_static);
    let b = alias(tcx.lifetimes.re_erased);
    assert_ne!(a, b);
    // Distinct interned proofs can relate when their ordinary regions relate.
    assert_eq!(EraseRegions::new(tcx).relate(a, b).unwrap(), b);
    let const_ty = ty::Const::new_alias(tcx, ty::IsRigid::Yes, a);
    let walked: Vec<_> = const_ty.walk().collect();
    assert!(walked.contains(&tcx.types.unit.into()));
    assert!(walked.contains(&tcx.lifetimes.re_static.into()));
    assert!(matches!(
        tcx.const_eval_resolve_for_typeck(
            ty::TypingEnv::fully_monomorphized(),
            a,
            rustc_span::DUMMY_SP,
        ),
        Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_))
    ));
    let ty::AliasConstKind::EvidenceProjection { projection } = a.kind else { panic!() };
    let projection = tcx.mk_evidence_projection(ty::EvidenceProjectionData {
        item_def_id: item,
        evidence: bound_evidence(tcx, projection.trait_ref(), 0, 0),
    });
    let unresolved = ty::AliasConst::new(
        tcx,
        ty::AliasConstKind::EvidenceProjection { projection },
        tcx.mk_args(&[]),
    );
    assert!(matches!(
        tcx.const_eval_resolve_for_typeck(
            ty::TypingEnv::fully_monomorphized(),
            unresolved,
            rustc_span::DUMMY_SP,
        ),
        Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_))
    ));
}

fn test_projection_own_variance(tcx: TyCtxt<'_>) {
    let trait_id = find_item(tcx, "ReturnType");
    let method = associated_item(tcx, trait_id);
    let item = tcx.associated_types_for_impl_traits_in_associated_fn(method)[0];
    let trait_ref =
        ty::TraitRef::new_from_args(tcx, trait_id, tcx.mk_args(&[tcx.types.unit.into()]));
    let evidence = tcx.mk_trait_evidence(CandidateEvidence::new(
        trait_ref,
        CandidateEvidenceSource::ParamEnv {
            source: ty::solve::ParamEnvSource::NonGlobal,
            origin: ty::solve::ParamEnvAssumption::CallerBound { index: 0 },
        },
        [],
    ));
    let projection =
        tcx.mk_evidence_projection(ty::EvidenceProjectionData { item_def_id: item, evidence });
    let a = ty::AliasTy::new_from_args(
        tcx,
        ty::EvidenceProjection { projection },
        tcx.mk_args(&[tcx.lifetimes.re_static.into()]),
    );
    let b = ty::AliasTy::new_from_args(
        tcx,
        ty::EvidenceProjection { projection },
        tcx.mk_args(&[tcx.lifetimes.re_erased.into()]),
    );
    let variances = tcx.variances_of(item);
    assert_ne!(variances[0], variances[trait_ref.args.len()]);
    let mut relation = EraseRegions::new(tcx);
    assert_eq!(relation.relate(a, b).unwrap(), b);
    assert_eq!(relation.variances, [variances[trait_ref.args.len()]]);
}

struct EraseRegions<'tcx> {
    tcx: TyCtxt<'tcx>,
    variances: Vec<ty::Variance>,
}

impl<'tcx> EraseRegions<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, variances: Vec::new() }
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for EraseRegions<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn tys(
        &mut self,
        a: ty::Ty<'tcx>,
        b: ty::Ty<'tcx>,
    ) -> RelateResult<TyCtxt<'tcx>, ty::Ty<'tcx>> {
        relate::structurally_relate_tys(self, a, b)
    }

    fn regions(
        &mut self,
        _: ty::Region<'tcx>,
        _: ty::Region<'tcx>,
    ) -> RelateResult<TyCtxt<'tcx>, ty::Region<'tcx>> {
        Ok(self.tcx.lifetimes.re_erased)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<TyCtxt<'tcx>, ty::Const<'tcx>> {
        relate::structurally_relate_consts(self, a, b)
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _: VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<TyCtxt<'tcx>, T> {
        self.variances.push(variance);
        self.relate(a, b)
    }

    fn relate_ty_args(
        &mut self,
        _: ty::Ty<'tcx>,
        _: ty::Ty<'tcx>,
        _: DefId,
        a: ty::GenericArgsRef<'tcx>,
        b: ty::GenericArgsRef<'tcx>,
        mk: impl FnOnce(ty::GenericArgsRef<'tcx>) -> ty::Ty<'tcx>,
    ) -> RelateResult<TyCtxt<'tcx>, ty::Ty<'tcx>> {
        Ok(mk(relate::relate_args_invariantly(self, a, b)?))
    }

    fn binders<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<TyCtxt<'tcx>, ty::Binder<'tcx, T>> {
        if a == b { Ok(a) } else { Err(ty::error::TypeError::Mismatch) }
    }
}

fn test_constructor_validation<'tcx>(
    tcx: TyCtxt<'tcx>,
    other_item: DefId,
    recipe: CandidateEvidence<TyCtxt<'tcx>>,
    evidence: TraitEvidence<'tcx>,
) {
    let mut out_of_bounds = recipe.clone();
    out_of_bounds.nodes[0].nested.push(1);
    assert_rejected(|| {
        tcx.mk_trait_evidence(out_of_bounds);
    });

    let mut cyclic = recipe;
    cyclic.nodes[0].nested.push(0);
    assert_rejected(|| {
        tcx.mk_trait_evidence(cyclic);
    });

    assert_rejected(|| {
        tcx.mk_evidence_projection(ty::EvidenceProjectionData {
            item_def_id: other_item,
            evidence,
        });
    });
}

fn assert_rejected(f: impl FnOnce()) {
    // These constructors assert internal invariants before interning their input.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    assert!(result.is_err());
}
