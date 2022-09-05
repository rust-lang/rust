#![allow(rustc::potential_query_instability)]
#![feature(box_patterns)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(map_try_insert)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(option_get_or_insert_default)]
#![feature(trusted_step)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
#![feature(if_let_guard)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use required_consts::RequiredConstsVisitor;
use rustc_const_eval::util;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::Visitor as _;
use rustc_middle::mir::{
    traversal, AnalysisPhase, Body, ConstQualifs, Constant, LocalDecl, MirPass, MirPhase, Operand,
    Place, ProjectionElem, Promoted, RuntimePhase, Rvalue, SourceInfo, Statement, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable};
use rustc_span::sym;

#[macro_use]
mod pass_manager;

use pass_manager::{self as pm, Lint, MirLint, WithMinOptLevel};

mod abort_unwinding_calls;
mod add_call_guards;
mod add_moves_for_packed_drops;
mod add_retag;
mod check_const_item_mutation;
mod check_packed_ref;
pub mod check_unsafety;
// This pass is public to allow external drivers to perform MIR cleanup
pub mod cleanup_post_borrowck;
mod const_debuginfo;
mod const_goto;
mod const_prop;
mod const_prop_lint;
mod coverage;
mod dead_store_elimination;
mod deaggregator;
mod deduplicate_blocks;
mod deref_separator;
mod dest_prop;
pub mod dump_mir;
mod early_otherwise_branch;
mod elaborate_box_derefs;
mod elaborate_drops;
mod ffi_unwind_calls;
mod function_item_references;
mod generator;
mod inline;
mod instcombine;
mod lower_intrinsics;
mod lower_slice_len;
mod marker;
mod match_branches;
mod multiple_return_terminators;
mod normalize_array_len;
mod nrvo;
// This pass is public to allow external drivers to perform MIR cleanup
pub mod remove_false_edges;
mod remove_noop_landing_pads;
mod remove_storage_markers;
mod remove_uninit_drops;
mod remove_unneeded_drops;
mod remove_zsts;
mod required_consts;
mod reveal_all;
mod separate_const_switch;
mod shim;
// This pass is public to allow external drivers to perform MIR cleanup
pub mod simplify;
mod simplify_branches;
mod simplify_comparison_integral;
mod simplify_try;
mod uninhabited_enum_branching;
mod unreachable_prop;

use rustc_const_eval::transform::check_consts::{self, ConstCx};
use rustc_const_eval::transform::promote_consts;
use rustc_const_eval::transform::validate;
use rustc_mir_dataflow::rustc_peek;

pub fn provide(providers: &mut Providers) {
    check_unsafety::provide(providers);
    check_packed_ref::provide(providers);
    coverage::query::provide(providers);
    ffi_unwind_calls::provide(providers);
    shim::provide(providers);
    *providers = Providers {
        mir_keys,
        mir_const,
        mir_const_qualif: |tcx, def_id| {
            let def_id = def_id.expect_local();
            if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
                tcx.mir_const_qualif_const_arg(def)
            } else {
                mir_const_qualif(tcx, ty::WithOptConstParam::unknown(def_id))
            }
        },
        mir_const_qualif_const_arg: |tcx, (did, param_did)| {
            mir_const_qualif(tcx, ty::WithOptConstParam { did, const_param_did: Some(param_did) })
        },
        mir_promoted,
        mir_drops_elaborated_and_const_checked,
        mir_for_ctfe,
        mir_for_ctfe_of_const_arg,
        optimized_mir,
        is_mir_available,
        is_ctfe_mir_available: |tcx, did| is_mir_available(tcx, did),
        mir_callgraph_reachable: inline::cycle::mir_callgraph_reachable,
        mir_inliner_callees: inline::cycle::mir_inliner_callees,
        promoted_mir: |tcx, def_id| {
            let def_id = def_id.expect_local();
            if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
                tcx.promoted_mir_of_const_arg(def)
            } else {
                promoted_mir(tcx, ty::WithOptConstParam::unknown(def_id))
            }
        },
        promoted_mir_of_const_arg: |tcx, (did, param_did)| {
            promoted_mir(tcx, ty::WithOptConstParam { did, const_param_did: Some(param_did) })
        },
        ..*providers
    };
}

fn remap_mir_for_const_eval_select<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut body: Body<'tcx>,
    context: hir::Constness,
) -> Body<'tcx> {
    for bb in body.basic_blocks.as_mut().iter_mut() {
        let terminator = bb.terminator.as_mut().expect("invalid terminator");
        match terminator.kind {
            TerminatorKind::Call {
                func: Operand::Constant(box Constant { ref literal, .. }),
                ref mut args,
                destination,
                target,
                cleanup,
                fn_span,
                ..
            } if let ty::FnDef(def_id, _) = *literal.ty().kind()
                && tcx.item_name(def_id) == sym::const_eval_select
                && tcx.is_intrinsic(def_id) =>
            {
                let [tupled_args, called_in_const, called_at_rt]: [_; 3] = std::mem::take(args).try_into().unwrap();
                let ty = tupled_args.ty(&body.local_decls, tcx);
                let fields = ty.tuple_fields();
                let num_args = fields.len();
                let func = if context == hir::Constness::Const { called_in_const } else { called_at_rt };
                let (method, place): (fn(Place<'tcx>) -> Operand<'tcx>, Place<'tcx>) = match tupled_args {
                    Operand::Constant(_) => {
                        // there is no good way of extracting a tuple arg from a constant (const generic stuff)
                        // so we just create a temporary and deconstruct that.
                        let local = body.local_decls.push(LocalDecl::new(ty, fn_span));
                        bb.statements.push(Statement {
                            source_info: SourceInfo::outermost(fn_span),
                            kind: StatementKind::Assign(Box::new((local.into(), Rvalue::Use(tupled_args.clone())))),
                        });
                        (Operand::Move, local.into())
                    }
                    Operand::Move(place) => (Operand::Move, place),
                    Operand::Copy(place) => (Operand::Copy, place),
                };
                let place_elems = place.projection;
                let arguments = (0..num_args).map(|x| {
                    let mut place_elems = place_elems.to_vec();
                    place_elems.push(ProjectionElem::Field(x.into(), fields[x]));
                    let projection = tcx.intern_place_elems(&place_elems);
                    let place = Place {
                        local: place.local,
                        projection,
                    };
                    method(place)
                }).collect();
                terminator.kind = TerminatorKind::Call { func, args: arguments, destination, target, cleanup, from_hir_call: false, fn_span };
            }
            _ => {}
        }
    }
    body
}

fn is_mir_available(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let def_id = def_id.expect_local();
    tcx.mir_keys(()).contains(&def_id)
}

/// Finds the full set of `DefId`s within the current crate that have
/// MIR associated with them.
fn mir_keys(tcx: TyCtxt<'_>, (): ()) -> FxIndexSet<LocalDefId> {
    let mut set = FxIndexSet::default();

    // All body-owners have MIR associated with them.
    set.extend(tcx.hir().body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        set: &'a mut FxIndexSet<LocalDefId>,
    }
    impl<'tcx> Visitor<'tcx> for GatherCtors<'_, 'tcx> {
        fn visit_variant_data(&mut self, v: &'tcx hir::VariantData<'tcx>) {
            if let hir::VariantData::Tuple(_, hir_id) = *v {
                self.set.insert(self.tcx.hir().local_def_id(hir_id));
            }
            intravisit::walk_struct_def(self, v)
        }
    }
    tcx.hir().visit_all_item_likes_in_crate(&mut GatherCtors { tcx, set: &mut set });

    set
}

fn mir_const_qualif(tcx: TyCtxt<'_>, def: ty::WithOptConstParam<LocalDefId>) -> ConstQualifs {
    let const_kind = tcx.hir().body_const_context(def.did);

    // No need to const-check a non-const `fn`.
    if const_kind.is_none() {
        return Default::default();
    }

    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_promoted()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let body = &tcx.mir_const(def).borrow();

    if body.return_ty().references_error() {
        tcx.sess.delay_span_bug(body.span, "mir_const_qualif: MIR had errors");
        return Default::default();
    }

    let ccx = check_consts::ConstCx { body, tcx, const_kind, param_env: tcx.param_env(def.did) };

    let mut validator = check_consts::check::Checker::new(&ccx);
    validator.check_body();

    // We return the qualifs in the return place for every MIR body, even though it is only used
    // when deciding to promote a reference to a `const` for now.
    validator.qualifs_in_return_place()
}

/// Make MIR ready for const evaluation. This is run on all MIR, not just on consts!
/// FIXME(oli-obk): it's unclear whether we still need this phase (and its corresponding query).
/// We used to have this for pre-miri MIR based const eval.
fn mir_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx Steal<Body<'tcx>> {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_const(def);
    }

    // Unsafety check uses the raw mir, so make sure it is run.
    if !tcx.sess.opts.unstable_opts.thir_unsafeck {
        if let Some(param_did) = def.const_param_did {
            tcx.ensure().unsafety_check_result_for_const_arg((def.did, param_did));
        } else {
            tcx.ensure().unsafety_check_result(def.did);
        }
    }

    // has_ffi_unwind_calls query uses the raw mir, so make sure it is run.
    tcx.ensure().has_ffi_unwind_calls(def.did);

    let mut body = tcx.mir_built(def).steal();

    rustc_middle::mir::dump_mir(tcx, None, "mir_map", &0, &body, |_, _| Ok(()));

    pm::run_passes(
        tcx,
        &mut body,
        &[
            // MIR-level lints.
            &Lint(check_packed_ref::CheckPackedRef),
            &Lint(check_const_item_mutation::CheckConstItemMutation),
            &Lint(function_item_references::FunctionItemReferences),
            // What we need to do constant evaluation.
            &simplify::SimplifyCfg::new("initial"),
            &rustc_peek::SanityCheck, // Just a lint
        ],
    );
    tcx.alloc_steal_mir(body)
}

/// Compute the main MIR body and the list of MIR bodies of the promoteds.
fn mir_promoted<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> (&'tcx Steal<Body<'tcx>>, &'tcx Steal<IndexVec<Promoted, Body<'tcx>>>) {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_promoted(def);
    }

    // Ensure that we compute the `mir_const_qualif` for constants at
    // this point, before we steal the mir-const result.
    // Also this means promotion can rely on all const checks having been done.
    let const_qualifs = tcx.mir_const_qualif_opt_const_arg(def);
    let mut body = tcx.mir_const(def).steal();
    if let Some(error_reported) = const_qualifs.tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    let mut required_consts = Vec::new();
    let mut required_consts_visitor = RequiredConstsVisitor::new(&mut required_consts);
    for (bb, bb_data) in traversal::reverse_postorder(&body) {
        required_consts_visitor.visit_basic_block_data(bb, bb_data);
    }
    body.required_consts = required_consts;

    // What we need to run borrowck etc.
    let promote_pass = promote_consts::PromoteTemps::default();
    pm::run_passes(
        tcx,
        &mut body,
        &[
            &promote_pass,
            &simplify::SimplifyCfg::new("promote-consts"),
            &coverage::InstrumentCoverage,
        ],
    );

    let promoted = promote_pass.promoted_fragments.into_inner();
    (tcx.alloc_steal_mir(body), tcx.alloc_steal_promoted(promoted))
}

/// Compute the MIR that is used during CTFE (and thus has no optimizations run on it)
fn mir_for_ctfe<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx Body<'tcx> {
    let did = def_id.expect_local();
    if let Some(def) = ty::WithOptConstParam::try_lookup(did, tcx) {
        tcx.mir_for_ctfe_of_const_arg(def)
    } else {
        tcx.arena.alloc(inner_mir_for_ctfe(tcx, ty::WithOptConstParam::unknown(did)))
    }
}

/// Same as `mir_for_ctfe`, but used to get the MIR of a const generic parameter.
/// The docs on `WithOptConstParam` explain this a bit more, but the TLDR is that
/// we'd get cycle errors with `mir_for_ctfe`, because typeck would need to typeck
/// the const parameter while type checking the main body, which in turn would try
/// to type check the main body again.
fn mir_for_ctfe_of_const_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    (did, param_did): (LocalDefId, DefId),
) -> &'tcx Body<'tcx> {
    tcx.arena.alloc(inner_mir_for_ctfe(
        tcx,
        ty::WithOptConstParam { did, const_param_did: Some(param_did) },
    ))
}

fn inner_mir_for_ctfe(tcx: TyCtxt<'_>, def: ty::WithOptConstParam<LocalDefId>) -> Body<'_> {
    // FIXME: don't duplicate this between the optimized_mir/mir_for_ctfe queries
    if tcx.is_constructor(def.did.to_def_id()) {
        // There's no reason to run all of the MIR passes on constructors when
        // we can just output the MIR we want directly. This also saves const
        // qualification and borrow checking the trouble of special casing
        // constructors.
        return shim::build_adt_ctor(tcx, def.did.to_def_id());
    }

    let context = tcx
        .hir()
        .body_const_context(def.did)
        .expect("mir_for_ctfe should not be used for runtime functions");

    let body = tcx.mir_drops_elaborated_and_const_checked(def).borrow().clone();

    let mut body = remap_mir_for_const_eval_select(tcx, body, hir::Constness::Const);

    match context {
        // Do not const prop functions, either they get executed at runtime or exported to metadata,
        // so we run const prop on them, or they don't, in which case we const evaluate some control
        // flow paths of the function and any errors in those paths will get emitted as const eval
        // errors.
        hir::ConstContext::ConstFn => {}
        // Static items always get evaluated, so we can just let const eval see if any erroneous
        // control flow paths get executed.
        hir::ConstContext::Static(_) => {}
        // Associated constants get const prop run so we detect common failure situations in the
        // crate that defined the constant.
        // Technically we want to not run on regular const items, but oli-obk doesn't know how to
        // conveniently detect that at this point without looking at the HIR.
        hir::ConstContext::Const => {
            pm::run_passes(
                tcx,
                &mut body,
                &[
                    &const_prop::ConstProp,
                    &marker::PhaseChange(MirPhase::Runtime(RuntimePhase::Optimized)),
                ],
            );
        }
    }

    debug_assert!(!body.has_free_regions(), "Free regions in MIR for CTFE");

    body
}

/// Obtain just the main MIR (no promoteds) and run some cleanups on it. This also runs
/// mir borrowck *before* doing so in order to ensure that borrowck can be run and doesn't
/// end up missing the source MIR due to stealing happening.
fn mir_drops_elaborated_and_const_checked<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx Steal<Body<'tcx>> {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_drops_elaborated_and_const_checked(def);
    }

    let mir_borrowck = tcx.mir_borrowck_opt_const_arg(def);

    let is_fn_like = tcx.def_kind(def.did).is_fn_like();
    if is_fn_like {
        let did = def.did.to_def_id();
        let def = ty::WithOptConstParam::unknown(did);

        // Do not compute the mir call graph without said call graph actually being used.
        if inline::Inline.is_enabled(&tcx.sess) {
            let _ = tcx.mir_inliner_callees(ty::InstanceDef::Item(def));
        }
    }

    let (body, _) = tcx.mir_promoted(def);
    let mut body = body.steal();
    if let Some(error_reported) = mir_borrowck.tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    run_analysis_to_runtime_passes(tcx, &mut body);

    tcx.alloc_steal_mir(body)
}

fn run_analysis_to_runtime_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    assert!(body.phase == MirPhase::Analysis(AnalysisPhase::Initial));
    let did = body.source.def_id();

    debug!("analysis_mir_cleanup({:?})", did);
    run_analysis_cleanup_passes(tcx, body);
    assert!(body.phase == MirPhase::Analysis(AnalysisPhase::PostCleanup));

    // Do a little drop elaboration before const-checking if `const_precise_live_drops` is enabled.
    if check_consts::post_drop_elaboration::checking_enabled(&ConstCx::new(tcx, &body)) {
        pm::run_passes(
            tcx,
            body,
            &[
                &remove_uninit_drops::RemoveUninitDrops,
                &simplify::SimplifyCfg::new("remove-false-edges"),
            ],
        );
        check_consts::post_drop_elaboration::check_live_drops(tcx, &body); // FIXME: make this a MIR lint
    }

    debug!("runtime_mir_lowering({:?})", did);
    run_runtime_lowering_passes(tcx, body);
    assert!(body.phase == MirPhase::Runtime(RuntimePhase::Initial));

    debug!("runtime_mir_cleanup({:?})", did);
    run_runtime_cleanup_passes(tcx, body);
    assert!(body.phase == MirPhase::Runtime(RuntimePhase::PostCleanup));
}

// FIXME(JakobDegen): Can we make these lists of passes consts?

/// After this series of passes, no lifetime analysis based on borrowing can be done.
fn run_analysis_cleanup_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let passes: &[&dyn MirPass<'tcx>] = &[
        &remove_false_edges::RemoveFalseEdges,
        &simplify_branches::SimplifyConstCondition::new("initial"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &cleanup_post_borrowck::CleanupNonCodegenStatements,
        &simplify::SimplifyCfg::new("early-opt"),
        &deref_separator::Derefer,
        &marker::PhaseChange(MirPhase::Analysis(AnalysisPhase::PostCleanup)),
    ];

    pm::run_passes(tcx, body, passes);
}

/// Returns the sequence of passes that lowers analysis to runtime MIR.
fn run_runtime_lowering_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let passes: &[&dyn MirPass<'tcx>] = &[
        // These next passes must be executed together
        &add_call_guards::CriticalCallEdges,
        &elaborate_drops::ElaborateDrops,
        // This will remove extraneous landing pads which are no longer
        // necessary as well as well as forcing any call in a non-unwinding
        // function calling a possibly-unwinding function to abort the process.
        &abort_unwinding_calls::AbortUnwindingCalls,
        // AddMovesForPackedDrops needs to run after drop
        // elaboration.
        &add_moves_for_packed_drops::AddMovesForPackedDrops,
        // `AddRetag` needs to run after `ElaborateDrops`. Otherwise it should run fairly late,
        // but before optimizations begin.
        &elaborate_box_derefs::ElaborateBoxDerefs,
        &generator::StateTransform,
        &add_retag::AddRetag,
        // Deaggregator is necessary for const prop. We may want to consider implementing
        // CTFE support for aggregates.
        &deaggregator::Deaggregator,
        &Lint(const_prop_lint::ConstProp),
        &marker::PhaseChange(MirPhase::Runtime(RuntimePhase::Initial)),
    ];
    pm::run_passes_no_validate(tcx, body, passes);
}

/// Returns the sequence of passes that do the initial cleanup of runtime MIR.
fn run_runtime_cleanup_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let passes: &[&dyn MirPass<'tcx>] = &[
        &elaborate_box_derefs::ElaborateBoxDerefs,
        &lower_intrinsics::LowerIntrinsics,
        &simplify::SimplifyCfg::new("elaborate-drops"),
        &marker::PhaseChange(MirPhase::Runtime(RuntimePhase::PostCleanup)),
    ];

    pm::run_passes(tcx, body, passes);
}

fn run_optimization_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    fn o1<T>(x: T) -> WithMinOptLevel<T> {
        WithMinOptLevel(1, x)
    }

    // The main optimizations that we do on MIR.
    pm::run_passes(
        tcx,
        body,
        &[
            &reveal_all::RevealAll, // has to be done before inlining, since inlined code is in RevealAll mode.
            &lower_slice_len::LowerSliceLenCalls, // has to be done before inlining, otherwise actual call will be almost always inlined. Also simple, so can just do first
            &normalize_array_len::NormalizeArrayLen, // has to run after `slice::len` lowering
            &unreachable_prop::UnreachablePropagation,
            &uninhabited_enum_branching::UninhabitedEnumBranching,
            &o1(simplify::SimplifyCfg::new("after-uninhabited-enum-branching")),
            &inline::Inline,
            &remove_storage_markers::RemoveStorageMarkers,
            &remove_zsts::RemoveZsts,
            &const_goto::ConstGoto,
            &remove_unneeded_drops::RemoveUnneededDrops,
            &match_branches::MatchBranchSimplification,
            // inst combine is after MatchBranchSimplification to clean up Ne(_1, false)
            &multiple_return_terminators::MultipleReturnTerminators,
            &instcombine::InstCombine,
            &separate_const_switch::SeparateConstSwitch,
            //
            // FIXME(#70073): This pass is responsible for both optimization as well as some lints.
            &const_prop::ConstProp,
            //
            // Const-prop runs unconditionally, but doesn't mutate the MIR at mir-opt-level=0.
            &const_debuginfo::ConstDebugInfo,
            &o1(simplify_branches::SimplifyConstCondition::new("after-const-prop")),
            &early_otherwise_branch::EarlyOtherwiseBranch,
            &simplify_comparison_integral::SimplifyComparisonIntegral,
            &simplify_try::SimplifyArmIdentity,
            &simplify_try::SimplifyBranchSame,
            &dead_store_elimination::DeadStoreElimination,
            &dest_prop::DestinationPropagation,
            &o1(simplify_branches::SimplifyConstCondition::new("final")),
            &o1(remove_noop_landing_pads::RemoveNoopLandingPads),
            &o1(simplify::SimplifyCfg::new("final")),
            &nrvo::RenameReturnPlace,
            &simplify::SimplifyLocals,
            &multiple_return_terminators::MultipleReturnTerminators,
            &deduplicate_blocks::DeduplicateBlocks,
            // Some cleanup necessary at least for LLVM and potentially other codegen backends.
            &add_call_guards::CriticalCallEdges,
            &marker::PhaseChange(MirPhase::Runtime(RuntimePhase::Optimized)),
            // Dump the end result for testing and debugging purposes.
            &dump_mir::Marker("PreCodegen"),
        ],
    );
}

/// Optimize the MIR and prepare it for codegen.
fn optimized_mir<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> &'tcx Body<'tcx> {
    let did = did.expect_local();
    assert_eq!(ty::WithOptConstParam::try_lookup(did, tcx), None);
    tcx.arena.alloc(inner_optimized_mir(tcx, did))
}

fn inner_optimized_mir(tcx: TyCtxt<'_>, did: LocalDefId) -> Body<'_> {
    if tcx.is_constructor(did.to_def_id()) {
        // There's no reason to run all of the MIR passes on constructors when
        // we can just output the MIR we want directly. This also saves const
        // qualification and borrow checking the trouble of special casing
        // constructors.
        return shim::build_adt_ctor(tcx, did.to_def_id());
    }

    match tcx.hir().body_const_context(did) {
        // Run the `mir_for_ctfe` query, which depends on `mir_drops_elaborated_and_const_checked`
        // which we are going to steal below. Thus we need to run `mir_for_ctfe` first, so it
        // computes and caches its result.
        Some(hir::ConstContext::ConstFn) => tcx.ensure().mir_for_ctfe(did),
        None => {}
        Some(other) => panic!("do not use `optimized_mir` for constants: {:?}", other),
    }
    debug!("about to call mir_drops_elaborated...");
    let body =
        tcx.mir_drops_elaborated_and_const_checked(ty::WithOptConstParam::unknown(did)).steal();
    let mut body = remap_mir_for_const_eval_select(tcx, body, hir::Constness::NotConst);
    debug!("body: {:#?}", body);
    run_optimization_passes(tcx, &mut body);

    debug_assert!(!body.has_free_regions(), "Free regions in optimized MIR");

    body
}

/// Fetch all the promoteds of an item and prepare their MIR bodies to be ready for
/// constant evaluation once all substitutions become known.
fn promoted_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx IndexVec<Promoted, Body<'tcx>> {
    if tcx.is_constructor(def.did.to_def_id()) {
        return tcx.arena.alloc(IndexVec::new());
    }

    let tainted_by_errors = tcx.mir_borrowck_opt_const_arg(def).tainted_by_errors;
    let mut promoted = tcx.mir_promoted(def).1.steal();

    for body in &mut promoted {
        if let Some(error_reported) = tainted_by_errors {
            body.tainted_by_errors = Some(error_reported);
        }
        run_analysis_to_runtime_passes(tcx, body);
    }

    debug_assert!(!promoted.has_free_regions(), "Free regions in promoted MIR");

    tcx.arena.alloc(promoted)
}
