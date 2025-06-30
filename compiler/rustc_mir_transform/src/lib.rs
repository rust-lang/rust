// tidy-alphabetical-start
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(const_type_name)]
#![feature(cow_is_borrowed)]
#![feature(file_buffered)]
#![feature(if_let_guard)]
#![feature(impl_trait_in_assoc_type)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
// tidy-alphabetical-end

use hir::ConstContext;
use required_consts::RequiredConstsVisitor;
use rustc_const_eval::check_consts::{self, ConstCx};
use rustc_const_eval::util;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_middle::mir::{
    AnalysisPhase, Body, CallSource, ClearCrossCrate, ConstOperand, ConstQualifs, LocalDecl,
    MirPhase, Operand, Place, ProjectionElem, Promoted, RuntimePhase, Rvalue, START_BLOCK,
    SourceInfo, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_middle::util::Providers;
use rustc_middle::{bug, query, span_bug};
use rustc_mir_build::builder::build_mir;
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, sym};
use tracing::debug;

#[macro_use]
mod pass_manager;

use std::sync::LazyLock;

use pass_manager::{self as pm, Lint, MirLint, MirPass, WithMinOptLevel};

mod check_pointers;
mod cost_checker;
mod cross_crate_inline;
mod deduce_param_attrs;
mod elaborate_drop;
mod errors;
mod ffi_unwind_calls;
mod lint;
mod lint_tail_expr_drop_order;
mod patch;
mod shim;
mod ssa;

/// We import passes via this macro so that we can have a static list of pass names
/// (used to verify CLI arguments). It takes a list of modules, followed by the passes
/// declared within them.
/// ```ignore,macro-test
/// declare_passes! {
///     // Declare a single pass from the module `abort_unwinding_calls`
///     mod abort_unwinding_calls : AbortUnwindingCalls;
///     // When passes are grouped together as an enum, declare the two constituent passes
///     mod add_call_guards : AddCallGuards {
///         AllCallEdges,
///         CriticalCallEdges
///     };
///     // Declares multiple pass groups, each containing their own constituent passes
///     mod simplify : SimplifyCfg {
///         Initial,
///         /* omitted */
///     }, SimplifyLocals {
///         BeforeConstProp,
///         /* omitted */
///     };
/// }
/// ```
macro_rules! declare_passes {
    (
        $(
            $vis:vis mod $mod_name:ident : $($pass_name:ident $( { $($ident:ident),* } )?),+ $(,)?;
        )*
    ) => {
        $(
            $vis mod $mod_name;
            $(
                // Make sure the type name is correct
                #[allow(unused_imports)]
                use $mod_name::$pass_name as _;
            )+
        )*

        static PASS_NAMES: LazyLock<FxIndexSet<&str>> = LazyLock::new(|| [
            // Fake marker pass
            "PreCodegen",
            $(
                $(
                    stringify!($pass_name),
                    $(
                        $(
                            $mod_name::$pass_name::$ident.name(),
                        )*
                    )?
                )+
            )*
        ].into_iter().collect());
    };
}

declare_passes! {
    mod abort_unwinding_calls : AbortUnwindingCalls;
    mod add_call_guards : AddCallGuards { AllCallEdges, CriticalCallEdges };
    mod add_moves_for_packed_drops : AddMovesForPackedDrops;
    mod add_retag : AddRetag;
    mod add_subtyping_projections : Subtyper;
    mod check_inline : CheckForceInline;
    mod check_call_recursion : CheckCallRecursion, CheckDropRecursion;
    mod check_alignment : CheckAlignment;
    mod check_enums : CheckEnums;
    mod check_const_item_mutation : CheckConstItemMutation;
    mod check_null : CheckNull;
    mod check_packed_ref : CheckPackedRef;
    // This pass is public to allow external drivers to perform MIR cleanup
    pub mod cleanup_post_borrowck : CleanupPostBorrowck;

    mod copy_prop : CopyProp;
    mod coroutine : StateTransform;
    mod coverage : InstrumentCoverage;
    mod ctfe_limit : CtfeLimit;
    mod dataflow_const_prop : DataflowConstProp;
    mod dead_store_elimination : DeadStoreElimination {
        Initial,
        Final
    };
    mod deref_separator : Derefer;
    mod dest_prop : DestinationPropagation;
    pub mod dump_mir : Marker;
    mod early_otherwise_branch : EarlyOtherwiseBranch;
    mod elaborate_box_derefs : ElaborateBoxDerefs;
    mod elaborate_drops : ElaborateDrops;
    mod function_item_references : FunctionItemReferences;
    mod gvn : GVN;
    // Made public so that `mir_drops_elaborated_and_const_checked` can be overridden
    // by custom rustc drivers, running all the steps by themselves. See #114628.
    pub mod inline : Inline, ForceInline;
    mod impossible_predicates : ImpossiblePredicates;
    mod instsimplify : InstSimplify { BeforeInline, AfterSimplifyCfg };
    mod jump_threading : JumpThreading;
    mod known_panics_lint : KnownPanicsLint;
    mod large_enums : EnumSizeOpt;
    mod lower_intrinsics : LowerIntrinsics;
    mod lower_slice_len : LowerSliceLenCalls;
    mod match_branches : MatchBranchSimplification;
    mod mentioned_items : MentionedItems;
    mod multiple_return_terminators : MultipleReturnTerminators;
    mod nrvo : RenameReturnPlace;
    mod post_drop_elaboration : CheckLiveDrops;
    mod prettify : ReorderBasicBlocks, ReorderLocals;
    mod promote_consts : PromoteTemps;
    mod ref_prop : ReferencePropagation;
    mod remove_noop_landing_pads : RemoveNoopLandingPads;
    mod remove_place_mention : RemovePlaceMention;
    mod remove_storage_markers : RemoveStorageMarkers;
    mod remove_uninit_drops : RemoveUninitDrops;
    mod remove_unneeded_drops : RemoveUnneededDrops;
    mod remove_zsts : RemoveZsts;
    mod required_consts : RequiredConstsVisitor;
    mod post_analysis_normalize : PostAnalysisNormalize;
    mod sanity_check : SanityCheck;
    // This pass is public to allow external drivers to perform MIR cleanup
    pub mod simplify :
        SimplifyCfg {
            Initial,
            PromoteConsts,
            RemoveFalseEdges,
            PostAnalysis,
            PreOptimizations,
            Final,
            MakeShim,
            AfterUnreachableEnumBranching
        },
        SimplifyLocals {
            BeforeConstProp,
            AfterGVN,
            Final
        };
    mod simplify_branches : SimplifyConstCondition {
        AfterConstProp,
        Final
    };
    mod simplify_comparison_integral : SimplifyComparisonIntegral;
    mod single_use_consts : SingleUseConsts;
    mod sroa : ScalarReplacementOfAggregates;
    mod strip_debuginfo : StripDebugInfo;
    mod unreachable_enum_branching : UnreachableEnumBranching;
    mod unreachable_prop : UnreachablePropagation;
    mod validate : Validator;
}

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    coverage::query::provide(providers);
    ffi_unwind_calls::provide(providers);
    shim::provide(providers);
    cross_crate_inline::provide(providers);
    providers.queries = query::Providers {
        mir_keys,
        mir_built,
        mir_const_qualif,
        mir_promoted,
        mir_drops_elaborated_and_const_checked,
        mir_for_ctfe,
        mir_coroutine_witnesses: coroutine::mir_coroutine_witnesses,
        optimized_mir,
        is_mir_available,
        is_ctfe_mir_available: is_mir_available,
        mir_callgraph_cyclic: inline::cycle::mir_callgraph_cyclic,
        mir_inliner_callees: inline::cycle::mir_inliner_callees,
        promoted_mir,
        deduced_param_attrs: deduce_param_attrs::deduced_param_attrs,
        coroutine_by_move_body_def_id: coroutine::coroutine_by_move_body_def_id,
        ..providers.queries
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
                func: Operand::Constant(box ConstOperand { ref const_, .. }),
                ref mut args,
                destination,
                target,
                unwind,
                fn_span,
                ..
            } if let ty::FnDef(def_id, _) = *const_.ty().kind()
                && tcx.is_intrinsic(def_id, sym::const_eval_select) =>
            {
                let Ok([tupled_args, called_in_const, called_at_rt]) = take_array(args) else {
                    unreachable!()
                };
                let ty = tupled_args.node.ty(&body.local_decls, tcx);
                let fields = ty.tuple_fields();
                let num_args = fields.len();
                let func =
                    if context == hir::Constness::Const { called_in_const } else { called_at_rt };
                let (method, place): (fn(Place<'tcx>) -> Operand<'tcx>, Place<'tcx>) =
                    match tupled_args.node {
                        Operand::Constant(_) => {
                            // There is no good way of extracting a tuple arg from a constant
                            // (const generic stuff) so we just create a temporary and deconstruct
                            // that.
                            let local = body.local_decls.push(LocalDecl::new(ty, fn_span));
                            bb.statements.push(Statement::new(
                                SourceInfo::outermost(fn_span),
                                StatementKind::Assign(Box::new((
                                    local.into(),
                                    Rvalue::Use(tupled_args.node.clone()),
                                ))),
                            ));
                            (Operand::Move, local.into())
                        }
                        Operand::Move(place) => (Operand::Move, place),
                        Operand::Copy(place) => (Operand::Copy, place),
                    };
                let place_elems = place.projection;
                let arguments = (0..num_args)
                    .map(|x| {
                        let mut place_elems = place_elems.to_vec();
                        place_elems.push(ProjectionElem::Field(x.into(), fields[x]));
                        let projection = tcx.mk_place_elems(&place_elems);
                        let place = Place { local: place.local, projection };
                        Spanned { node: method(place), span: DUMMY_SP }
                    })
                    .collect();
                terminator.kind = TerminatorKind::Call {
                    func: func.node,
                    args: arguments,
                    destination,
                    target,
                    unwind,
                    call_source: CallSource::Misc,
                    fn_span,
                };
            }
            _ => {}
        }
    }
    body
}

fn take_array<T, const N: usize>(b: &mut Box<[T]>) -> Result<[T; N], Box<[T]>> {
    let b: Box<[T; N]> = std::mem::take(b).try_into()?;
    Ok(*b)
}

fn is_mir_available(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    tcx.mir_keys(()).contains(&def_id)
}

/// Finds the full set of `DefId`s within the current crate that have
/// MIR associated with them.
fn mir_keys(tcx: TyCtxt<'_>, (): ()) -> FxIndexSet<LocalDefId> {
    // All body-owners have MIR associated with them.
    let mut set: FxIndexSet<_> = tcx.hir_body_owners().collect();

    // Remove the fake bodies for `global_asm!`, since they're not useful
    // to be emitted (`--emit=mir`) or encoded (in metadata).
    set.retain(|&def_id| !matches!(tcx.def_kind(def_id), DefKind::GlobalAsm));

    // Coroutine-closures (e.g. async closures) have an additional by-move MIR
    // body that isn't in the HIR.
    for body_owner in tcx.hir_body_owners() {
        if let DefKind::Closure = tcx.def_kind(body_owner)
            && tcx.needs_coroutine_by_move_body_def_id(body_owner.to_def_id())
        {
            set.insert(tcx.coroutine_by_move_body_def_id(body_owner).expect_local());
        }
    }

    // tuple struct/variant constructors have MIR, but they don't have a BodyId,
    // so we need to build them separately.
    for item in tcx.hir_crate_items(()).free_items() {
        if let DefKind::Struct | DefKind::Enum = tcx.def_kind(item.owner_id) {
            for variant in tcx.adt_def(item.owner_id).variants() {
                if let Some((CtorKind::Fn, ctor_def_id)) = variant.ctor {
                    set.insert(ctor_def_id.expect_local());
                }
            }
        }
    }

    set
}

fn mir_const_qualif(tcx: TyCtxt<'_>, def: LocalDefId) -> ConstQualifs {
    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_promoted()`, which steals
    // from `mir_built()`, forces this query to execute before
    // performing the steal.
    let body = &tcx.mir_built(def).borrow();
    let ccx = check_consts::ConstCx::new(tcx, body);
    // No need to const-check a non-const `fn`.
    match ccx.const_kind {
        Some(ConstContext::Const { .. } | ConstContext::Static(_) | ConstContext::ConstFn) => {}
        None => span_bug!(
            tcx.def_span(def),
            "`mir_const_qualif` should only be called on const fns and const items"
        ),
    }

    if body.return_ty().references_error() {
        // It's possible to reach here without an error being emitted (#121103).
        tcx.dcx().span_delayed_bug(body.span, "mir_const_qualif: MIR had errors");
        return Default::default();
    }

    let mut validator = check_consts::check::Checker::new(&ccx);
    validator.check_body();

    // We return the qualifs in the return place for every MIR body, even though it is only used
    // when deciding to promote a reference to a `const` for now.
    validator.qualifs_in_return_place()
}

fn mir_built(tcx: TyCtxt<'_>, def: LocalDefId) -> &Steal<Body<'_>> {
    let mut body = build_mir(tcx, def);

    pass_manager::dump_mir_for_phase_change(tcx, &body);

    pm::run_passes(
        tcx,
        &mut body,
        &[
            // MIR-level lints.
            &Lint(check_inline::CheckForceInline),
            &Lint(check_call_recursion::CheckCallRecursion),
            &Lint(check_packed_ref::CheckPackedRef),
            &Lint(check_const_item_mutation::CheckConstItemMutation),
            &Lint(function_item_references::FunctionItemReferences),
            // What we need to do constant evaluation.
            &simplify::SimplifyCfg::Initial,
            &Lint(sanity_check::SanityCheck),
        ],
        None,
        pm::Optimizations::Allowed,
    );
    tcx.alloc_steal_mir(body)
}

/// Compute the main MIR body and the list of MIR bodies of the promoteds.
fn mir_promoted(
    tcx: TyCtxt<'_>,
    def: LocalDefId,
) -> (&Steal<Body<'_>>, &Steal<IndexVec<Promoted, Body<'_>>>) {
    // Ensure that we compute the `mir_const_qualif` for constants at
    // this point, before we steal the mir-const result.
    // Also this means promotion can rely on all const checks having been done.

    let const_qualifs = match tcx.def_kind(def) {
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure
            if tcx.constness(def) == hir::Constness::Const
                || tcx.is_const_default_method(def.to_def_id()) =>
        {
            tcx.mir_const_qualif(def)
        }
        DefKind::AssocConst
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::InlineConst
        | DefKind::AnonConst => tcx.mir_const_qualif(def),
        _ => ConstQualifs::default(),
    };

    // the `has_ffi_unwind_calls` query uses the raw mir, so make sure it is run.
    tcx.ensure_done().has_ffi_unwind_calls(def);

    // the `by_move_body` query uses the raw mir, so make sure it is run.
    if tcx.needs_coroutine_by_move_body_def_id(def.to_def_id()) {
        tcx.ensure_done().coroutine_by_move_body_def_id(def);
    }

    let mut body = tcx.mir_built(def).steal();
    if let Some(error_reported) = const_qualifs.tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    // Collect `required_consts` *before* promotion, so if there are any consts being promoted
    // we still add them to the list in the outer MIR body.
    RequiredConstsVisitor::compute_required_consts(&mut body);

    // What we need to run borrowck etc.
    let promote_pass = promote_consts::PromoteTemps::default();
    pm::run_passes(
        tcx,
        &mut body,
        &[&promote_pass, &simplify::SimplifyCfg::PromoteConsts, &coverage::InstrumentCoverage],
        Some(MirPhase::Analysis(AnalysisPhase::Initial)),
        pm::Optimizations::Allowed,
    );

    lint_tail_expr_drop_order::run_lint(tcx, def, &body);

    let promoted = promote_pass.promoted_fragments.into_inner();
    (tcx.alloc_steal_mir(body), tcx.alloc_steal_promoted(promoted))
}

/// Compute the MIR that is used during CTFE (and thus has no optimizations run on it)
fn mir_for_ctfe(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &Body<'_> {
    tcx.arena.alloc(inner_mir_for_ctfe(tcx, def_id))
}

fn inner_mir_for_ctfe(tcx: TyCtxt<'_>, def: LocalDefId) -> Body<'_> {
    // FIXME: don't duplicate this between the optimized_mir/mir_for_ctfe queries
    if tcx.is_constructor(def.to_def_id()) {
        // There's no reason to run all of the MIR passes on constructors when
        // we can just output the MIR we want directly. This also saves const
        // qualification and borrow checking the trouble of special casing
        // constructors.
        return shim::build_adt_ctor(tcx, def.to_def_id());
    }

    let body = tcx.mir_drops_elaborated_and_const_checked(def);
    let body = match tcx.hir_body_const_context(def) {
        // consts and statics do not have `optimized_mir`, so we can steal the body instead of
        // cloning it.
        Some(hir::ConstContext::Const { .. } | hir::ConstContext::Static(_)) => body.steal(),
        Some(hir::ConstContext::ConstFn) => body.borrow().clone(),
        None => bug!("`mir_for_ctfe` called on non-const {def:?}"),
    };

    let mut body = remap_mir_for_const_eval_select(tcx, body, hir::Constness::Const);
    pm::run_passes(tcx, &mut body, &[&ctfe_limit::CtfeLimit], None, pm::Optimizations::Allowed);

    body
}

/// Obtain just the main MIR (no promoteds) and run some cleanups on it. This also runs
/// mir borrowck *before* doing so in order to ensure that borrowck can be run and doesn't
/// end up missing the source MIR due to stealing happening.
fn mir_drops_elaborated_and_const_checked(tcx: TyCtxt<'_>, def: LocalDefId) -> &Steal<Body<'_>> {
    if tcx.is_coroutine(def.to_def_id()) {
        tcx.ensure_done().mir_coroutine_witnesses(def);
    }

    // We only need to borrowck non-synthetic MIR.
    let tainted_by_errors = if !tcx.is_synthetic_mir(def) {
        tcx.mir_borrowck(tcx.typeck_root_def_id(def.to_def_id()).expect_local()).err()
    } else {
        None
    };

    let is_fn_like = tcx.def_kind(def).is_fn_like();
    if is_fn_like {
        // Do not compute the mir call graph without said call graph actually being used.
        if pm::should_run_pass(tcx, &inline::Inline, pm::Optimizations::Allowed)
            || inline::ForceInline::should_run_pass_for_callee(tcx, def.to_def_id())
        {
            tcx.ensure_done().mir_inliner_callees(ty::InstanceKind::Item(def.to_def_id()));
        }
    }

    let (body, _) = tcx.mir_promoted(def);
    let mut body = body.steal();

    if let Some(error_reported) = tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    // Also taint the body if it's within a top-level item that is not well formed.
    //
    // We do this check here and not during `mir_promoted` because that may result
    // in borrowck cycles if WF requires looking into an opaque hidden type.
    let root = tcx.typeck_root_def_id(def.to_def_id());
    match tcx.def_kind(root) {
        DefKind::Fn
        | DefKind::AssocFn
        | DefKind::Static { .. }
        | DefKind::Const
        | DefKind::AssocConst => {
            if let Err(guar) = tcx.ensure_ok().check_well_formed(root.expect_local()) {
                body.tainted_by_errors = Some(guar);
            }
        }
        _ => {}
    }

    run_analysis_to_runtime_passes(tcx, &mut body);

    tcx.alloc_steal_mir(body)
}

// Made public so that `mir_drops_elaborated_and_const_checked` can be overridden
// by custom rustc drivers, running all the steps by themselves. See #114628.
pub fn run_analysis_to_runtime_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    assert!(body.phase == MirPhase::Analysis(AnalysisPhase::Initial));
    let did = body.source.def_id();

    debug!("analysis_mir_cleanup({:?})", did);
    run_analysis_cleanup_passes(tcx, body);
    assert!(body.phase == MirPhase::Analysis(AnalysisPhase::PostCleanup));

    // Do a little drop elaboration before const-checking if `const_precise_live_drops` is enabled.
    if check_consts::post_drop_elaboration::checking_enabled(&ConstCx::new(tcx, body)) {
        pm::run_passes(
            tcx,
            body,
            &[
                &remove_uninit_drops::RemoveUninitDrops,
                &simplify::SimplifyCfg::RemoveFalseEdges,
                &Lint(post_drop_elaboration::CheckLiveDrops),
            ],
            None,
            pm::Optimizations::Allowed,
        );
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
        &impossible_predicates::ImpossiblePredicates,
        &cleanup_post_borrowck::CleanupPostBorrowck,
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &simplify::SimplifyCfg::PostAnalysis,
        &deref_separator::Derefer,
    ];

    pm::run_passes(
        tcx,
        body,
        passes,
        Some(MirPhase::Analysis(AnalysisPhase::PostCleanup)),
        pm::Optimizations::Allowed,
    );
}

/// Returns the sequence of passes that lowers analysis to runtime MIR.
fn run_runtime_lowering_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let passes: &[&dyn MirPass<'tcx>] = &[
        // These next passes must be executed together.
        &add_call_guards::CriticalCallEdges,
        // Must be done before drop elaboration because we need to drop opaque types, too.
        &post_analysis_normalize::PostAnalysisNormalize,
        // Calling this after `PostAnalysisNormalize` ensures that we don't deal with opaque types.
        &add_subtyping_projections::Subtyper,
        &elaborate_drops::ElaborateDrops,
        // Needs to happen after drop elaboration.
        &Lint(check_call_recursion::CheckDropRecursion),
        // This will remove extraneous landing pads which are no longer
        // necessary as well as forcing any call in a non-unwinding
        // function calling a possibly-unwinding function to abort the process.
        &abort_unwinding_calls::AbortUnwindingCalls,
        // AddMovesForPackedDrops needs to run after drop
        // elaboration.
        &add_moves_for_packed_drops::AddMovesForPackedDrops,
        // `AddRetag` needs to run after `ElaborateDrops` but before `ElaborateBoxDerefs`.
        // Otherwise it should run fairly late, but before optimizations begin.
        &add_retag::AddRetag,
        &elaborate_box_derefs::ElaborateBoxDerefs,
        &coroutine::StateTransform,
        &Lint(known_panics_lint::KnownPanicsLint),
    ];
    pm::run_passes_no_validate(tcx, body, passes, Some(MirPhase::Runtime(RuntimePhase::Initial)));
}

/// Returns the sequence of passes that do the initial cleanup of runtime MIR.
fn run_runtime_cleanup_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let passes: &[&dyn MirPass<'tcx>] = &[
        &lower_intrinsics::LowerIntrinsics,
        &remove_place_mention::RemovePlaceMention,
        &simplify::SimplifyCfg::PreOptimizations,
    ];

    pm::run_passes(
        tcx,
        body,
        passes,
        Some(MirPhase::Runtime(RuntimePhase::PostCleanup)),
        pm::Optimizations::Allowed,
    );

    // Clear this by anticipation. Optimizations and runtime MIR have no reason to look
    // into this information, which is meant for borrowck diagnostics.
    for decl in &mut body.local_decls {
        decl.local_info = ClearCrossCrate::Clear;
    }
}

pub(crate) fn run_optimization_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    fn o1<T>(x: T) -> WithMinOptLevel<T> {
        WithMinOptLevel(1, x)
    }

    let def_id = body.source.def_id();
    let optimizations = if tcx.def_kind(def_id).has_codegen_attrs()
        && tcx.codegen_fn_attrs(def_id).optimize.do_not_optimize()
    {
        pm::Optimizations::Suppressed
    } else {
        pm::Optimizations::Allowed
    };

    // The main optimizations that we do on MIR.
    pm::run_passes(
        tcx,
        body,
        &[
            // Add some UB checks before any UB gets optimized away.
            &check_alignment::CheckAlignment,
            &check_null::CheckNull,
            &check_enums::CheckEnums,
            // Before inlining: trim down MIR with passes to reduce inlining work.

            // Has to be done before inlining, otherwise actual call will be almost always inlined.
            // Also simple, so can just do first.
            &lower_slice_len::LowerSliceLenCalls,
            // Perform instsimplify before inline to eliminate some trivial calls (like clone
            // shims).
            &instsimplify::InstSimplify::BeforeInline,
            // Perform inlining of `#[rustc_force_inline]`-annotated callees.
            &inline::ForceInline,
            // Perform inlining, which may add a lot of code.
            &inline::Inline,
            // Code from other crates may have storage markers, so this needs to happen after
            // inlining.
            &remove_storage_markers::RemoveStorageMarkers,
            // Inlining and instantiation may introduce ZST and useless drops.
            &remove_zsts::RemoveZsts,
            &remove_unneeded_drops::RemoveUnneededDrops,
            // Type instantiation may create uninhabited enums.
            // Also eliminates some unreachable branches based on variants of enums.
            &unreachable_enum_branching::UnreachableEnumBranching,
            &unreachable_prop::UnreachablePropagation,
            &o1(simplify::SimplifyCfg::AfterUnreachableEnumBranching),
            // Inlining may have introduced a lot of redundant code and a large move pattern.
            // Now, we need to shrink the generated MIR.
            &ref_prop::ReferencePropagation,
            &sroa::ScalarReplacementOfAggregates,
            &multiple_return_terminators::MultipleReturnTerminators,
            // After simplifycfg, it allows us to discover new opportunities for peephole
            // optimizations.
            &instsimplify::InstSimplify::AfterSimplifyCfg,
            &simplify::SimplifyLocals::BeforeConstProp,
            &dead_store_elimination::DeadStoreElimination::Initial,
            &gvn::GVN,
            &simplify::SimplifyLocals::AfterGVN,
            &match_branches::MatchBranchSimplification,
            &dataflow_const_prop::DataflowConstProp,
            &single_use_consts::SingleUseConsts,
            &o1(simplify_branches::SimplifyConstCondition::AfterConstProp),
            &jump_threading::JumpThreading,
            &early_otherwise_branch::EarlyOtherwiseBranch,
            &simplify_comparison_integral::SimplifyComparisonIntegral,
            &dest_prop::DestinationPropagation,
            &o1(simplify_branches::SimplifyConstCondition::Final),
            &o1(remove_noop_landing_pads::RemoveNoopLandingPads),
            &o1(simplify::SimplifyCfg::Final),
            // After the last SimplifyCfg, because this wants one-block functions.
            &strip_debuginfo::StripDebugInfo,
            &copy_prop::CopyProp,
            &dead_store_elimination::DeadStoreElimination::Final,
            &nrvo::RenameReturnPlace,
            &simplify::SimplifyLocals::Final,
            &multiple_return_terminators::MultipleReturnTerminators,
            &large_enums::EnumSizeOpt { discrepancy: 128 },
            // Some cleanup necessary at least for LLVM and potentially other codegen backends.
            &add_call_guards::CriticalCallEdges,
            // Cleanup for human readability, off by default.
            &prettify::ReorderBasicBlocks,
            &prettify::ReorderLocals,
            // Dump the end result for testing and debugging purposes.
            &dump_mir::Marker("PreCodegen"),
        ],
        Some(MirPhase::Runtime(RuntimePhase::Optimized)),
        optimizations,
    );
}

/// Optimize the MIR and prepare it for codegen.
fn optimized_mir(tcx: TyCtxt<'_>, did: LocalDefId) -> &Body<'_> {
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

    match tcx.hir_body_const_context(did) {
        // Run the `mir_for_ctfe` query, which depends on `mir_drops_elaborated_and_const_checked`
        // which we are going to steal below. Thus we need to run `mir_for_ctfe` first, so it
        // computes and caches its result.
        Some(hir::ConstContext::ConstFn) => tcx.ensure_done().mir_for_ctfe(did),
        None => {}
        Some(other) => panic!("do not use `optimized_mir` for constants: {other:?}"),
    }
    debug!("about to call mir_drops_elaborated...");
    let body = tcx.mir_drops_elaborated_and_const_checked(did).steal();
    let mut body = remap_mir_for_const_eval_select(tcx, body, hir::Constness::NotConst);

    if body.tainted_by_errors.is_some() {
        return body;
    }

    // Before doing anything, remember which items are being mentioned so that the set of items
    // visited does not depend on the optimization level.
    // We do not use `run_passes` for this as that might skip the pass if `injection_phase` is set.
    mentioned_items::MentionedItems.run_pass(tcx, &mut body);

    // If `mir_drops_elaborated_and_const_checked` found that the current body has unsatisfiable
    // predicates, it will shrink the MIR to a single `unreachable` terminator.
    // More generally, if MIR is a lone `unreachable`, there is nothing to optimize.
    if let TerminatorKind::Unreachable = body.basic_blocks[START_BLOCK].terminator().kind
        && body.basic_blocks[START_BLOCK].statements.is_empty()
    {
        return body;
    }

    run_optimization_passes(tcx, &mut body);

    body
}

/// Fetch all the promoteds of an item and prepare their MIR bodies to be ready for
/// constant evaluation once all generic parameters become known.
fn promoted_mir(tcx: TyCtxt<'_>, def: LocalDefId) -> &IndexVec<Promoted, Body<'_>> {
    if tcx.is_constructor(def.to_def_id()) {
        return tcx.arena.alloc(IndexVec::new());
    }

    if !tcx.is_synthetic_mir(def) {
        tcx.ensure_done().mir_borrowck(tcx.typeck_root_def_id(def.to_def_id()).expect_local());
    }
    let mut promoted = tcx.mir_promoted(def).1.steal();

    for body in &mut promoted {
        run_analysis_to_runtime_passes(tcx, body);
    }

    tcx.arena.alloc(promoted)
}
