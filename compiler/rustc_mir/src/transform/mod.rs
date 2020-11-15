use crate::{shim, util};
use required_consts::RequiredConstsVisitor;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::Visitor as _;
use rustc_middle::mir::{traversal, Body, ConstQualifs, MirPhase, Promoted};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

pub mod add_call_guards;
pub mod add_moves_for_packed_drops;
pub mod add_retag;
pub mod check_const_item_mutation;
pub mod check_consts;
pub mod check_packed_ref;
pub mod check_unsafety;
pub mod cleanup_post_borrowck;
pub mod const_prop;
pub mod coverage;
pub mod deaggregator;
pub mod dest_prop;
pub mod dump_mir;
pub mod early_otherwise_branch;
pub mod elaborate_drops;
pub mod function_item_references;
pub mod generator;
pub mod inline;
pub mod instcombine;
pub mod lower_intrinsics;
pub mod match_branches;
pub mod multiple_return_terminators;
pub mod no_landing_pads;
pub mod nrvo;
pub mod promote_consts;
pub mod remove_noop_landing_pads;
pub mod remove_unneeded_drops;
pub mod required_consts;
pub mod rustc_peek;
pub mod simplify;
pub mod simplify_branches;
pub mod simplify_comparison_integral;
pub mod simplify_try;
pub mod uninhabited_enum_branching;
pub mod unreachable_prop;
pub mod validate;

pub use rustc_middle::mir::MirSource;

pub(crate) fn provide(providers: &mut Providers) {
    self::check_unsafety::provide(providers);
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
        optimized_mir,
        optimized_mir_of_const_arg,
        is_mir_available,
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
    coverage::query::provide(providers);
}

fn is_mir_available(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.mir_keys(def_id.krate).contains(&def_id.expect_local())
}

/// Finds the full set of `DefId`s within the current crate that have
/// MIR associated with them.
fn mir_keys(tcx: TyCtxt<'_>, krate: CrateNum) -> FxHashSet<LocalDefId> {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = FxHashSet::default();

    // All body-owners have MIR associated with them.
    set.extend(tcx.body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        set: &'a mut FxHashSet<LocalDefId>,
    }
    impl<'a, 'tcx> Visitor<'tcx> for GatherCtors<'a, 'tcx> {
        fn visit_variant_data(
            &mut self,
            v: &'tcx hir::VariantData<'tcx>,
            _: Symbol,
            _: &'tcx hir::Generics<'tcx>,
            _: hir::HirId,
            _: Span,
        ) {
            if let hir::VariantData::Tuple(_, hir_id) = *v {
                self.set.insert(self.tcx.hir().local_def_id(hir_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        type Map = intravisit::ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }
    }
    tcx.hir()
        .krate()
        .visit_all_item_likes(&mut GatherCtors { tcx, set: &mut set }.as_deep_visitor());

    set
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
pub fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = std::any::type_name::<T>();
    if let Some(tail) = name.rfind(':') { Cow::from(&name[tail + 1..]) } else { Cow::from(name) }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass<'tcx> {
    fn name(&self) -> Cow<'_, str> {
        default_name::<Self>()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);
}

pub fn run_passes(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    mir_phase: MirPhase,
    passes: &[&[&dyn MirPass<'tcx>]],
) {
    let phase_index = mir_phase.phase_index();
    let validate = tcx.sess.opts.debugging_opts.validate_mir;

    if body.phase >= mir_phase {
        return;
    }

    if validate {
        validate::Validator { when: format!("input to phase {:?}", mir_phase), mir_phase }
            .run_pass(tcx, body);
    }

    let mut index = 0;
    let mut run_pass = |pass: &dyn MirPass<'tcx>| {
        let run_hooks = |body: &_, index, is_after| {
            dump_mir::on_mir_pass(
                tcx,
                &format_args!("{:03}-{:03}", phase_index, index),
                &pass.name(),
                body,
                is_after,
            );
        };
        run_hooks(body, index, false);
        pass.run_pass(tcx, body);
        run_hooks(body, index, true);

        if validate {
            validate::Validator {
                when: format!("after {} in phase {:?}", pass.name(), mir_phase),
                mir_phase,
            }
            .run_pass(tcx, body);
        }

        index += 1;
    };

    for pass_group in passes {
        for pass in *pass_group {
            run_pass(*pass);
        }
    }

    body.phase = mir_phase;

    if mir_phase == MirPhase::Optimization {
        validate::Validator { when: format!("end of phase {:?}", mir_phase), mir_phase }
            .run_pass(tcx, body);
    }
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

    let mut validator = check_consts::validation::Validator::new(&ccx);
    validator.check_body();

    // We return the qualifs in the return place for every MIR body, even though it is only used
    // when deciding to promote a reference to a `const` for now.
    validator.qualifs_in_return_place()
}

/// Make MIR ready for const evaluation. This is run on all MIR, not just on consts!
fn mir_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx Steal<Body<'tcx>> {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_const(def);
    }

    // Unsafety check uses the raw mir, so make sure it is run.
    if let Some(param_did) = def.const_param_did {
        tcx.ensure().unsafety_check_result_for_const_arg((def.did, param_did));
    } else {
        tcx.ensure().unsafety_check_result(def.did);
    }

    let mut body = tcx.mir_built(def).steal();

    util::dump_mir(tcx, None, "mir_map", &0, &body, |_, _| Ok(()));

    run_passes(
        tcx,
        &mut body,
        MirPhase::Const,
        &[&[
            // MIR-level lints.
            &check_packed_ref::CheckPackedRef,
            &check_const_item_mutation::CheckConstItemMutation,
            &function_item_references::FunctionItemReferences,
            // What we need to do constant evaluation.
            &simplify::SimplifyCfg::new("initial"),
            &rustc_peek::SanityCheck,
        ]],
    );
    tcx.alloc_steal_mir(body)
}

fn mir_promoted(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> (&'tcx Steal<Body<'tcx>>, &'tcx Steal<IndexVec<Promoted, Body<'tcx>>>) {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_promoted(def);
    }

    // Ensure that we compute the `mir_const_qualif` for constants at
    // this point, before we steal the mir-const result.
    // Also this means promotion can rely on all const checks having been done.
    let _ = tcx.mir_const_qualif_opt_const_arg(def);
    let _ = tcx.mir_abstract_const_opt_const_arg(def.to_global());
    let mut body = tcx.mir_const(def).steal();

    let mut required_consts = Vec::new();
    let mut required_consts_visitor = RequiredConstsVisitor::new(&mut required_consts);
    for (bb, bb_data) in traversal::reverse_postorder(&body) {
        required_consts_visitor.visit_basic_block_data(bb, bb_data);
    }
    body.required_consts = required_consts;

    let promote_pass = promote_consts::PromoteTemps::default();
    let promote: &[&dyn MirPass<'tcx>] = &[
        // What we need to run borrowck etc.
        &promote_pass,
        &simplify::SimplifyCfg::new("promote-consts"),
    ];

    let opt_coverage: &[&dyn MirPass<'tcx>] = if tcx.sess.opts.debugging_opts.instrument_coverage {
        &[&coverage::InstrumentCoverage]
    } else {
        &[]
    };

    run_passes(tcx, &mut body, MirPhase::ConstPromotion, &[promote, opt_coverage]);

    let promoted = promote_pass.promoted_fragments.into_inner();
    (tcx.alloc_steal_mir(body), tcx.alloc_steal_promoted(promoted))
}

fn mir_drops_elaborated_and_const_checked<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx Steal<Body<'tcx>> {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_drops_elaborated_and_const_checked(def);
    }

    // (Mir-)Borrowck uses `mir_promoted`, so we have to force it to
    // execute before we can steal.
    if let Some(param_did) = def.const_param_did {
        tcx.ensure().mir_borrowck_const_arg((def.did, param_did));
    } else {
        tcx.ensure().mir_borrowck(def.did);
    }

    let (body, _) = tcx.mir_promoted(def);
    let mut body = body.steal();

    run_post_borrowck_cleanup_passes(tcx, &mut body);
    check_consts::post_drop_elaboration::check_live_drops(tcx, &body);
    tcx.alloc_steal_mir(body)
}

/// After this series of passes, no lifetime analysis based on borrowing can be done.
fn run_post_borrowck_cleanup_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    debug!("post_borrowck_cleanup({:?})", body.source.def_id());

    let post_borrowck_cleanup: &[&dyn MirPass<'tcx>] = &[
        // Remove all things only needed by analysis
        &no_landing_pads::NoLandingPads::new(tcx),
        &simplify_branches::SimplifyBranches::new("initial"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &cleanup_post_borrowck::CleanupNonCodegenStatements,
        &simplify::SimplifyCfg::new("early-opt"),
        // These next passes must be executed together
        &add_call_guards::CriticalCallEdges,
        &elaborate_drops::ElaborateDrops,
        &no_landing_pads::NoLandingPads::new(tcx),
        // AddMovesForPackedDrops needs to run after drop
        // elaboration.
        &add_moves_for_packed_drops::AddMovesForPackedDrops,
        // `AddRetag` needs to run after `ElaborateDrops`. Otherwise it should run fairly late,
        // but before optimizations begin.
        &add_retag::AddRetag,
        &simplify::SimplifyCfg::new("elaborate-drops"),
        // `Deaggregator` is conceptually part of MIR building, some backends rely on it happening
        // and it can help optimizations.
        &deaggregator::Deaggregator,
    ];

    run_passes(tcx, body, MirPhase::DropLowering, &[post_borrowck_cleanup]);
}

fn run_optimization_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let mir_opt_level = tcx.sess.opts.debugging_opts.mir_opt_level;

    // Lowering generator control-flow and variables has to happen before we do anything else
    // to them. We run some optimizations before that, because they may be harder to do on the state
    // machine than on MIR with async primitives.
    let optimizations_with_generators: &[&dyn MirPass<'tcx>] = &[
        &unreachable_prop::UnreachablePropagation,
        &uninhabited_enum_branching::UninhabitedEnumBranching,
        &simplify::SimplifyCfg::new("after-uninhabited-enum-branching"),
        &inline::Inline,
        &generator::StateTransform,
    ];

    // Even if we don't do optimizations, we still have to lower generators for codegen.
    let no_optimizations_with_generators: &[&dyn MirPass<'tcx>] = &[&generator::StateTransform];

    // The main optimizations that we do on MIR.
    let optimizations: &[&dyn MirPass<'tcx>] = &[
        &lower_intrinsics::LowerIntrinsics,
        &remove_unneeded_drops::RemoveUnneededDrops,
        &match_branches::MatchBranchSimplification,
        // inst combine is after MatchBranchSimplification to clean up Ne(_1, false)
        &multiple_return_terminators::MultipleReturnTerminators,
        &instcombine::InstCombine,
        &const_prop::ConstProp,
        &simplify_branches::SimplifyBranches::new("after-const-prop"),
        &early_otherwise_branch::EarlyOtherwiseBranch,
        &simplify_comparison_integral::SimplifyComparisonIntegral,
        &simplify_try::SimplifyArmIdentity,
        &simplify_try::SimplifyBranchSame,
        &dest_prop::DestinationPropagation,
        &simplify_branches::SimplifyBranches::new("final"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &simplify::SimplifyCfg::new("final"),
        &nrvo::RenameReturnPlace,
        &simplify::SimplifyLocals,
        &multiple_return_terminators::MultipleReturnTerminators,
    ];

    // Optimizations to run even if mir optimizations have been disabled.
    let no_optimizations: &[&dyn MirPass<'tcx>] = &[
        // FIXME(#70073): This pass is responsible for both optimization as well as some lints.
        &const_prop::ConstProp,
    ];

    // Some cleanup necessary at least for LLVM and potentially other codegen backends.
    let pre_codegen_cleanup: &[&dyn MirPass<'tcx>] = &[
        &add_call_guards::CriticalCallEdges,
        // Dump the end result for testing and debugging purposes.
        &dump_mir::Marker("PreCodegen"),
    ];

    // End of pass declarations, now actually run the passes.
    // Generator Lowering
    #[rustfmt::skip]
    run_passes(
        tcx,
        body,
        MirPhase::GeneratorLowering,
        &[
            if mir_opt_level > 0 {
                optimizations_with_generators
            } else {
                no_optimizations_with_generators
            }
        ],
    );

    // Main optimization passes
    #[rustfmt::skip]
    run_passes(
        tcx,
        body,
        MirPhase::Optimization,
        &[
            if mir_opt_level > 0 { optimizations } else { no_optimizations },
            pre_codegen_cleanup,
        ],
    );
}

fn optimized_mir<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> &'tcx Body<'tcx> {
    let did = did.expect_local();
    if let Some(def) = ty::WithOptConstParam::try_lookup(did, tcx) {
        tcx.optimized_mir_of_const_arg(def)
    } else {
        tcx.arena.alloc(inner_optimized_mir(tcx, ty::WithOptConstParam::unknown(did)))
    }
}

fn optimized_mir_of_const_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    (did, param_did): (LocalDefId, DefId),
) -> &'tcx Body<'tcx> {
    tcx.arena.alloc(inner_optimized_mir(
        tcx,
        ty::WithOptConstParam { did, const_param_did: Some(param_did) },
    ))
}

fn inner_optimized_mir(tcx: TyCtxt<'_>, def: ty::WithOptConstParam<LocalDefId>) -> Body<'_> {
    if tcx.is_constructor(def.did.to_def_id()) {
        // There's no reason to run all of the MIR passes on constructors when
        // we can just output the MIR we want directly. This also saves const
        // qualification and borrow checking the trouble of special casing
        // constructors.
        return shim::build_adt_ctor(tcx, def.did.to_def_id());
    }

    let mut body = tcx.mir_drops_elaborated_and_const_checked(def).steal();
    run_optimization_passes(tcx, &mut body);

    debug_assert!(!body.has_free_regions(), "Free regions in optimized MIR");

    body
}

fn promoted_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx IndexVec<Promoted, Body<'tcx>> {
    if tcx.is_constructor(def.did.to_def_id()) {
        return tcx.arena.alloc(IndexVec::new());
    }

    if let Some(param_did) = def.const_param_did {
        tcx.ensure().mir_borrowck_const_arg((def.did, param_did));
    } else {
        tcx.ensure().mir_borrowck(def.did);
    }
    let (_, promoted) = tcx.mir_promoted(def);
    let mut promoted = promoted.steal();

    for body in &mut promoted {
        run_post_borrowck_cleanup_passes(tcx, body);
        run_optimization_passes(tcx, body);
    }

    debug_assert!(!promoted.has_free_regions(), "Free regions in promoted MIR");

    tcx.arena.alloc(promoted)
}
