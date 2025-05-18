//! Validates the MIR to ensure that invariants are upheld.

use rustc_abi::{ExternAbi, FIRST_VARIANT, Size};
use rustc_attr_parsing::InlineAttr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::LangItem;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::mir::visit::{NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, CoroutineArgsExt, InstanceKind, ScalarInt, Ty, TyCtxt, TypeVisitableExt, Upcast, Variance,
};
use rustc_middle::{bug, span_bug};
use rustc_trait_selection::traits::ObligationCtxt;

use crate::util::{self, is_within_packed};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EdgeKind {
    Unwind,
    Normal,
}

pub(super) struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
}

impl<'tcx> crate::MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // FIXME(JakobDegen): These bodies never instantiated in codegend anyway, so it's not
        // terribly important that they pass the validator. However, I think other passes might
        // still see them, in which case they might be surprised. It would probably be better if we
        // didn't put this through the MIR pipeline at all.
        if matches!(body.source.instance, InstanceKind::Intrinsic(..) | InstanceKind::Virtual(..)) {
            return;
        }
        let def_id = body.source.def_id();
        let typing_env = body.typing_env(tcx);
        let can_unwind = if body.phase <= MirPhase::Runtime(RuntimePhase::Initial) {
            // In this case `AbortUnwindingCalls` haven't yet been executed.
            true
        } else if !tcx.def_kind(def_id).is_fn_like() {
            true
        } else {
            let body_ty = tcx.type_of(def_id).skip_binder();
            let body_abi = match body_ty.kind() {
                ty::FnDef(..) => body_ty.fn_sig(tcx).abi(),
                ty::Closure(..) => ExternAbi::RustCall,
                ty::CoroutineClosure(..) => ExternAbi::RustCall,
                ty::Coroutine(..) => ExternAbi::Rust,
                // No need to do MIR validation on error bodies
                ty::Error(_) => return,
                _ => span_bug!(body.span, "unexpected body ty: {body_ty}"),
            };

            ty::layout::fn_can_unwind(tcx, Some(def_id), body_abi)
        };

        let mut cfg_checker = CfgChecker {
            when: &self.when,
            body,
            tcx,
            unwind_edge_count: 0,
            reachable_blocks: traversal::reachable_as_bitset(body),
            value_cache: FxHashSet::default(),
            can_unwind,
        };
        cfg_checker.visit_body(body);
        cfg_checker.check_cleanup_control_flow();

        // Also run the TypeChecker.
        for (location, msg) in validate_types(tcx, typing_env, body, body) {
            cfg_checker.fail(location, msg);
        }

        if let MirPhase::Runtime(_) = body.phase {
            if let ty::InstanceKind::Item(_) = body.source.instance {
                if body.has_free_regions() {
                    cfg_checker.fail(
                        Location::START,
                        format!("Free regions in optimized {} MIR", body.phase.name()),
                    );
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// This checker covers basic properties of the control-flow graph, (dis)allowed statements and terminators.
/// Everything checked here must be stable under substitution of generic parameters. In other words,
/// this is about the *structure* of the MIR, not the *contents*.
///
/// Everything that depends on types, or otherwise can be affected by generic parameters,
/// must be checked in `TypeChecker`.
struct CfgChecker<'a, 'tcx> {
    when: &'a str,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    unwind_edge_count: usize,
    reachable_blocks: DenseBitSet<BasicBlock>,
    value_cache: FxHashSet<u128>,
    // If `false`, then the MIR must not contain `UnwindAction::Continue` or
    // `TerminatorKind::Resume`.
    can_unwind: bool,
}

impl<'a, 'tcx> CfgChecker<'a, 'tcx> {
    #[track_caller]
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        // We might see broken MIR when other errors have already occurred.
        assert!(
            self.tcx.dcx().has_errors().is_some(),
            "broken MIR in {:?} ({}) at {:?}:\n{}",
            self.body.source.instance,
            self.when,
            location,
            msg.as_ref(),
        );
    }

    fn check_edge(&mut self, location: Location, bb: BasicBlock, edge_kind: EdgeKind) {
        if bb == START_BLOCK {
            self.fail(location, "start block must not have predecessors")
        }
        if let Some(bb) = self.body.basic_blocks.get(bb) {
            let src = self.body.basic_blocks.get(location.block).unwrap();
            match (src.is_cleanup, bb.is_cleanup, edge_kind) {
                // Non-cleanup blocks can jump to non-cleanup blocks along non-unwind edges
                (false, false, EdgeKind::Normal)
                // Cleanup blocks can jump to cleanup blocks along non-unwind edges
                | (true, true, EdgeKind::Normal) => {}
                // Non-cleanup blocks can jump to cleanup blocks along unwind edges
                (false, true, EdgeKind::Unwind) => {
                    self.unwind_edge_count += 1;
                }
                // All other jumps are invalid
                _ => {
                    self.fail(
                        location,
                        format!(
                            "{:?} edge to {:?} violates unwind invariants (cleanup {:?} -> {:?})",
                            edge_kind,
                            bb,
                            src.is_cleanup,
                            bb.is_cleanup,
                        )
                    )
                }
            }
        } else {
            self.fail(location, format!("encountered jump to invalid basic block {bb:?}"))
        }
    }

    fn check_cleanup_control_flow(&self) {
        if self.unwind_edge_count <= 1 {
            return;
        }
        let doms = self.body.basic_blocks.dominators();
        let mut post_contract_node = FxHashMap::default();
        // Reusing the allocation across invocations of the closure
        let mut dom_path = vec![];
        let mut get_post_contract_node = |mut bb| {
            let root = loop {
                if let Some(root) = post_contract_node.get(&bb) {
                    break *root;
                }
                let parent = doms.immediate_dominator(bb).unwrap();
                dom_path.push(bb);
                if !self.body.basic_blocks[parent].is_cleanup {
                    break bb;
                }
                bb = parent;
            };
            for bb in dom_path.drain(..) {
                post_contract_node.insert(bb, root);
            }
            root
        };

        let mut parent = IndexVec::from_elem(None, &self.body.basic_blocks);
        for (bb, bb_data) in self.body.basic_blocks.iter_enumerated() {
            if !bb_data.is_cleanup || !self.reachable_blocks.contains(bb) {
                continue;
            }
            let bb = get_post_contract_node(bb);
            for s in bb_data.terminator().successors() {
                let s = get_post_contract_node(s);
                if s == bb {
                    continue;
                }
                let parent = &mut parent[bb];
                match parent {
                    None => {
                        *parent = Some(s);
                    }
                    Some(e) if *e == s => (),
                    Some(e) => self.fail(
                        Location { block: bb, statement_index: 0 },
                        format!(
                            "Cleanup control flow violation: The blocks dominated by {:?} have edges to both {:?} and {:?}",
                            bb,
                            s,
                            *e
                        )
                    ),
                }
            }
        }

        // Check for cycles
        let mut stack = FxHashSet::default();
        for (mut bb, parent) in parent.iter_enumerated_mut() {
            stack.clear();
            stack.insert(bb);
            loop {
                let Some(parent) = parent.take() else { break };
                let no_cycle = stack.insert(parent);
                if !no_cycle {
                    self.fail(
                        Location { block: bb, statement_index: 0 },
                        format!(
                            "Cleanup control flow violation: Cycle involving edge {bb:?} -> {parent:?}",
                        ),
                    );
                    break;
                }
                bb = parent;
            }
        }
    }

    fn check_unwind_edge(&mut self, location: Location, unwind: UnwindAction) {
        let is_cleanup = self.body.basic_blocks[location.block].is_cleanup;
        match unwind {
            UnwindAction::Cleanup(unwind) => {
                if is_cleanup {
                    self.fail(location, "`UnwindAction::Cleanup` in cleanup block");
                }
                self.check_edge(location, unwind, EdgeKind::Unwind);
            }
            UnwindAction::Continue => {
                if is_cleanup {
                    self.fail(location, "`UnwindAction::Continue` in cleanup block");
                }

                if !self.can_unwind {
                    self.fail(location, "`UnwindAction::Continue` in no-unwind function");
                }
            }
            UnwindAction::Terminate(UnwindTerminateReason::InCleanup) => {
                if !is_cleanup {
                    self.fail(
                        location,
                        "`UnwindAction::Terminate(InCleanup)` in a non-cleanup block",
                    );
                }
            }
            // These are allowed everywhere.
            UnwindAction::Unreachable | UnwindAction::Terminate(UnwindTerminateReason::Abi) => (),
        }
    }

    fn is_critical_call_edge(&self, target: Option<BasicBlock>, unwind: UnwindAction) -> bool {
        let Some(target) = target else { return false };
        matches!(unwind, UnwindAction::Cleanup(_) | UnwindAction::Terminate(_))
            && self.body.basic_blocks.predecessors()[target].len() > 1
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CfgChecker<'a, 'tcx> {
    fn visit_local(&mut self, local: Local, _context: PlaceContext, location: Location) {
        if self.body.local_decls.get(local).is_none() {
            self.fail(
                location,
                format!("local {local:?} has no corresponding declaration in `body.local_decls`"),
            );
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::AscribeUserType(..) => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`AscribeUserType` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::FakeRead(..) => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FakeRead` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::SetDiscriminant { .. } => {
                if self.body.phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`SetDiscriminant`is not allowed until deaggregation");
                }
            }
            StatementKind::Deinit(..) => {
                if self.body.phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`Deinit`is not allowed until deaggregation");
                }
            }
            StatementKind::Retag(kind, _) => {
                // FIXME(JakobDegen) The validator should check that `self.body.phase <
                // DropsLowered`. However, this causes ICEs with generation of drop shims, which
                // seem to fail to set their `MirPhase` correctly.
                if matches!(kind, RetagKind::TwoPhase) {
                    self.fail(location, format!("explicit `{kind:?}` is forbidden"));
                }
            }
            StatementKind::Coverage(kind) => {
                if self.body.phase >= MirPhase::Analysis(AnalysisPhase::PostCleanup)
                    && let CoverageKind::BlockMarker { .. } | CoverageKind::SpanMarker { .. } = kind
                {
                    self.fail(
                        location,
                        format!("{kind:?} should have been removed after analysis"),
                    );
                }
            }
            StatementKind::Assign(..)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Intrinsic(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => {}
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                self.check_edge(location, *target, EdgeKind::Normal);
            }
            TerminatorKind::SwitchInt { targets, discr: _ } => {
                for (_, target) in targets.iter() {
                    self.check_edge(location, target, EdgeKind::Normal);
                }
                self.check_edge(location, targets.otherwise(), EdgeKind::Normal);

                self.value_cache.clear();
                self.value_cache.extend(targets.iter().map(|(value, _)| value));
                let has_duplicates = targets.iter().len() != self.value_cache.len();
                if has_duplicates {
                    self.fail(
                        location,
                        format!(
                            "duplicated values in `SwitchInt` terminator: {:?}",
                            terminator.kind,
                        ),
                    );
                }
            }
            TerminatorKind::Drop { target, unwind, drop, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                self.check_unwind_edge(location, *unwind);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::Call { func, args, .. }
            | TerminatorKind::TailCall { func, args, .. } => {
                // FIXME(explicit_tail_calls): refactor this & add tail-call specific checks
                if let TerminatorKind::Call { target, unwind, destination, .. } = terminator.kind {
                    if let Some(target) = target {
                        self.check_edge(location, target, EdgeKind::Normal);
                    }
                    self.check_unwind_edge(location, unwind);

                    // The code generation assumes that there are no critical call edges. The
                    // assumption is used to simplify inserting code that should be executed along
                    // the return edge from the call. FIXME(tmiasko): Since this is a strictly code
                    // generation concern, the code generation should be responsible for handling
                    // it.
                    if self.body.phase >= MirPhase::Runtime(RuntimePhase::Optimized)
                        && self.is_critical_call_edge(target, unwind)
                    {
                        self.fail(
                            location,
                            format!(
                                "encountered critical edge in `Call` terminator {:?}",
                                terminator.kind,
                            ),
                        );
                    }

                    // The call destination place and Operand::Move place used as an argument might
                    // be passed by a reference to the callee. Consequently they cannot be packed.
                    if is_within_packed(self.tcx, &self.body.local_decls, destination).is_some() {
                        // This is bad! The callee will expect the memory to be aligned.
                        self.fail(
                            location,
                            format!(
                                "encountered packed place in `Call` terminator destination: {:?}",
                                terminator.kind,
                            ),
                        );
                    }
                }

                for arg in args {
                    if let Operand::Move(place) = &arg.node {
                        if is_within_packed(self.tcx, &self.body.local_decls, *place).is_some() {
                            // This is bad! The callee will expect the memory to be aligned.
                            self.fail(
                                location,
                                format!(
                                    "encountered `Move` of a packed place in `Call` terminator: {:?}",
                                    terminator.kind,
                                ),
                            );
                        }
                    }
                }

                if let ty::FnDef(did, ..) = func.ty(&self.body.local_decls, self.tcx).kind()
                    && self.body.phase >= MirPhase::Runtime(RuntimePhase::Optimized)
                    && matches!(self.tcx.codegen_fn_attrs(did).inline, InlineAttr::Force { .. })
                {
                    self.fail(location, "`#[rustc_force_inline]`-annotated function not inlined");
                }
            }
            TerminatorKind::Assert { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                if self.body.coroutine.is_none() {
                    self.fail(location, "`Yield` cannot appear outside coroutine bodies");
                }
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`Yield` should have been replaced by coroutine lowering");
                }
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FalseEdge` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FalseUnwind` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::InlineAsm { targets, unwind, .. } => {
                for &target in targets {
                    self.check_edge(location, target, EdgeKind::Normal);
                }
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::CoroutineDrop => {
                if self.body.coroutine.is_none() {
                    self.fail(location, "`CoroutineDrop` cannot appear outside coroutine bodies");
                }
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`CoroutineDrop` should have been replaced by coroutine lowering",
                    );
                }
            }
            TerminatorKind::UnwindResume => {
                let bb = location.block;
                if !self.body.basic_blocks[bb].is_cleanup {
                    self.fail(location, "Cannot `UnwindResume` from non-cleanup basic block")
                }
                if !self.can_unwind {
                    self.fail(location, "Cannot `UnwindResume` in a function that cannot unwind")
                }
            }
            TerminatorKind::UnwindTerminate(_) => {
                let bb = location.block;
                if !self.body.basic_blocks[bb].is_cleanup {
                    self.fail(location, "Cannot `UnwindTerminate` from non-cleanup basic block")
                }
            }
            TerminatorKind::Return => {
                let bb = location.block;
                if self.body.basic_blocks[bb].is_cleanup {
                    self.fail(location, "Cannot `Return` from cleanup basic block")
                }
            }
            TerminatorKind::Unreachable => {}
        }

        self.super_terminator(terminator, location);
    }

    fn visit_source_scope(&mut self, scope: SourceScope) {
        if self.body.source_scopes.get(scope).is_none() {
            self.tcx.dcx().span_bug(
                self.body.span,
                format!(
                    "broken MIR in {:?} ({}):\ninvalid source scope {:?}",
                    self.body.source.instance, self.when, scope,
                ),
            );
        }
    }
}

/// A faster version of the validation pass that only checks those things which may break when
/// instantiating any generic parameters.
///
/// `caller_body` is used to detect cycles in MIR inlining and MIR validation before
/// `optimized_mir` is available.
pub(super) fn validate_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &Body<'tcx>,
    caller_body: &Body<'tcx>,
) -> Vec<(Location, String)> {
    let mut type_checker = TypeChecker { body, caller_body, tcx, typing_env, failures: Vec::new() };
    // The type checker formats a bunch of strings with type names in it, but these strings
    // are not always going to be encountered on the error path since the inliner also uses
    // the validator, and there are certain kinds of inlining (even for valid code) that
    // can cause validation errors (mostly around where clauses and rigid projections).
    with_no_trimmed_paths!({
        type_checker.visit_body(body);
    });
    type_checker.failures
}

struct TypeChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    caller_body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    failures: Vec<(Location, String)>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn fail(&mut self, location: Location, msg: impl Into<String>) {
        self.failures.push((location, msg.into()));
    }

    /// Check if src can be assigned into dest.
    /// This is not precise, it will accept some incorrect assignments.
    fn mir_assign_valid_types(&self, src: Ty<'tcx>, dest: Ty<'tcx>) -> bool {
        // Fast path before we normalize.
        if src == dest {
            // Equal types, all is good.
            return true;
        }

        // We sometimes have to use `defining_opaque_types` for subtyping
        // to succeed here and figuring out how exactly that should work
        // is annoying. It is harmless enough to just not validate anything
        // in that case. We still check this after analysis as all opaque
        // types have been revealed at this point.
        if (src, dest).has_opaque_types() {
            return true;
        }

        // After borrowck subtyping should be fully explicit via
        // `Subtype` projections.
        let variance = if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
            Variance::Invariant
        } else {
            Variance::Covariant
        };

        crate::util::relate_types(self.tcx, self.typing_env, variance, src, dest)
    }

    /// Check that the given predicate definitely holds in the param-env of this MIR body.
    fn predicate_must_hold_modulo_regions(
        &self,
        pred: impl Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
    ) -> bool {
        let pred: ty::Predicate<'tcx> = pred.upcast(self.tcx);

        // We sometimes have to use `defining_opaque_types` for predicates
        // to succeed here and figuring out how exactly that should work
        // is annoying. It is harmless enough to just not validate anything
        // in that case. We still check this after analysis as all opaque
        // types have been revealed at this point.
        if pred.has_opaque_types() {
            return true;
        }

        let (infcx, param_env) = self.tcx.infer_ctxt().build_with_typing_env(self.typing_env);
        let ocx = ObligationCtxt::new(&infcx);
        ocx.register_obligation(Obligation::new(
            self.tcx,
            ObligationCause::dummy(),
            param_env,
            pred,
        ));
        ocx.select_all_or_error().is_empty()
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // This check is somewhat expensive, so only run it when -Zvalidate-mir is passed.
        if self.tcx.sess.opts.unstable_opts.validate_mir
            && self.body.phase < MirPhase::Runtime(RuntimePhase::Initial)
        {
            // `Operand::Copy` is only supposed to be used with `Copy` types.
            if let Operand::Copy(place) = operand {
                let ty = place.ty(&self.body.local_decls, self.tcx).ty;

                if !self.tcx.type_is_copy_modulo_regions(self.typing_env, ty) {
                    self.fail(location, format!("`Operand::Copy` with non-`Copy` type {ty}"));
                }
            }
        }

        self.super_operand(operand, location);
    }

    fn visit_projection_elem(
        &mut self,
        place_ref: PlaceRef<'tcx>,
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        match elem {
            ProjectionElem::OpaqueCast(ty)
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) =>
            {
                self.fail(
                    location,
                    format!("explicit opaque type cast to `{ty}` after `PostAnalysisNormalize`"),
                )
            }
            ProjectionElem::Index(index) => {
                let index_ty = self.body.local_decls[index].ty;
                if index_ty != self.tcx.types.usize {
                    self.fail(location, format!("bad index ({index_ty} != usize)"))
                }
            }
            ProjectionElem::Deref
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::PostCleanup) =>
            {
                let base_ty = place_ref.ty(&self.body.local_decls, self.tcx).ty;

                if base_ty.is_box() {
                    self.fail(location, format!("{base_ty} dereferenced after ElaborateBoxDerefs"))
                }
            }
            ProjectionElem::Field(f, ty) => {
                let parent_ty = place_ref.ty(&self.body.local_decls, self.tcx);
                let fail_out_of_bounds = |this: &mut Self, location| {
                    this.fail(location, format!("Out of bounds field {f:?} for {parent_ty:?}"));
                };
                let check_equal = |this: &mut Self, location, f_ty| {
                    if !this.mir_assign_valid_types(ty, f_ty) {
                        this.fail(
                            location,
                            format!(
                                "Field projection `{place_ref:?}.{f:?}` specified type `{ty}`, but actual type is `{f_ty}`"
                            )
                        )
                    }
                };

                let kind = match parent_ty.ty.kind() {
                    &ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
                        self.tcx.type_of(def_id).instantiate(self.tcx, args).kind()
                    }
                    kind => kind,
                };

                match kind {
                    ty::Tuple(fields) => {
                        let Some(f_ty) = fields.get(f.as_usize()) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, *f_ty);
                    }
                    ty::Adt(adt_def, args) => {
                        // see <https://github.com/rust-lang/rust/blob/7601adcc764d42c9f2984082b49948af652df986/compiler/rustc_middle/src/ty/layout.rs#L861-L864>
                        if self.tcx.is_lang_item(adt_def.did(), LangItem::DynMetadata) {
                            self.fail(
                                location,
                                format!(
                                    "You can't project to field {f:?} of `DynMetadata` because \
                                     layout is weird and thinks it doesn't have fields."
                                ),
                            );
                        }

                        let var = parent_ty.variant_index.unwrap_or(FIRST_VARIANT);
                        let Some(field) = adt_def.variant(var).fields.get(f) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, field.ty(self.tcx, args));
                    }
                    ty::Closure(_, args) => {
                        let args = args.as_closure();
                        let Some(&f_ty) = args.upvar_tys().get(f.as_usize()) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, f_ty);
                    }
                    ty::CoroutineClosure(_, args) => {
                        let args = args.as_coroutine_closure();
                        let Some(&f_ty) = args.upvar_tys().get(f.as_usize()) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, f_ty);
                    }
                    &ty::Coroutine(def_id, args) => {
                        let f_ty = if let Some(var) = parent_ty.variant_index {
                            // If we're currently validating an inlined copy of this body,
                            // then it will no longer be parameterized over the original
                            // args of the coroutine. Otherwise, we prefer to use this body
                            // since we may be in the process of computing this MIR in the
                            // first place.
                            let layout = if def_id == self.caller_body.source.def_id() {
                                self.caller_body
                                    .coroutine_layout_raw()
                                    .or_else(|| self.tcx.coroutine_layout(def_id, args).ok())
                            } else if self.tcx.needs_coroutine_by_move_body_def_id(def_id)
                                && let ty::ClosureKind::FnOnce =
                                    args.as_coroutine().kind_ty().to_opt_closure_kind().unwrap()
                                && self.caller_body.source.def_id()
                                    == self.tcx.coroutine_by_move_body_def_id(def_id)
                            {
                                // Same if this is the by-move body of a coroutine-closure.
                                self.caller_body.coroutine_layout_raw()
                            } else {
                                self.tcx.coroutine_layout(def_id, args).ok()
                            };

                            let Some(layout) = layout else {
                                self.fail(
                                    location,
                                    format!("No coroutine layout for {parent_ty:?}"),
                                );
                                return;
                            };

                            let Some(&local) = layout.variant_fields[var].get(f) else {
                                fail_out_of_bounds(self, location);
                                return;
                            };

                            let Some(f_ty) = layout.field_tys.get(local) else {
                                self.fail(
                                    location,
                                    format!("Out of bounds local {local:?} for {parent_ty:?}"),
                                );
                                return;
                            };

                            ty::EarlyBinder::bind(f_ty.ty).instantiate(self.tcx, args)
                        } else {
                            let Some(&f_ty) = args.as_coroutine().prefix_tys().get(f.index())
                            else {
                                fail_out_of_bounds(self, location);
                                return;
                            };

                            f_ty
                        };

                        check_equal(self, location, f_ty);
                    }
                    _ => {
                        self.fail(location, format!("{:?} does not have fields", parent_ty.ty));
                    }
                }
            }
            ProjectionElem::Subtype(ty) => {
                if !util::sub_types(
                    self.tcx,
                    self.typing_env,
                    ty,
                    place_ref.ty(&self.body.local_decls, self.tcx).ty,
                ) {
                    self.fail(
                        location,
                        format!(
                            "Failed subtyping {ty} and {}",
                            place_ref.ty(&self.body.local_decls, self.tcx).ty
                        ),
                    )
                }
            }
            ProjectionElem::UnwrapUnsafeBinder(unwrapped_ty) => {
                let binder_ty = place_ref.ty(&self.body.local_decls, self.tcx);
                let ty::UnsafeBinder(binder_ty) = *binder_ty.ty.kind() else {
                    self.fail(
                        location,
                        format!("WrapUnsafeBinder does not produce a ty::UnsafeBinder"),
                    );
                    return;
                };
                let binder_inner_ty = self.tcx.instantiate_bound_regions_with_erased(*binder_ty);
                if !self.mir_assign_valid_types(unwrapped_ty, binder_inner_ty) {
                    self.fail(
                        location,
                        format!(
                            "Cannot unwrap unsafe binder {binder_ty:?} into type {unwrapped_ty}"
                        ),
                    );
                }
            }
            _ => {}
        }
        self.super_projection_elem(place_ref, elem, context, location);
    }

    fn visit_var_debug_info(&mut self, debuginfo: &VarDebugInfo<'tcx>) {
        if let Some(box VarDebugInfoFragment { ty, ref projection }) = debuginfo.composite {
            if ty.is_union() || ty.is_enum() {
                self.fail(
                    START_BLOCK.start_location(),
                    format!("invalid type {ty} in debuginfo for {:?}", debuginfo.name),
                );
            }
            if projection.is_empty() {
                self.fail(
                    START_BLOCK.start_location(),
                    format!("invalid empty projection in debuginfo for {:?}", debuginfo.name),
                );
            }
            if projection.iter().any(|p| !matches!(p, PlaceElem::Field(..))) {
                self.fail(
                    START_BLOCK.start_location(),
                    format!(
                        "illegal projection {:?} in debuginfo for {:?}",
                        projection, debuginfo.name
                    ),
                );
            }
        }
        match debuginfo.value {
            VarDebugInfoContents::Const(_) => {}
            VarDebugInfoContents::Place(place) => {
                if place.projection.iter().any(|p| !p.can_use_in_debuginfo()) {
                    self.fail(
                        START_BLOCK.start_location(),
                        format!("illegal place {:?} in debuginfo for {:?}", place, debuginfo.name),
                    );
                }
            }
        }
        self.super_var_debug_info(debuginfo);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, cntxt: PlaceContext, location: Location) {
        // Set off any `bug!`s in the type computation code
        let _ = place.ty(&self.body.local_decls, self.tcx);

        if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial)
            && place.projection.len() > 1
            && cntxt != PlaceContext::NonUse(NonUseContext::VarDebugInfo)
            && place.projection[1..].contains(&ProjectionElem::Deref)
        {
            self.fail(
                location,
                format!("place {place:?} has deref as a later projection (it is only permitted as the first projection)"),
            );
        }

        // Ensure all downcast projections are followed by field projections.
        let mut projections_iter = place.projection.iter();
        while let Some(proj) = projections_iter.next() {
            if matches!(proj, ProjectionElem::Downcast(..)) {
                if !matches!(projections_iter.next(), Some(ProjectionElem::Field(..))) {
                    self.fail(
                        location,
                        format!(
                            "place {place:?} has `Downcast` projection not followed by `Field`"
                        ),
                    );
                }
            }
        }

        self.super_place(place, cntxt, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        macro_rules! check_kinds {
            ($t:expr, $text:literal, $typat:pat) => {
                if !matches!(($t).kind(), $typat) {
                    self.fail(location, format!($text, $t));
                }
            };
        }
        match rvalue {
            Rvalue::Use(_) | Rvalue::CopyForDeref(_) => {}
            Rvalue::Aggregate(kind, fields) => match **kind {
                AggregateKind::Tuple => {}
                AggregateKind::Array(dest) => {
                    for src in fields {
                        if !self.mir_assign_valid_types(src.ty(self.body, self.tcx), dest) {
                            self.fail(location, "array field has the wrong type");
                        }
                    }
                }
                AggregateKind::Adt(def_id, idx, args, _, Some(field)) => {
                    let adt_def = self.tcx.adt_def(def_id);
                    assert!(adt_def.is_union());
                    assert_eq!(idx, FIRST_VARIANT);
                    let dest_ty = self.tcx.normalize_erasing_regions(
                        self.typing_env,
                        adt_def.non_enum_variant().fields[field].ty(self.tcx, args),
                    );
                    if let [field] = fields.raw.as_slice() {
                        let src_ty = field.ty(self.body, self.tcx);
                        if !self.mir_assign_valid_types(src_ty, dest_ty) {
                            self.fail(location, "union field has the wrong type");
                        }
                    } else {
                        self.fail(location, "unions should have one initialized field");
                    }
                }
                AggregateKind::Adt(def_id, idx, args, _, None) => {
                    let adt_def = self.tcx.adt_def(def_id);
                    assert!(!adt_def.is_union());
                    let variant = &adt_def.variants()[idx];
                    if variant.fields.len() != fields.len() {
                        self.fail(location, "adt has the wrong number of initialized fields");
                    }
                    for (src, dest) in std::iter::zip(fields, &variant.fields) {
                        let dest_ty = self
                            .tcx
                            .normalize_erasing_regions(self.typing_env, dest.ty(self.tcx, args));
                        if !self.mir_assign_valid_types(src.ty(self.body, self.tcx), dest_ty) {
                            self.fail(location, "adt field has the wrong type");
                        }
                    }
                }
                AggregateKind::Closure(_, args) => {
                    let upvars = args.as_closure().upvar_tys();
                    if upvars.len() != fields.len() {
                        self.fail(location, "closure has the wrong number of initialized fields");
                    }
                    for (src, dest) in std::iter::zip(fields, upvars) {
                        if !self.mir_assign_valid_types(src.ty(self.body, self.tcx), dest) {
                            self.fail(location, "closure field has the wrong type");
                        }
                    }
                }
                AggregateKind::Coroutine(_, args) => {
                    let upvars = args.as_coroutine().upvar_tys();
                    if upvars.len() != fields.len() {
                        self.fail(location, "coroutine has the wrong number of initialized fields");
                    }
                    for (src, dest) in std::iter::zip(fields, upvars) {
                        if !self.mir_assign_valid_types(src.ty(self.body, self.tcx), dest) {
                            self.fail(location, "coroutine field has the wrong type");
                        }
                    }
                }
                AggregateKind::CoroutineClosure(_, args) => {
                    let upvars = args.as_coroutine_closure().upvar_tys();
                    if upvars.len() != fields.len() {
                        self.fail(
                            location,
                            "coroutine-closure has the wrong number of initialized fields",
                        );
                    }
                    for (src, dest) in std::iter::zip(fields, upvars) {
                        if !self.mir_assign_valid_types(src.ty(self.body, self.tcx), dest) {
                            self.fail(location, "coroutine-closure field has the wrong type");
                        }
                    }
                }
                AggregateKind::RawPtr(pointee_ty, mutability) => {
                    if !matches!(self.body.phase, MirPhase::Runtime(_)) {
                        // It would probably be fine to support this in earlier phases, but at the
                        // time of writing it's only ever introduced from intrinsic lowering, so
                        // earlier things just `bug!` on it.
                        self.fail(location, "RawPtr should be in runtime MIR only");
                    }

                    if let [data_ptr, metadata] = fields.raw.as_slice() {
                        let data_ptr_ty = data_ptr.ty(self.body, self.tcx);
                        let metadata_ty = metadata.ty(self.body, self.tcx);
                        if let ty::RawPtr(in_pointee, in_mut) = data_ptr_ty.kind() {
                            if *in_mut != mutability {
                                self.fail(location, "input and output mutability must match");
                            }

                            // FIXME: check `Thin` instead of `Sized`
                            if !in_pointee.is_sized(self.tcx, self.typing_env) {
                                self.fail(location, "input pointer must be thin");
                            }
                        } else {
                            self.fail(
                                location,
                                "first operand to raw pointer aggregate must be a raw pointer",
                            );
                        }

                        // FIXME: Check metadata more generally
                        if pointee_ty.is_slice() {
                            if !self.mir_assign_valid_types(metadata_ty, self.tcx.types.usize) {
                                self.fail(location, "slice metadata must be usize");
                            }
                        } else if pointee_ty.is_sized(self.tcx, self.typing_env) {
                            if metadata_ty != self.tcx.types.unit {
                                self.fail(location, "metadata for pointer-to-thin must be unit");
                            }
                        }
                    } else {
                        self.fail(location, "raw pointer aggregate must have 2 fields");
                    }
                }
            },
            Rvalue::Ref(_, BorrowKind::Fake(_), _) => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`Assign` statement with a `Fake` borrow should have been removed in runtime MIR",
                    );
                }
            }
            Rvalue::Ref(..) => {}
            Rvalue::Len(p) => {
                let pty = p.ty(&self.body.local_decls, self.tcx).ty;
                check_kinds!(
                    pty,
                    "Cannot compute length of non-array type {:?}",
                    ty::Array(..) | ty::Slice(..)
                );
            }
            Rvalue::BinaryOp(op, vals) => {
                use BinOp::*;
                let a = vals.0.ty(&self.body.local_decls, self.tcx);
                let b = vals.1.ty(&self.body.local_decls, self.tcx);
                if crate::util::binop_right_homogeneous(*op) {
                    if let Eq | Lt | Le | Ne | Ge | Gt = op {
                        // The function pointer types can have lifetimes
                        if !self.mir_assign_valid_types(a, b) {
                            self.fail(
                                location,
                                format!("Cannot {op:?} compare incompatible types {a} and {b}"),
                            );
                        }
                    } else if a != b {
                        self.fail(
                            location,
                            format!("Cannot perform binary op {op:?} on unequal types {a} and {b}"),
                        );
                    }
                }

                match op {
                    Offset => {
                        check_kinds!(a, "Cannot offset non-pointer type {:?}", ty::RawPtr(..));
                        if b != self.tcx.types.isize && b != self.tcx.types.usize {
                            self.fail(location, format!("Cannot offset by non-isize type {b}"));
                        }
                    }
                    Eq | Lt | Le | Ne | Ge | Gt => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot {op:?} compare type {:?}",
                                ty::Bool
                                    | ty::Char
                                    | ty::Int(..)
                                    | ty::Uint(..)
                                    | ty::Float(..)
                                    | ty::RawPtr(..)
                                    | ty::FnPtr(..)
                            )
                        }
                    }
                    Cmp => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot three-way compare non-integer type {:?}",
                                ty::Char | ty::Uint(..) | ty::Int(..)
                            )
                        }
                    }
                    AddUnchecked | AddWithOverflow | SubUnchecked | SubWithOverflow
                    | MulUnchecked | MulWithOverflow | Shl | ShlUnchecked | Shr | ShrUnchecked => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot {op:?} non-integer type {:?}",
                                ty::Uint(..) | ty::Int(..)
                            )
                        }
                    }
                    BitAnd | BitOr | BitXor => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform bitwise op {op:?} on type {:?}",
                                ty::Uint(..) | ty::Int(..) | ty::Bool
                            )
                        }
                    }
                    Add | Sub | Mul | Div | Rem => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform arithmetic {op:?} on type {:?}",
                                ty::Uint(..) | ty::Int(..) | ty::Float(..)
                            )
                        }
                    }
                }
            }
            Rvalue::UnaryOp(op, operand) => {
                let a = operand.ty(&self.body.local_decls, self.tcx);
                match op {
                    UnOp::Neg => {
                        check_kinds!(a, "Cannot negate type {:?}", ty::Int(..) | ty::Float(..))
                    }
                    UnOp::Not => {
                        check_kinds!(
                            a,
                            "Cannot binary not type {:?}",
                            ty::Int(..) | ty::Uint(..) | ty::Bool
                        );
                    }
                    UnOp::PtrMetadata => {
                        check_kinds!(
                            a,
                            "Cannot PtrMetadata non-pointer non-reference type {:?}",
                            ty::RawPtr(..) | ty::Ref(..)
                        );
                    }
                }
            }
            Rvalue::ShallowInitBox(operand, _) => {
                let a = operand.ty(&self.body.local_decls, self.tcx);
                check_kinds!(a, "Cannot shallow init type {:?}", ty::RawPtr(..));
            }
            Rvalue::Cast(kind, operand, target_type) => {
                let op_ty = operand.ty(self.body, self.tcx);
                match kind {
                    // FIXME: Add Checks for these
                    CastKind::PointerWithExposedProvenance | CastKind::PointerExposeProvenance => {}
                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _) => {
                        // FIXME: check signature compatibility.
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a fn item, not {:?}",
                            ty::FnDef(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a fn pointer, not {:?}",
                            ty::FnPtr(..)
                        );
                    }
                    CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer, _) => {
                        // FIXME: check safety and signature compatibility.
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a fn pointer, not {:?}",
                            ty::FnPtr(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a fn pointer, not {:?}",
                            ty::FnPtr(..)
                        );
                    }
                    CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(..), _) => {
                        // FIXME: check safety, captures, and signature compatibility.
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a closure, not {:?}",
                            ty::Closure(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a fn pointer, not {:?}",
                            ty::FnPtr(..)
                        );
                    }
                    CastKind::PointerCoercion(PointerCoercion::MutToConstPointer, _) => {
                        // FIXME: check same pointee?
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a raw mut pointer, not {:?}",
                            ty::RawPtr(_, Mutability::Mut)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a raw const pointer, not {:?}",
                            ty::RawPtr(_, Mutability::Not)
                        );
                        if self.body.phase >= MirPhase::Analysis(AnalysisPhase::PostCleanup) {
                            self.fail(location, format!("After borrowck, MIR disallows {kind:?}"));
                        }
                    }
                    CastKind::PointerCoercion(PointerCoercion::ArrayToPointer, _) => {
                        // FIXME: Check pointee types
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a raw pointer, not {:?}",
                            ty::RawPtr(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a raw pointer, not {:?}",
                            ty::RawPtr(..)
                        );
                        if self.body.phase >= MirPhase::Analysis(AnalysisPhase::PostCleanup) {
                            self.fail(location, format!("After borrowck, MIR disallows {kind:?}"));
                        }
                    }
                    CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                        // Pointers being unsize coerced should at least implement
                        // `CoerceUnsized`.
                        if !self.predicate_must_hold_modulo_regions(ty::TraitRef::new(
                            self.tcx,
                            self.tcx.require_lang_item(
                                LangItem::CoerceUnsized,
                                Some(self.body.source_info(location).span),
                            ),
                            [op_ty, *target_type],
                        )) {
                            self.fail(location, format!("Unsize coercion, but `{op_ty}` isn't coercible to `{target_type}`"));
                        }
                    }
                    CastKind::PointerCoercion(PointerCoercion::DynStar, _) => {
                        // FIXME(dyn-star): make sure nothing needs to be done here.
                    }
                    CastKind::IntToInt | CastKind::IntToFloat => {
                        let input_valid = op_ty.is_integral() || op_ty.is_char() || op_ty.is_bool();
                        let target_valid = target_type.is_numeric() || target_type.is_char();
                        if !input_valid || !target_valid {
                            self.fail(
                                location,
                                format!("Wrong cast kind {kind:?} for the type {op_ty}"),
                            );
                        }
                    }
                    CastKind::FnPtrToPtr => {
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a fn pointer, not {:?}",
                            ty::FnPtr(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a raw pointer, not {:?}",
                            ty::RawPtr(..)
                        );
                    }
                    CastKind::PtrToPtr => {
                        check_kinds!(
                            op_ty,
                            "CastKind::{kind:?} input must be a raw pointer, not {:?}",
                            ty::RawPtr(..)
                        );
                        check_kinds!(
                            target_type,
                            "CastKind::{kind:?} output must be a raw pointer, not {:?}",
                            ty::RawPtr(..)
                        );
                    }
                    CastKind::FloatToFloat | CastKind::FloatToInt => {
                        if !op_ty.is_floating_point() || !target_type.is_numeric() {
                            self.fail(
                                location,
                                format!(
                                    "Trying to cast non 'Float' as {kind:?} into {target_type:?}"
                                ),
                            );
                        }
                    }
                    CastKind::Transmute => {
                        if let MirPhase::Runtime(..) = self.body.phase {
                            // Unlike `mem::transmute`, a MIR `Transmute` is well-formed
                            // for any two `Sized` types, just potentially UB to run.

                            if !self
                                .tcx
                                .normalize_erasing_regions(self.typing_env, op_ty)
                                .is_sized(self.tcx, self.typing_env)
                            {
                                self.fail(
                                    location,
                                    format!("Cannot transmute from non-`Sized` type {op_ty}"),
                                );
                            }
                            if !self
                                .tcx
                                .normalize_erasing_regions(self.typing_env, *target_type)
                                .is_sized(self.tcx, self.typing_env)
                            {
                                self.fail(
                                    location,
                                    format!("Cannot transmute to non-`Sized` type {target_type:?}"),
                                );
                            }
                        } else {
                            self.fail(
                                location,
                                format!(
                                    "Transmute is not supported in non-runtime phase {:?}.",
                                    self.body.phase
                                ),
                            );
                        }
                    }
                }
            }
            Rvalue::NullaryOp(NullOp::OffsetOf(indices), container) => {
                let fail_out_of_bounds = |this: &mut Self, location, field, ty| {
                    this.fail(location, format!("Out of bounds field {field:?} for {ty}"));
                };

                let mut current_ty = *container;

                for (variant, field) in indices.iter() {
                    match current_ty.kind() {
                        ty::Tuple(fields) => {
                            if variant != FIRST_VARIANT {
                                self.fail(
                                    location,
                                    format!("tried to get variant {variant:?} of tuple"),
                                );
                                return;
                            }
                            let Some(&f_ty) = fields.get(field.as_usize()) else {
                                fail_out_of_bounds(self, location, field, current_ty);
                                return;
                            };

                            current_ty = self.tcx.normalize_erasing_regions(self.typing_env, f_ty);
                        }
                        ty::Adt(adt_def, args) => {
                            let Some(field) = adt_def.variant(variant).fields.get(field) else {
                                fail_out_of_bounds(self, location, field, current_ty);
                                return;
                            };

                            let f_ty = field.ty(self.tcx, args);
                            current_ty = self.tcx.normalize_erasing_regions(self.typing_env, f_ty);
                        }
                        _ => {
                            self.fail(
                                location,
                                format!("Cannot get offset ({variant:?}, {field:?}) from type {current_ty}"),
                            );
                            return;
                        }
                    }
                }
            }
            Rvalue::Repeat(_, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::RawPtr(_, _)
            | Rvalue::NullaryOp(
                NullOp::SizeOf | NullOp::AlignOf | NullOp::UbChecks | NullOp::ContractChecks,
                _,
            )
            | Rvalue::Discriminant(_) => {}

            Rvalue::WrapUnsafeBinder(op, ty) => {
                let unwrapped_ty = op.ty(self.body, self.tcx);
                let ty::UnsafeBinder(binder_ty) = *ty.kind() else {
                    self.fail(
                        location,
                        format!("WrapUnsafeBinder does not produce a ty::UnsafeBinder"),
                    );
                    return;
                };
                let binder_inner_ty = self.tcx.instantiate_bound_regions_with_erased(*binder_ty);
                if !self.mir_assign_valid_types(unwrapped_ty, binder_inner_ty) {
                    self.fail(
                        location,
                        format!("Cannot wrap {unwrapped_ty} into unsafe binder {binder_ty:?}"),
                    );
                }
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(box (dest, rvalue)) => {
                // LHS and RHS of the assignment must have the same type.
                let left_ty = dest.ty(&self.body.local_decls, self.tcx).ty;
                let right_ty = rvalue.ty(&self.body.local_decls, self.tcx);

                if !self.mir_assign_valid_types(right_ty, left_ty) {
                    self.fail(
                        location,
                        format!(
                            "encountered `{:?}` with incompatible types:\n\
                            left-hand side has type: {}\n\
                            right-hand side has type: {}",
                            statement.kind, left_ty, right_ty,
                        ),
                    );
                }
                if let Rvalue::CopyForDeref(place) = rvalue {
                    if place.ty(&self.body.local_decls, self.tcx).ty.builtin_deref(true).is_none() {
                        self.fail(
                            location,
                            "`CopyForDeref` should only be used for dereferenceable types",
                        )
                    }
                }
            }
            StatementKind::AscribeUserType(..) => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`AscribeUserType` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::FakeRead(..) => {
                if self.body.phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FakeRead` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(op)) => {
                let ty = op.ty(&self.body.local_decls, self.tcx);
                if !ty.is_bool() {
                    self.fail(
                        location,
                        format!("`assume` argument must be `bool`, but got: `{ty}`"),
                    );
                }
            }
            StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(
                CopyNonOverlapping { src, dst, count },
            )) => {
                let src_ty = src.ty(&self.body.local_decls, self.tcx);
                let op_src_ty = if let Some(src_deref) = src_ty.builtin_deref(true) {
                    src_deref
                } else {
                    self.fail(
                        location,
                        format!("Expected src to be ptr in copy_nonoverlapping, got: {src_ty}"),
                    );
                    return;
                };
                let dst_ty = dst.ty(&self.body.local_decls, self.tcx);
                let op_dst_ty = if let Some(dst_deref) = dst_ty.builtin_deref(true) {
                    dst_deref
                } else {
                    self.fail(
                        location,
                        format!("Expected dst to be ptr in copy_nonoverlapping, got: {dst_ty}"),
                    );
                    return;
                };
                // since CopyNonOverlapping is parametrized by 1 type,
                // we only need to check that they are equal and not keep an extra parameter.
                if !self.mir_assign_valid_types(op_src_ty, op_dst_ty) {
                    self.fail(location, format!("bad arg ({op_src_ty} != {op_dst_ty})"));
                }

                let op_cnt_ty = count.ty(&self.body.local_decls, self.tcx);
                if op_cnt_ty != self.tcx.types.usize {
                    self.fail(location, format!("bad arg ({op_cnt_ty} != usize)"))
                }
            }
            StatementKind::SetDiscriminant { place, .. } => {
                if self.body.phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`SetDiscriminant`is not allowed until deaggregation");
                }
                let pty = place.ty(&self.body.local_decls, self.tcx).ty;
                if !matches!(
                    pty.kind(),
                    ty::Adt(..) | ty::Coroutine(..) | ty::Alias(ty::Opaque, ..)
                ) {
                    self.fail(
                        location,
                        format!(
                            "`SetDiscriminant` is only allowed on ADTs and coroutines, not {pty}"
                        ),
                    );
                }
            }
            StatementKind::Deinit(..) => {
                if self.body.phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`Deinit`is not allowed until deaggregation");
                }
            }
            StatementKind::Retag(kind, _) => {
                // FIXME(JakobDegen) The validator should check that `self.body.phase <
                // DropsLowered`. However, this causes ICEs with generation of drop shims, which
                // seem to fail to set their `MirPhase` correctly.
                if matches!(kind, RetagKind::TwoPhase) {
                    self.fail(location, format!("explicit `{kind:?}` is forbidden"));
                }
            }
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Coverage(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => {}
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::SwitchInt { targets, discr } => {
                let switch_ty = discr.ty(&self.body.local_decls, self.tcx);

                let target_width = self.tcx.sess.target.pointer_width;

                let size = Size::from_bits(match switch_ty.kind() {
                    ty::Uint(uint) => uint.normalize(target_width).bit_width().unwrap(),
                    ty::Int(int) => int.normalize(target_width).bit_width().unwrap(),
                    ty::Char => 32,
                    ty::Bool => 1,
                    other => bug!("unhandled type: {:?}", other),
                });

                for (value, _) in targets.iter() {
                    if ScalarInt::try_from_uint(value, size).is_none() {
                        self.fail(
                            location,
                            format!("the value {value:#x} is not a proper {switch_ty}"),
                        )
                    }
                }
            }
            TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } => {
                let func_ty = func.ty(&self.body.local_decls, self.tcx);
                match func_ty.kind() {
                    ty::FnPtr(..) | ty::FnDef(..) => {}
                    _ => self.fail(
                        location,
                        format!(
                            "encountered non-callable type {func_ty} in `{}` terminator",
                            terminator.kind.name()
                        ),
                    ),
                }

                if let TerminatorKind::TailCall { .. } = terminator.kind {
                    // FIXME(explicit_tail_calls): implement tail-call specific checks here (such
                    // as signature matching, forbidding closures, etc)
                }
            }
            TerminatorKind::Assert { cond, .. } => {
                let cond_ty = cond.ty(&self.body.local_decls, self.tcx);
                if cond_ty != self.tcx.types.bool {
                    self.fail(
                        location,
                        format!(
                            "encountered non-boolean condition of type {cond_ty} in `Assert` terminator"
                        ),
                    );
                }
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable => {}
        }

        self.super_terminator(terminator, location);
    }
}
