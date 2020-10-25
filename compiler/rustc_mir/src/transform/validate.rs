//! Validates the MIR to ensure that invariants are upheld.

use crate::dataflow::impls::MaybeStorageLive;
use crate::dataflow::{Analysis, ResultsCursor};
use crate::util::storage::AlwaysLiveLocals;

use super::MirPass;
use rustc_middle::mir::{
    interpret::Scalar,
    visit::{PlaceContext, Visitor},
};
use rustc_middle::mir::{
    AggregateKind, BasicBlock, Body, BorrowKind, Local, Location, MirPhase, Operand, Rvalue,
    SourceScope, Statement, StatementKind, Terminator, TerminatorKind, VarDebugInfo,
};
use rustc_middle::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_target::abi::Size;

#[derive(Copy, Clone, Debug)]
enum EdgeKind {
    Unwind,
    Normal,
}

pub struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
    /// The phase for which we are upholding the dialect. If the given phase forbids a specific
    /// element, this validator will now emit errors if that specific element is encountered.
    /// Note that phases that change the dialect cause all *following* phases to check the
    /// invariants of the new dialect. A phase that changes dialects never checks the new invariants
    /// itself.
    pub mir_phase: MirPhase,
}

impl<'tcx> MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let param_env = tcx.param_env(def_id);
        let mir_phase = self.mir_phase;

        let always_live_locals = AlwaysLiveLocals::new(body);
        let storage_liveness = MaybeStorageLive::new(always_live_locals)
            .into_engine(tcx, body)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        TypeChecker { when: &self.when, body, tcx, param_env, mir_phase, storage_liveness }
            .visit_body(body);
    }
}

/// Returns whether the two types are equal up to lifetimes.
/// All lifetimes, including higher-ranked ones, get ignored for this comparison.
/// (This is unlike the `erasing_regions` methods, which keep higher-ranked lifetimes for soundness reasons.)
///
/// The point of this function is to approximate "equal up to subtyping".  However,
/// the approximation is incorrect as variance is ignored.
pub fn equal_up_to_regions(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    // Fast path.
    if src == dest {
        return true;
    }

    struct LifetimeIgnoreRelation<'tcx> {
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    }

    impl TypeRelation<'tcx> for LifetimeIgnoreRelation<'tcx> {
        fn tcx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn param_env(&self) -> ty::ParamEnv<'tcx> {
            self.param_env
        }

        fn tag(&self) -> &'static str {
            "librustc_mir::transform::validate"
        }

        fn a_is_expected(&self) -> bool {
            true
        }

        fn relate_with_variance<T: Relate<'tcx>>(
            &mut self,
            _: ty::Variance,
            a: T,
            b: T,
        ) -> RelateResult<'tcx, T> {
            // Ignore variance, require types to be exactly the same.
            self.relate(a, b)
        }

        fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
            if a == b {
                // Short-circuit.
                return Ok(a);
            }
            ty::relate::super_relate_tys(self, a, b)
        }

        fn regions(
            &mut self,
            a: ty::Region<'tcx>,
            _b: ty::Region<'tcx>,
        ) -> RelateResult<'tcx, ty::Region<'tcx>> {
            // Ignore regions.
            Ok(a)
        }

        fn consts(
            &mut self,
            a: &'tcx ty::Const<'tcx>,
            b: &'tcx ty::Const<'tcx>,
        ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
            ty::relate::super_relate_consts(self, a, b)
        }

        fn binders<T>(
            &mut self,
            a: ty::Binder<T>,
            b: ty::Binder<T>,
        ) -> RelateResult<'tcx, ty::Binder<T>>
        where
            T: Relate<'tcx>,
        {
            self.relate(a.skip_binder(), b.skip_binder())?;
            Ok(a)
        }
    }

    // Instantiate and run relation.
    let mut relator: LifetimeIgnoreRelation<'tcx> = LifetimeIgnoreRelation { tcx: tcx, param_env };
    relator.relate(src, dest).is_ok()
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    mir_phase: MirPhase,
    storage_liveness: ResultsCursor<'a, 'tcx, MaybeStorageLive>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        let span = self.body.source_info(location).span;
        // We use `delay_span_bug` as we might see broken MIR when other errors have already
        // occurred.
        self.tcx.sess.diagnostic().delay_span_bug(
            span,
            &format!(
                "broken MIR in {:?} ({}) at {:?}:\n{}",
                self.body.source.instance,
                self.when,
                location,
                msg.as_ref()
            ),
        );
    }

    fn check_edge(&self, location: Location, bb: BasicBlock, edge_kind: EdgeKind) {
        if let Some(bb) = self.body.basic_blocks().get(bb) {
            let src = self.body.basic_blocks().get(location.block).unwrap();
            match (src.is_cleanup, bb.is_cleanup, edge_kind) {
                // Non-cleanup blocks can jump to non-cleanup blocks along non-unwind edges
                (false, false, EdgeKind::Normal)
                // Non-cleanup blocks can jump to cleanup blocks along unwind edges
                | (false, true, EdgeKind::Unwind)
                // Cleanup blocks can jump to cleanup blocks along non-unwind edges
                | (true, true, EdgeKind::Normal) => {}
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
            self.fail(location, format!("encountered jump to invalid basic block {:?}", bb))
        }
    }

    /// Check if src can be assigned into dest.
    /// This is not precise, it will accept some incorrect assignments.
    fn mir_assign_valid_types(&self, src: Ty<'tcx>, dest: Ty<'tcx>) -> bool {
        // Fast path before we normalize.
        if src == dest {
            // Equal types, all is good.
            return true;
        }
        // Normalize projections and things like that.
        // FIXME: We need to reveal_all, as some optimizations change types in ways
        // that require unfolding opaque types.
        let param_env = self.param_env.with_reveal_all_normalized(self.tcx);
        let src = self.tcx.normalize_erasing_regions(param_env, src);
        let dest = self.tcx.normalize_erasing_regions(param_env, dest);

        // Type-changing assignments can happen when subtyping is used. While
        // all normal lifetimes are erased, higher-ranked types with their
        // late-bound lifetimes are still around and can lead to type
        // differences. So we compare ignoring lifetimes.
        equal_up_to_regions(self.tcx, param_env, src, dest)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, location: Location) {
        if context.is_use() {
            // Uses of locals must occur while the local's storage is allocated.
            self.storage_liveness.seek_after_primary_effect(location);
            let locals_with_storage = self.storage_liveness.get();
            if !locals_with_storage.contains(*local) {
                self.fail(location, format!("use of local {:?}, which has no storage here", local));
            }
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &VarDebugInfo<'tcx>) {
        // Debuginfo can contain field projections, which count as a use of the base local. Skip
        // debuginfo so that we avoid the storage liveness assertion in that case.
        self.visit_source_info(&var_debug_info.source_info);
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // `Operand::Copy` is only supposed to be used with `Copy` types.
        if let Operand::Copy(place) = operand {
            let ty = place.ty(&self.body.local_decls, self.tcx).ty;
            let span = self.body.source_info(location).span;

            if !ty.is_copy_modulo_regions(self.tcx.at(span), self.param_env) {
                self.fail(location, format!("`Operand::Copy` with non-`Copy` type {}", ty));
            }
        }

        self.super_operand(operand, location);
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
                match rvalue {
                    // The sides of an assignment must not alias. Currently this just checks whether the places
                    // are identical.
                    Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                        if dest == src {
                            self.fail(
                                location,
                                "encountered `Assign` statement with overlapping memory",
                            );
                        }
                    }
                    // The deaggregator currently does not deaggreagate arrays.
                    // So for now, we ignore them here.
                    Rvalue::Aggregate(box AggregateKind::Array { .. }, _) => {}
                    // All other aggregates must be gone after some phases.
                    Rvalue::Aggregate(box kind, _) => {
                        if self.mir_phase > MirPhase::DropLowering
                            && !matches!(kind, AggregateKind::Generator(..))
                        {
                            // Generators persist until the state machine transformation, but all
                            // other aggregates must have been lowered.
                            self.fail(
                                location,
                                format!("{:?} have been lowered to field assignments", rvalue),
                            )
                        } else if self.mir_phase > MirPhase::GeneratorLowering {
                            // No more aggregates after drop and generator lowering.
                            self.fail(
                                location,
                                format!("{:?} have been lowered to field assignments", rvalue),
                            )
                        }
                    }
                    Rvalue::Ref(_, BorrowKind::Shallow, _) => {
                        if self.mir_phase > MirPhase::DropLowering {
                            self.fail(
                                location,
                                "`Assign` statement with a `Shallow` borrow should have been removed after drop lowering phase",
                            );
                        }
                    }
                    _ => {}
                }
            }
            StatementKind::AscribeUserType(..) => {
                if self.mir_phase > MirPhase::DropLowering {
                    self.fail(
                        location,
                        "`AscribeUserType` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::FakeRead(..) => {
                if self.mir_phase > MirPhase::DropLowering {
                    self.fail(
                        location,
                        "`FakeRead` should have been removed after drop lowering phase",
                    );
                }
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                self.check_edge(location, *target, EdgeKind::Normal);
            }
            TerminatorKind::SwitchInt { targets, switch_ty, discr } => {
                let ty = discr.ty(&self.body.local_decls, self.tcx);
                if ty != *switch_ty {
                    self.fail(
                        location,
                        format!(
                            "encountered `SwitchInt` terminator with type mismatch: {:?} != {:?}",
                            ty, switch_ty,
                        ),
                    );
                }

                let target_width = self.tcx.sess.target.pointer_width;

                let size = Size::from_bits(match switch_ty.kind() {
                    ty::Uint(uint) => uint.normalize(target_width).bit_width().unwrap(),
                    ty::Int(int) => int.normalize(target_width).bit_width().unwrap(),
                    ty::Char => 32,
                    ty::Bool => 1,
                    other => bug!("unhandled type: {:?}", other),
                });

                for (value, target) in targets.iter() {
                    if Scalar::<()>::try_from_uint(value, size).is_none() {
                        self.fail(
                            location,
                            format!("the value {:#x} is not a proper {:?}", value, switch_ty),
                        )
                    }

                    self.check_edge(location, target, EdgeKind::Normal);
                }
                self.check_edge(location, targets.otherwise(), EdgeKind::Normal);
            }
            TerminatorKind::Drop { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::DropAndReplace { target, unwind, .. } => {
                if self.mir_phase > MirPhase::DropLowering {
                    self.fail(
                        location,
                        "`DropAndReplace` is not permitted to exist after drop elaboration",
                    );
                }
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Call { func, destination, cleanup, .. } => {
                let func_ty = func.ty(&self.body.local_decls, self.tcx);
                match func_ty.kind() {
                    ty::FnPtr(..) | ty::FnDef(..) => {}
                    _ => self.fail(
                        location,
                        format!("encountered non-callable type {} in `Call` terminator", func_ty),
                    ),
                }
                if let Some((_, target)) = destination {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Assert { cond, target, cleanup, .. } => {
                let cond_ty = cond.ty(&self.body.local_decls, self.tcx);
                if cond_ty != self.tcx.types.bool {
                    self.fail(
                        location,
                        format!(
                            "encountered non-boolean condition of type {} in `Assert` terminator",
                            cond_ty
                        ),
                    );
                }
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                if self.mir_phase > MirPhase::GeneratorLowering {
                    self.fail(location, "`Yield` should have been replaced by generator lowering");
                }
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::InlineAsm { destination, .. } => {
                if let Some(destination) = destination {
                    self.check_edge(location, *destination, EdgeKind::Normal);
                }
            }
            // Nothing to validate for these.
            TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop => {}
        }
    }

    fn visit_source_scope(&mut self, scope: &SourceScope) {
        if self.body.source_scopes.get(*scope).is_none() {
            self.tcx.sess.diagnostic().delay_span_bug(
                self.body.span,
                &format!(
                    "broken MIR in {:?} ({}):\ninvalid source scope {:?}",
                    self.body.source.instance, self.when, scope,
                ),
            );
        }
    }
}
