//! Validates the MIR to ensure that invariants are upheld.

use super::MirPass;
use rustc_index::bit_set::BitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::traversal;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    AggregateKind, BasicBlock, Body, BorrowKind, Local, Location, MirPhase, Operand, PlaceElem,
    PlaceRef, ProjectionElem, Rvalue, SourceScope, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt, TypeFoldable};
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::storage::AlwaysLiveLocals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
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

        TypeChecker {
            when: &self.when,
            body,
            tcx,
            param_env,
            mir_phase,
            reachable_blocks: traversal::reachable_as_bitset(body),
            storage_liveness,
            place_cache: Vec::new(),
        }
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

    // Normalize lifetimes away on both sides, then compare.
    let param_env = param_env.with_reveal_all_normalized(tcx);
    let normalize = |ty: Ty<'tcx>| {
        tcx.normalize_erasing_regions(
            param_env,
            ty.fold_with(&mut BottomUpFolder {
                tcx,
                // FIXME: We erase all late-bound lifetimes, but this is not fully correct.
                // If you have a type like `<for<'a> fn(&'a u32) as SomeTrait>::Assoc`,
                // this is not necessarily equivalent to `<fn(&'static u32) as SomeTrait>::Assoc`,
                // since one may have an `impl SomeTrait for fn(&32)` and
                // `impl SomeTrait for fn(&'static u32)` at the same time which
                // specify distinct values for Assoc. (See also #56105)
                lt_op: |_| tcx.lifetimes.re_erased,
                // Leave consts and types unchanged.
                ct_op: |ct| ct,
                ty_op: |ty| ty,
            }),
        )
    };
    tcx.infer_ctxt().enter(|infcx| infcx.can_eq(param_env, normalize(src), normalize(dest)).is_ok())
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    mir_phase: MirPhase,
    reachable_blocks: BitSet<BasicBlock>,
    storage_liveness: ResultsCursor<'a, 'tcx, MaybeStorageLive>,
    place_cache: Vec<PlaceRef<'tcx>>,
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
        if self.body.local_decls.get(*local).is_none() {
            self.fail(
                location,
                format!("local {:?} has no corresponding declaration in `body.local_decls`", local),
            );
        }

        if self.reachable_blocks.contains(location.block) && context.is_use() {
            // Uses of locals must occur while the local's storage is allocated.
            self.storage_liveness.seek_after_primary_effect(location);
            let locals_with_storage = self.storage_liveness.get();
            if !locals_with_storage.contains(*local) {
                self.fail(location, format!("use of local {:?}, which has no storage here", local));
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // This check is somewhat expensive, so only run it when -Zvalidate-mir is passed.
        if self.tcx.sess.opts.debugging_opts.validate_mir {
            // `Operand::Copy` is only supposed to be used with `Copy` types.
            if let Operand::Copy(place) = operand {
                let ty = place.ty(&self.body.local_decls, self.tcx).ty;
                let span = self.body.source_info(location).span;

                if !ty.is_copy_modulo_regions(self.tcx.at(span), self.param_env) {
                    self.fail(location, format!("`Operand::Copy` with non-`Copy` type {}", ty));
                }
            }
        }

        self.super_operand(operand, location);
    }

    fn visit_projection_elem(
        &mut self,
        local: Local,
        proj_base: &[PlaceElem<'tcx>],
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        if let ProjectionElem::Index(index) = elem {
            let index_ty = self.body.local_decls[index].ty;
            if index_ty != self.tcx.types.usize {
                self.fail(location, format!("bad index ({:?} != usize)", index_ty))
            }
        }
        self.super_projection_elem(local, proj_base, elem, context, location);
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
            StatementKind::CopyNonOverlapping(box rustc_middle::mir::CopyNonOverlapping {
                ref src,
                ref dst,
                ref count,
            }) => {
                let src_ty = src.ty(&self.body.local_decls, self.tcx);
                let op_src_ty = if let Some(src_deref) = src_ty.builtin_deref(true) {
                    src_deref.ty
                } else {
                    self.fail(
                        location,
                        format!("Expected src to be ptr in copy_nonoverlapping, got: {}", src_ty),
                    );
                    return;
                };
                let dst_ty = dst.ty(&self.body.local_decls, self.tcx);
                let op_dst_ty = if let Some(dst_deref) = dst_ty.builtin_deref(true) {
                    dst_deref.ty
                } else {
                    self.fail(
                        location,
                        format!("Expected dst to be ptr in copy_nonoverlapping, got: {}", dst_ty),
                    );
                    return;
                };
                // since CopyNonOverlapping is parametrized by 1 type,
                // we only need to check that they are equal and not keep an extra parameter.
                if op_src_ty != op_dst_ty {
                    self.fail(location, format!("bad arg ({:?} != {:?})", op_src_ty, op_dst_ty));
                }

                let op_cnt_ty = count.ty(&self.body.local_decls, self.tcx);
                if op_cnt_ty != self.tcx.types.usize {
                    self.fail(location, format!("bad arg ({:?} != usize)", op_cnt_ty))
                }
            }
            StatementKind::SetDiscriminant { .. }
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::LlvmInlineAsm(..)
            | StatementKind::Retag(_, _)
            | StatementKind::Coverage(_)
            | StatementKind::Nop => {}
        }

        self.super_statement(statement, location);
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
            TerminatorKind::Call { func, args, destination, cleanup, .. } => {
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

                // The call destination place and Operand::Move place used as an argument might be
                // passed by a reference to the callee. Consequently they must be non-overlapping.
                // Currently this simply checks for duplicate places.
                self.place_cache.clear();
                if let Some((destination, _)) = destination {
                    self.place_cache.push(destination.as_ref());
                }
                for arg in args {
                    if let Operand::Move(place) = arg {
                        self.place_cache.push(place.as_ref());
                    }
                }
                let all_len = self.place_cache.len();
                self.place_cache.sort_unstable();
                self.place_cache.dedup();
                let has_duplicates = all_len != self.place_cache.len();
                if has_duplicates {
                    self.fail(
                        location,
                        format!(
                            "encountered overlapping memory in `Call` terminator: {:?}",
                            terminator.kind,
                        ),
                    );
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

        self.super_terminator(terminator, location);
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
