//! Validates the MIR to ensure that invariants are upheld.

use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::BitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::NonUseContext::VarDebugInfo;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    traversal, AggregateKind, BasicBlock, BinOp, Body, BorrowKind, CastKind, Local, Location,
    MirPass, MirPhase, Operand, Place, PlaceElem, PlaceRef, ProjectionElem, Rvalue, SourceScope,
    Statement, StatementKind, Terminator, TerminatorKind, UnOp, START_BLOCK,
};
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, InstanceDef, ParamEnv, Ty, TyCtxt, TypeFoldable, TypeVisitable};
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use rustc_target::abi::{Size, VariantIdx};

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
        // FIXME(JakobDegen): These bodies never instantiated in codegend anyway, so it's not
        // terribly important that they pass the validator. However, I think other passes might
        // still see them, in which case they might be surprised. It would probably be better if we
        // didn't put this through the MIR pipeline at all.
        if matches!(body.source.instance, InstanceDef::Intrinsic(..) | InstanceDef::Virtual(..)) {
            return;
        }
        let def_id = body.source.def_id();
        let param_env = tcx.param_env(def_id);
        let mir_phase = self.mir_phase;

        let always_live_locals = always_storage_live_locals(body);
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
            value_cache: Vec::new(),
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
pub fn equal_up_to_regions<'tcx>(
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
    let normalize = |ty: Ty<'tcx>| {
        let ty = ty.fold_with(&mut BottomUpFolder {
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
        });
        tcx.try_normalize_erasing_regions(param_env, ty).unwrap_or(ty)
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
    value_cache: Vec<u128>,
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
        if bb == START_BLOCK {
            self.fail(location, "start block must not have predecessors")
        }
        if let Some(bb) = self.body.basic_blocks.get(bb) {
            let src = self.body.basic_blocks.get(location.block).unwrap();
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
        // Normalization reveals opaque types, but we may be validating MIR while computing
        // said opaque types, causing cycles.
        if (src, dest).has_opaque_types() {
            return true;
        }
        // Normalize projections and things like that.
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
    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        if self.body.local_decls.get(local).is_none() {
            self.fail(
                location,
                format!("local {:?} has no corresponding declaration in `body.local_decls`", local),
            );
        }

        if self.reachable_blocks.contains(location.block) && context.is_use() {
            // We check that the local is live whenever it is used. Technically, violating this
            // restriction is only UB and not actually indicative of not well-formed MIR. This means
            // that an optimization which turns MIR that already has UB into MIR that fails this
            // check is not necessarily wrong. However, we have no such optimizations at the moment,
            // and so we include this check anyway to help us catch bugs. If you happen to write an
            // optimization that might cause this to incorrectly fire, feel free to remove this
            // check.
            self.storage_liveness.seek_after_primary_effect(location);
            let locals_with_storage = self.storage_liveness.get();
            if !locals_with_storage.contains(local) {
                self.fail(location, format!("use of local {:?}, which has no storage here", local));
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // This check is somewhat expensive, so only run it when -Zvalidate-mir is passed.
        if self.tcx.sess.opts.unstable_opts.validate_mir && self.mir_phase < MirPhase::DropsLowered
        {
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
        match elem {
            ProjectionElem::Index(index) => {
                let index_ty = self.body.local_decls[index].ty;
                if index_ty != self.tcx.types.usize {
                    self.fail(location, format!("bad index ({:?} != usize)", index_ty))
                }
            }
            ProjectionElem::Deref if self.mir_phase >= MirPhase::GeneratorsLowered => {
                let base_ty = Place::ty_from(local, proj_base, &self.body.local_decls, self.tcx).ty;

                if base_ty.is_box() {
                    self.fail(
                        location,
                        format!("{:?} dereferenced after ElaborateBoxDerefs", base_ty),
                    )
                }
            }
            ProjectionElem::Field(f, ty) => {
                let parent = Place { local, projection: self.tcx.intern_place_elems(proj_base) };
                let parent_ty = parent.ty(&self.body.local_decls, self.tcx);
                let fail_out_of_bounds = |this: &Self, location| {
                    this.fail(location, format!("Out of bounds field {:?} for {:?}", f, parent_ty));
                };
                let check_equal = |this: &Self, location, f_ty| {
                    if !this.mir_assign_valid_types(ty, f_ty) {
                        this.fail(
                        location,
                        format!(
                            "Field projection `{:?}.{:?}` specified type `{:?}`, but actual type is {:?}",
                            parent, f, ty, f_ty
                        )
                    )
                    }
                };

                let kind = match parent_ty.ty.kind() {
                    &ty::Opaque(def_id, substs) => {
                        self.tcx.bound_type_of(def_id).subst(self.tcx, substs).kind()
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
                    ty::Adt(adt_def, substs) => {
                        let var = parent_ty.variant_index.unwrap_or(VariantIdx::from_u32(0));
                        let Some(field) = adt_def.variant(var).fields.get(f.as_usize()) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, field.ty(self.tcx, substs));
                    }
                    ty::Closure(_, substs) => {
                        let substs = substs.as_closure();
                        let Some(f_ty) = substs.upvar_tys().nth(f.as_usize()) else {
                            fail_out_of_bounds(self, location);
                            return;
                        };
                        check_equal(self, location, f_ty);
                    }
                    &ty::Generator(def_id, substs, _) => {
                        let f_ty = if let Some(var) = parent_ty.variant_index {
                            let gen_body = if def_id == self.body.source.def_id() {
                                self.body
                            } else {
                                self.tcx.optimized_mir(def_id)
                            };

                            let Some(layout) = gen_body.generator_layout() else {
                                self.fail(location, format!("No generator layout for {:?}", parent_ty));
                                return;
                            };

                            let Some(&local) = layout.variant_fields[var].get(f) else {
                                fail_out_of_bounds(self, location);
                                return;
                            };

                            let Some(&f_ty) = layout.field_tys.get(local) else {
                                self.fail(location, format!("Out of bounds local {:?} for {:?}", local, parent_ty));
                                return;
                            };

                            f_ty
                        } else {
                            let Some(f_ty) = substs.as_generator().prefix_tys().nth(f.index()) else {
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
            _ => {}
        }
        self.super_projection_elem(local, proj_base, elem, context, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, cntxt: PlaceContext, location: Location) {
        // Set off any `bug!`s in the type computation code
        let _ = place.ty(&self.body.local_decls, self.tcx);

        if self.mir_phase >= MirPhase::Derefered
            && place.projection.len() > 1
            && cntxt != PlaceContext::NonUse(VarDebugInfo)
            && place.projection[1..].contains(&ProjectionElem::Deref)
        {
            self.fail(location, format!("{:?}, has deref at the wrong place", place));
        }

        self.super_place(place, cntxt, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        macro_rules! check_kinds {
            ($t:expr, $text:literal, $($patterns:tt)*) => {
                if !matches!(($t).kind(), $($patterns)*) {
                    self.fail(location, format!($text, $t));
                }
            };
        }
        match rvalue {
            Rvalue::Use(_) | Rvalue::CopyForDeref(_) => {}
            Rvalue::Aggregate(agg_kind, _) => {
                let disallowed = match **agg_kind {
                    AggregateKind::Array(..) => false,
                    AggregateKind::Generator(..) => self.mir_phase >= MirPhase::GeneratorsLowered,
                    _ => self.mir_phase >= MirPhase::Deaggregated,
                };
                if disallowed {
                    self.fail(
                        location,
                        format!("{:?} have been lowered to field assignments", rvalue),
                    )
                }
            }
            Rvalue::Ref(_, BorrowKind::Shallow, _) => {
                if self.mir_phase >= MirPhase::DropsLowered {
                    self.fail(
                        location,
                        "`Assign` statement with a `Shallow` borrow should have been removed after drop lowering phase",
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
                match op {
                    Offset => {
                        check_kinds!(a, "Cannot offset non-pointer type {:?}", ty::RawPtr(..));
                        if b != self.tcx.types.isize && b != self.tcx.types.usize {
                            self.fail(location, format!("Cannot offset by non-isize type {:?}", b));
                        }
                    }
                    Eq | Lt | Le | Ne | Ge | Gt => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot compare type {:?}",
                                ty::Bool
                                    | ty::Char
                                    | ty::Int(..)
                                    | ty::Uint(..)
                                    | ty::Float(..)
                                    | ty::RawPtr(..)
                                    | ty::FnPtr(..)
                            )
                        }
                        // The function pointer types can have lifetimes
                        if !self.mir_assign_valid_types(a, b) {
                            self.fail(
                                location,
                                format!("Cannot compare unequal types {:?} and {:?}", a, b),
                            );
                        }
                    }
                    Shl | Shr => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot shift non-integer type {:?}",
                                ty::Uint(..) | ty::Int(..)
                            )
                        }
                    }
                    BitAnd | BitOr | BitXor => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform bitwise op on type {:?}",
                                ty::Uint(..) | ty::Int(..) | ty::Bool
                            )
                        }
                        if a != b {
                            self.fail(
                                location,
                                format!(
                                    "Cannot perform bitwise op on unequal types {:?} and {:?}",
                                    a, b
                                ),
                            );
                        }
                    }
                    Add | Sub | Mul | Div | Rem => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform arithmetic on type {:?}",
                                ty::Uint(..) | ty::Int(..) | ty::Float(..)
                            )
                        }
                        if a != b {
                            self.fail(
                                location,
                                format!(
                                    "Cannot perform arithmetic on unequal types {:?} and {:?}",
                                    a, b
                                ),
                            );
                        }
                    }
                }
            }
            Rvalue::CheckedBinaryOp(op, vals) => {
                use BinOp::*;
                let a = vals.0.ty(&self.body.local_decls, self.tcx);
                let b = vals.1.ty(&self.body.local_decls, self.tcx);
                match op {
                    Add | Sub | Mul => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform checked arithmetic on type {:?}",
                                ty::Uint(..) | ty::Int(..)
                            )
                        }
                        if a != b {
                            self.fail(
                                location,
                                format!(
                                    "Cannot perform checked arithmetic on unequal types {:?} and {:?}",
                                    a, b
                                ),
                            );
                        }
                    }
                    Shl | Shr => {
                        for x in [a, b] {
                            check_kinds!(
                                x,
                                "Cannot perform checked shift on non-integer type {:?}",
                                ty::Uint(..) | ty::Int(..)
                            )
                        }
                    }
                    _ => self.fail(location, format!("There is no checked version of {:?}", op)),
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
                }
            }
            Rvalue::ShallowInitBox(operand, _) => {
                let a = operand.ty(&self.body.local_decls, self.tcx);
                check_kinds!(a, "Cannot shallow init type {:?}", ty::RawPtr(..));
            }
            Rvalue::Cast(kind, operand, target_type) => {
                match kind {
                    CastKind::Misc => {
                        let op_ty = operand.ty(self.body, self.tcx);
                        if op_ty.is_enum() {
                            self.fail(
                                location,
                                format!(
                                    "enum -> int casts should go through `Rvalue::Discriminant`: {operand:?}:{op_ty} as {target_type}",
                                ),
                            );
                        }
                    }
                    // Nothing to check here
                    CastKind::PointerFromExposedAddress
                    | CastKind::PointerExposeAddress
                    | CastKind::Pointer(_) => {}
                }
            }
            Rvalue::Repeat(_, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::AddressOf(_, _)
            | Rvalue::NullaryOp(_, _)
            | Rvalue::Discriminant(_) => {}
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
                    if !place.ty(&self.body.local_decls, self.tcx).ty.builtin_deref(true).is_some()
                    {
                        self.fail(
                            location,
                            "`CopyForDeref` should only be used for dereferenceable types",
                        )
                    }
                }
                // FIXME(JakobDegen): Check this for all rvalues, not just this one.
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) = rvalue {
                    // The sides of an assignment must not alias. Currently this just checks whether
                    // the places are identical.
                    if dest == src {
                        self.fail(
                            location,
                            "encountered `Assign` statement with overlapping memory",
                        );
                    }
                }
            }
            StatementKind::AscribeUserType(..) => {
                if self.mir_phase >= MirPhase::DropsLowered {
                    self.fail(
                        location,
                        "`AscribeUserType` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::FakeRead(..) => {
                if self.mir_phase >= MirPhase::DropsLowered {
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
                if !self.mir_assign_valid_types(op_src_ty, op_dst_ty) {
                    self.fail(location, format!("bad arg ({:?} != {:?})", op_src_ty, op_dst_ty));
                }

                let op_cnt_ty = count.ty(&self.body.local_decls, self.tcx);
                if op_cnt_ty != self.tcx.types.usize {
                    self.fail(location, format!("bad arg ({:?} != usize)", op_cnt_ty))
                }
            }
            StatementKind::SetDiscriminant { place, .. } => {
                if self.mir_phase < MirPhase::Deaggregated {
                    self.fail(location, "`SetDiscriminant`is not allowed until deaggregation");
                }
                let pty = place.ty(&self.body.local_decls, self.tcx).ty.kind();
                if !matches!(pty, ty::Adt(..) | ty::Generator(..) | ty::Opaque(..)) {
                    self.fail(
                        location,
                        format!(
                            "`SetDiscriminant` is only allowed on ADTs and generators, not {:?}",
                            pty
                        ),
                    );
                }
            }
            StatementKind::Deinit(..) => {
                if self.mir_phase < MirPhase::Deaggregated {
                    self.fail(location, "`Deinit`is not allowed until deaggregation");
                }
            }
            StatementKind::Retag(_, _) => {
                // FIXME(JakobDegen) The validator should check that `self.mir_phase <
                // DropsLowered`. However, this causes ICEs with generation of drop shims, which
                // seem to fail to set their `MirPhase` correctly.
            }
            StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
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

                self.value_cache.clear();
                self.value_cache.extend(targets.iter().map(|(value, _)| value));
                let all_len = self.value_cache.len();
                self.value_cache.sort_unstable();
                self.value_cache.dedup();
                let has_duplicates = all_len != self.value_cache.len();
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
            TerminatorKind::Drop { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::DropAndReplace { target, unwind, .. } => {
                if self.mir_phase >= MirPhase::DropsLowered {
                    self.fail(
                        location,
                        "`DropAndReplace` should have been removed during drop elaboration",
                    );
                }
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Call { func, args, destination, target, cleanup, .. } => {
                let func_ty = func.ty(&self.body.local_decls, self.tcx);
                match func_ty.kind() {
                    ty::FnPtr(..) | ty::FnDef(..) => {}
                    _ => self.fail(
                        location,
                        format!("encountered non-callable type {} in `Call` terminator", func_ty),
                    ),
                }
                if let Some(target) = target {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }

                // The call destination place and Operand::Move place used as an argument might be
                // passed by a reference to the callee. Consequently they must be non-overlapping.
                // Currently this simply checks for duplicate places.
                self.place_cache.clear();
                self.place_cache.push(destination.as_ref());
                for arg in args {
                    if let Operand::Move(place) = arg {
                        self.place_cache.push(place.as_ref());
                    }
                }
                let all_len = self.place_cache.len();
                let mut dedup = FxHashSet::default();
                self.place_cache.retain(|p| dedup.insert(*p));
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
                if self.body.generator.is_none() {
                    self.fail(location, "`Yield` cannot appear outside generator bodies");
                }
                if self.mir_phase >= MirPhase::GeneratorsLowered {
                    self.fail(location, "`Yield` should have been replaced by generator lowering");
                }
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                if self.mir_phase >= MirPhase::DropsLowered {
                    self.fail(
                        location,
                        "`FalseEdge` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                if self.mir_phase >= MirPhase::DropsLowered {
                    self.fail(
                        location,
                        "`FalseUnwind` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::InlineAsm { destination, cleanup, .. } => {
                if let Some(destination) = destination {
                    self.check_edge(location, *destination, EdgeKind::Normal);
                }
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::GeneratorDrop => {
                if self.body.generator.is_none() {
                    self.fail(location, "`GeneratorDrop` cannot appear outside generator bodies");
                }
                if self.mir_phase >= MirPhase::GeneratorsLowered {
                    self.fail(
                        location,
                        "`GeneratorDrop` should have been replaced by generator lowering",
                    );
                }
            }
            TerminatorKind::Resume | TerminatorKind::Abort => {
                let bb = location.block;
                if !self.body.basic_blocks[bb].is_cleanup {
                    self.fail(location, "Cannot `Resume` or `Abort` from non-cleanup basic block")
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
