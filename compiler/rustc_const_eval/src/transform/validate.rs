//! Validates the MIR to ensure that invariants are upheld.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_infer::traits::Reveal;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::{NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    traversal, BasicBlock, BinOp, Body, BorrowKind, CastKind, CopyNonOverlapping, Local, Location,
    MirPass, MirPhase, NonDivergingIntrinsic, NullOp, Operand, Place, PlaceElem, PlaceRef,
    ProjectionElem, RetagKind, RuntimePhase, Rvalue, SourceScope, Statement, StatementKind,
    Terminator, TerminatorKind, UnOp, UnwindAction, VarDebugInfo, VarDebugInfoContents,
    START_BLOCK,
};
use rustc_middle::ty::{self, InstanceDef, ParamEnv, Ty, TyCtxt, TypeVisitableExt};
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use rustc_target::abi::{Size, FIRST_VARIANT};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
        let mir_phase = self.mir_phase;
        let param_env = match mir_phase.reveal() {
            Reveal::UserFacing => tcx.param_env(def_id),
            Reveal::All => tcx.param_env_reveal_all_normalized(def_id),
        };

        let always_live_locals = always_storage_live_locals(body);
        let storage_liveness = MaybeStorageLive::new(std::borrow::Cow::Owned(always_live_locals))
            .into_engine(tcx, body)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        let mut checker = TypeChecker {
            when: &self.when,
            body,
            tcx,
            param_env,
            mir_phase,
            unwind_edge_count: 0,
            reachable_blocks: traversal::reachable_as_bitset(body),
            storage_liveness,
            place_cache: Vec::new(),
            value_cache: Vec::new(),
        };
        checker.visit_body(body);
        checker.check_cleanup_control_flow();

        if let MirPhase::Runtime(_) = body.phase {
            if let ty::InstanceDef::Item(_) = body.source.instance {
                if body.has_free_regions() {
                    checker.fail(
                        Location::START,
                        format!("Free regions in optimized {} MIR", body.phase.name()),
                    );
                }
            }
        }
    }
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    mir_phase: MirPhase,
    unwind_edge_count: usize,
    reachable_blocks: BitSet<BasicBlock>,
    storage_liveness: ResultsCursor<'a, 'tcx, MaybeStorageLive<'static>>,
    place_cache: Vec<PlaceRef<'tcx>>,
    value_cache: Vec<u128>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    #[track_caller]
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        let span = self.body.source_info(location).span;
        // We use `delay_span_bug` as we might see broken MIR when other errors have already
        // occurred.
        self.tcx.sess.diagnostic().delay_span_bug(
            span,
            format!(
                "broken MIR in {:?} ({}) at {:?}:\n{}",
                self.body.source.instance,
                self.when,
                location,
                msg.as_ref()
            ),
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
            self.fail(location, format!("encountered jump to invalid basic block {:?}", bb))
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
        for i in 0..parent.len() {
            let mut bb = BasicBlock::from_usize(i);
            stack.clear();
            stack.insert(bb);
            loop {
                let Some(parent)= parent[bb].take() else {
                    break
                };
                let no_cycle = stack.insert(parent);
                if !no_cycle {
                    self.fail(
                        Location { block: bb, statement_index: 0 },
                        format!(
                            "Cleanup control flow violation: Cycle involving edge {:?} -> {:?}",
                            bb, parent,
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
                    self.fail(location, "unwind on cleanup block");
                }
                self.check_edge(location, unwind, EdgeKind::Unwind);
            }
            UnwindAction::Continue => {
                if is_cleanup {
                    self.fail(location, "unwind on cleanup block");
                }
            }
            UnwindAction::Unreachable | UnwindAction::Terminate => (),
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

        // We sometimes have to use `defining_opaque_types` for subtyping
        // to succeed here and figuring out how exactly that should work
        // is annoying. It is harmless enough to just not validate anything
        // in that case. We still check this after analysis as all opaque
        // types have been revealed at this point.
        if (src, dest).has_opaque_types() {
            return true;
        }

        crate::util::is_subtype(self.tcx, self.param_env, src, dest)
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
        if self.tcx.sess.opts.unstable_opts.validate_mir
            && self.mir_phase < MirPhase::Runtime(RuntimePhase::Initial)
        {
            // `Operand::Copy` is only supposed to be used with `Copy` types.
            if let Operand::Copy(place) = operand {
                let ty = place.ty(&self.body.local_decls, self.tcx).ty;

                if !ty.is_copy_modulo_regions(self.tcx, self.param_env) {
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
            ProjectionElem::Deref
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::PostCleanup) =>
            {
                let base_ty = Place::ty_from(local, proj_base, &self.body.local_decls, self.tcx).ty;

                if base_ty.is_box() {
                    self.fail(
                        location,
                        format!("{:?} dereferenced after ElaborateBoxDerefs", base_ty),
                    )
                }
            }
            ProjectionElem::Field(f, ty) => {
                let parent = Place { local, projection: self.tcx.mk_place_elems(proj_base) };
                let parent_ty = parent.ty(&self.body.local_decls, self.tcx);
                let fail_out_of_bounds = |this: &Self, location| {
                    this.fail(location, format!("Out of bounds field {:?} for {:?}", f, parent_ty));
                };
                let check_equal = |this: &Self, location, f_ty| {
                    if !this.mir_assign_valid_types(ty, f_ty) {
                        this.fail(
                            location,
                            format!(
                                "Field projection `{:?}.{:?}` specified type `{:?}`, but actual type is `{:?}`",
                                parent, f, ty, f_ty
                            )
                        )
                    }
                };

                let kind = match parent_ty.ty.kind() {
                    &ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                        self.tcx.type_of(def_id).subst(self.tcx, substs).kind()
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
                        let var = parent_ty.variant_index.unwrap_or(FIRST_VARIANT);
                        let Some(field) = adt_def.variant(var).fields.get(f) else {
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

                            let Some(f_ty) = layout.field_tys.get(local) else {
                                self.fail(location, format!("Out of bounds local {:?} for {:?}", local, parent_ty));
                                return;
                            };

                            f_ty.ty
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

    fn visit_var_debug_info(&mut self, debuginfo: &VarDebugInfo<'tcx>) {
        let check_place = |place: Place<'_>| {
            if place.projection.iter().any(|p| !p.can_use_in_debuginfo()) {
                self.fail(
                    START_BLOCK.start_location(),
                    format!("illegal place {:?} in debuginfo for {:?}", place, debuginfo.name),
                );
            }
        };
        match debuginfo.value {
            VarDebugInfoContents::Const(_) => {}
            VarDebugInfoContents::Place(place) => {
                check_place(place);
                if debuginfo.references != 0 && place.projection.last() == Some(&PlaceElem::Deref) {
                    self.fail(
                        START_BLOCK.start_location(),
                        format!("debuginfo {:?}, has both ref and deref", debuginfo),
                    );
                }
            }
            VarDebugInfoContents::Composite { ty, ref fragments } => {
                for f in fragments {
                    check_place(f.contents);
                    if ty.is_union() || ty.is_enum() {
                        self.fail(
                            START_BLOCK.start_location(),
                            format!("invalid type {:?} for composite debuginfo", ty),
                        );
                    }
                    if f.projection.iter().any(|p| !matches!(p, PlaceElem::Field(..))) {
                        self.fail(
                            START_BLOCK.start_location(),
                            format!(
                                "illegal projection {:?} in debuginfo for {:?}",
                                f.projection, debuginfo.name
                            ),
                        );
                    }
                }
            }
        }
        self.super_var_debug_info(debuginfo);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, cntxt: PlaceContext, location: Location) {
        // Set off any `bug!`s in the type computation code
        let _ = place.ty(&self.body.local_decls, self.tcx);

        if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial)
            && place.projection.len() > 1
            && cntxt != PlaceContext::NonUse(NonUseContext::VarDebugInfo)
            && place.projection[1..].contains(&ProjectionElem::Deref)
        {
            self.fail(location, format!("{:?}, has deref at the wrong place", place));
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
            Rvalue::Use(_) | Rvalue::CopyForDeref(_) | Rvalue::Aggregate(..) => {}
            Rvalue::Ref(_, BorrowKind::Shallow, _) => {
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`Assign` statement with a `Shallow` borrow should have been removed in runtime MIR",
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
                                format!("Cannot {op:?} compare incompatible types {a:?} and {b:?}"),
                            );
                        }
                    } else if a != b {
                        self.fail(
                            location,
                            format!(
                                "Cannot perform binary op {op:?} on unequal types {a:?} and {b:?}"
                            ),
                        );
                    }
                }

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
                    AddUnchecked | SubUnchecked | MulUnchecked | Shl | ShlUnchecked | Shr
                    | ShrUnchecked => {
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
                let op_ty = operand.ty(self.body, self.tcx);
                match kind {
                    CastKind::DynStar => {
                        // FIXME(dyn-star): make sure nothing needs to be done here.
                    }
                    // FIXME: Add Checks for these
                    CastKind::PointerFromExposedAddress
                    | CastKind::PointerExposeAddress
                    | CastKind::Pointer(_) => {}
                    CastKind::IntToInt | CastKind::IntToFloat => {
                        let input_valid = op_ty.is_integral() || op_ty.is_char() || op_ty.is_bool();
                        let target_valid = target_type.is_numeric() || target_type.is_char();
                        if !input_valid || !target_valid {
                            self.fail(
                                location,
                                format!("Wrong cast kind {kind:?} for the type {op_ty}",),
                            );
                        }
                    }
                    CastKind::FnPtrToPtr | CastKind::PtrToPtr => {
                        if !(op_ty.is_any_ptr() && target_type.is_unsafe_ptr()) {
                            self.fail(location, "Can't cast {op_ty} into 'Ptr'");
                        }
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
                        if let MirPhase::Runtime(..) = self.mir_phase {
                            // Unlike `mem::transmute`, a MIR `Transmute` is well-formed
                            // for any two `Sized` types, just potentially UB to run.

                            if !self
                                .tcx
                                .normalize_erasing_regions(self.param_env, op_ty)
                                .is_sized(self.tcx, self.param_env)
                            {
                                self.fail(
                                    location,
                                    format!("Cannot transmute from non-`Sized` type {op_ty:?}"),
                                );
                            }
                            if !self
                                .tcx
                                .normalize_erasing_regions(self.param_env, *target_type)
                                .is_sized(self.tcx, self.param_env)
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
                                    self.mir_phase
                                ),
                            );
                        }
                    }
                }
            }
            Rvalue::NullaryOp(NullOp::OffsetOf(fields), container) => {
                let fail_out_of_bounds = |this: &Self, location, field, ty| {
                    this.fail(location, format!("Out of bounds field {field:?} for {ty:?}"));
                };

                let mut current_ty = *container;

                for field in fields.iter() {
                    match current_ty.kind() {
                        ty::Tuple(fields) => {
                            let Some(&f_ty) = fields.get(field.as_usize()) else {
                                fail_out_of_bounds(self, location, field, current_ty);
                                return;
                            };

                            current_ty = self.tcx.normalize_erasing_regions(self.param_env, f_ty);
                        }
                        ty::Adt(adt_def, substs) => {
                            if adt_def.is_enum() {
                                self.fail(
                                    location,
                                    format!("Cannot get field offset from enum {current_ty:?}"),
                                );
                                return;
                            }

                            let Some(field) = adt_def.non_enum_variant().fields.get(field) else {
                                fail_out_of_bounds(self, location, field, current_ty);
                                return;
                            };

                            let f_ty = field.ty(self.tcx, substs);
                            current_ty = self.tcx.normalize_erasing_regions(self.param_env, f_ty);
                        }
                        _ => {
                            self.fail(
                                location,
                                format!("Cannot get field offset from non-adt type {current_ty:?}"),
                            );
                            return;
                        }
                    }
                }
            }
            Rvalue::Repeat(_, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::AddressOf(_, _)
            | Rvalue::NullaryOp(NullOp::SizeOf | NullOp::AlignOf, _)
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
                    if place.ty(&self.body.local_decls, self.tcx).ty.builtin_deref(true).is_none() {
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
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`AscribeUserType` should have been removed after drop lowering phase",
                    );
                }
            }
            StatementKind::FakeRead(..) => {
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
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
                        format!("`assume` argument must be `bool`, but got: `{}`", ty),
                    );
                }
            }
            StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(
                CopyNonOverlapping { src, dst, count },
            )) => {
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
                if self.mir_phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`SetDiscriminant`is not allowed until deaggregation");
                }
                let pty = place.ty(&self.body.local_decls, self.tcx).ty.kind();
                if !matches!(pty, ty::Adt(..) | ty::Generator(..) | ty::Alias(ty::Opaque, ..)) {
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
                if self.mir_phase < MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`Deinit`is not allowed until deaggregation");
                }
            }
            StatementKind::Retag(kind, _) => {
                // FIXME(JakobDegen) The validator should check that `self.mir_phase <
                // DropsLowered`. However, this causes ICEs with generation of drop shims, which
                // seem to fail to set their `MirPhase` correctly.
                if matches!(kind, RetagKind::Raw | RetagKind::TwoPhase) {
                    self.fail(location, format!("explicit `{:?}` is forbidden", kind));
                }
            }
            StatementKind::StorageLive(local) => {
                // We check that the local is not live when entering a `StorageLive` for it.
                // Technically, violating this restriction is only UB and not actually indicative
                // of not well-formed MIR. This means that an optimization which turns MIR that
                // already has UB into MIR that fails this check is not necessarily wrong. However,
                // we have no such optimizations at the moment, and so we include this check anyway
                // to help us catch bugs. If you happen to write an optimization that might cause
                // this to incorrectly fire, feel free to remove this check.
                if self.reachable_blocks.contains(location.block) {
                    self.storage_liveness.seek_before_primary_effect(location);
                    let locals_with_storage = self.storage_liveness.get();
                    if locals_with_storage.contains(*local) {
                        self.fail(
                            location,
                            format!("StorageLive({local:?}) which already has storage here"),
                        );
                    }
                }
            }
            StatementKind::StorageDead(_)
            | StatementKind::Coverage(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::Nop => {}
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                self.check_edge(location, *target, EdgeKind::Normal);
            }
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
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::Call { func, args, destination, target, unwind, .. } => {
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
                self.check_unwind_edge(location, *unwind);

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
            TerminatorKind::Assert { cond, target, unwind, .. } => {
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
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                if self.body.generator.is_none() {
                    self.fail(location, "`Yield` cannot appear outside generator bodies");
                }
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(location, "`Yield` should have been replaced by generator lowering");
                }
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FalseEdge` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`FalseUnwind` should have been removed after drop elaboration",
                    );
                }
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::InlineAsm { destination, unwind, .. } => {
                if let Some(destination) = destination {
                    self.check_edge(location, *destination, EdgeKind::Normal);
                }
                self.check_unwind_edge(location, *unwind);
            }
            TerminatorKind::GeneratorDrop => {
                if self.body.generator.is_none() {
                    self.fail(location, "`GeneratorDrop` cannot appear outside generator bodies");
                }
                if self.mir_phase >= MirPhase::Runtime(RuntimePhase::Initial) {
                    self.fail(
                        location,
                        "`GeneratorDrop` should have been replaced by generator lowering",
                    );
                }
            }
            TerminatorKind::Resume | TerminatorKind::Terminate => {
                let bb = location.block;
                if !self.body.basic_blocks[bb].is_cleanup {
                    self.fail(
                        location,
                        "Cannot `Resume` or `Terminate` from non-cleanup basic block",
                    )
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
                format!(
                    "broken MIR in {:?} ({}):\ninvalid source scope {:?}",
                    self.body.source.instance, self.when, scope,
                ),
            );
        }
    }
}
