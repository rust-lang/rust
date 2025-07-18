//! A constant propagation optimization pass based on dataflow analysis.
//!
//! Currently, this pass only propagates scalar values.

use std::assert_matches::assert_matches;
use std::fmt::Formatter;

use rustc_abi::{BackendRepr, FIRST_VARIANT, FieldIdx, Size, VariantIdx};
use rustc_const_eval::const_eval::{DummyMachine, throw_machine_stop_str};
use rustc_const_eval::interpret::{
    ImmTy, Immediate, InterpCx, OpTy, PlaceTy, Projectable, interp_ok,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{InterpResult, Scalar};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::lattice::{FlatSet, HasBottom};
use rustc_mir_dataflow::value_analysis::{
    Map, PlaceIndex, State, TrackElem, ValueOrPlace, debug_with_context,
};
use rustc_mir_dataflow::{Analysis, ResultsVisitor, visit_reachable_results};
use rustc_span::DUMMY_SP;
use tracing::{debug, debug_span, instrument};

// These constants are somewhat random guesses and have not been optimized.
// If `tcx.sess.mir_opt_level() >= 4`, we ignore the limits (this can become very expensive).
const BLOCK_LIMIT: usize = 100;
const PLACE_LIMIT: usize = 100;

pub(super) struct DataflowConstProp;

impl<'tcx> crate::MirPass<'tcx> for DataflowConstProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 3
    }

    #[instrument(skip_all level = "debug")]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());
        if tcx.sess.mir_opt_level() < 4 && body.basic_blocks.len() > BLOCK_LIMIT {
            debug!("aborted dataflow const prop due too many basic blocks");
            return;
        }

        // We want to have a somewhat linear runtime w.r.t. the number of statements/terminators.
        // Let's call this number `n`. Dataflow analysis has `O(h*n)` transfer function
        // applications, where `h` is the height of the lattice. Because the height of our lattice
        // is linear w.r.t. the number of tracked places, this is `O(tracked_places * n)`. However,
        // because every transfer function application could traverse the whole map, this becomes
        // `O(num_nodes * tracked_places * n)` in terms of time complexity. Since the number of
        // map nodes is strongly correlated to the number of tracked places, this becomes more or
        // less `O(n)` if we place a constant limit on the number of tracked places.
        let place_limit = if tcx.sess.mir_opt_level() < 4 { Some(PLACE_LIMIT) } else { None };

        // Decide which places to track during the analysis.
        let map = Map::new(tcx, body, place_limit);

        // Perform the actual dataflow analysis.
        let mut const_ = debug_span!("analyze")
            .in_scope(|| ConstAnalysis::new(tcx, body, map).iterate_to_fixpoint(tcx, body, None));

        // Collect results and patch the body afterwards.
        let mut visitor = Collector::new(tcx, &body.local_decls);
        debug_span!("collect").in_scope(|| {
            visit_reachable_results(body, &mut const_.analysis, &const_.results, &mut visitor)
        });
        let mut patch = visitor.patch;
        debug_span!("patch").in_scope(|| patch.visit_body_preserves_cfg(body));
    }

    fn is_required(&self) -> bool {
        false
    }
}

// Note: Currently, places that have their reference taken cannot be tracked. Although this would
// be possible, it has to rely on some aliasing model, which we are not ready to commit to yet.
// Because of that, we can assume that the only way to change the value behind a tracked place is
// by direct assignment.
struct ConstAnalysis<'a, 'tcx> {
    map: Map<'tcx>,
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    ecx: InterpCx<'tcx, DummyMachine>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> Analysis<'tcx> for ConstAnalysis<'_, 'tcx> {
    type Domain = State<FlatSet<Scalar>>;

    const NAME: &'static str = "ConstAnalysis";

    // The bottom state denotes uninitialized memory. Because we are only doing a sound
    // approximation of the actual execution, we can also use this state for places where access
    // would be UB.
    fn bottom_value(&self, _body: &Body<'tcx>) -> Self::Domain {
        State::Unreachable
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // The initial state maps all tracked places of argument projections to ⊤ and the rest to ⊥.
        assert_matches!(state, State::Unreachable);
        *state = State::new_reachable();
        for arg in body.args_iter() {
            state.flood(PlaceRef { local: arg, projection: &[] }, &self.map);
        }
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        _location: Location,
    ) {
        if state.is_reachable() {
            self.handle_statement(statement, state);
        }
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        _location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        if state.is_reachable() {
            self.handle_terminator(terminator, state)
        } else {
            TerminatorEdges::None
        }
    }

    fn apply_call_return_effect(
        &mut self,
        state: &mut Self::Domain,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        if state.is_reachable() {
            self.handle_call_return(return_places, state)
        }
    }
}

impl<'a, 'tcx> ConstAnalysis<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, map: Map<'tcx>) -> Self {
        let typing_env = body.typing_env(tcx);
        Self {
            map,
            tcx,
            local_decls: &body.local_decls,
            ecx: InterpCx::new(tcx, DUMMY_SP, typing_env, DummyMachine),
            typing_env,
        }
    }

    fn handle_statement(&self, statement: &Statement<'tcx>, state: &mut State<FlatSet<Scalar>>) {
        match &statement.kind {
            StatementKind::Assign(box (place, rvalue)) => {
                self.handle_assign(*place, rvalue, state);
            }
            StatementKind::SetDiscriminant { box place, variant_index } => {
                self.handle_set_discriminant(*place, *variant_index, state);
            }
            StatementKind::Intrinsic(box intrinsic) => {
                self.handle_intrinsic(intrinsic);
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                // StorageLive leaves the local in an uninitialized state.
                // StorageDead makes it UB to access the local afterwards.
                state.flood_with(
                    Place::from(*local).as_ref(),
                    &self.map,
                    FlatSet::<Scalar>::BOTTOM,
                );
            }
            StatementKind::Deinit(box place) => {
                // Deinit makes the place uninitialized.
                state.flood_with(place.as_ref(), &self.map, FlatSet::<Scalar>::BOTTOM);
            }
            StatementKind::Retag(..) => {
                // We don't track references.
            }
            StatementKind::ConstEvalCounter
            | StatementKind::Nop
            | StatementKind::FakeRead(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::AscribeUserType(..) => {}
        }
    }

    fn handle_intrinsic(&self, intrinsic: &NonDivergingIntrinsic<'tcx>) {
        match intrinsic {
            NonDivergingIntrinsic::Assume(..) => {
                // Could use this, but ignoring it is sound.
            }
            NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                dst: _,
                src: _,
                count: _,
            }) => {
                // This statement represents `*dst = *src`, `count` times.
            }
        }
    }

    fn handle_operand(
        &self,
        operand: &Operand<'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) -> ValueOrPlace<FlatSet<Scalar>> {
        match operand {
            Operand::Constant(box constant) => {
                ValueOrPlace::Value(self.handle_constant(constant, state))
            }
            Operand::Copy(place) | Operand::Move(place) => {
                // On move, we would ideally flood the place with bottom. But with the current
                // framework this is not possible (similar to `InterpCx::eval_operand`).
                self.map.find(place.as_ref()).map(ValueOrPlace::Place).unwrap_or(ValueOrPlace::TOP)
            }
        }
    }

    /// The effect of a successful function call return should not be
    /// applied here, see [`Analysis::apply_primary_terminator_effect`].
    fn handle_terminator<'mir>(
        &self,
        terminator: &'mir Terminator<'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) -> TerminatorEdges<'mir, 'tcx> {
        match &terminator.kind {
            TerminatorKind::Call { .. } | TerminatorKind::InlineAsm { .. } => {
                // Effect is applied by `handle_call_return`.
            }
            TerminatorKind::Drop { place, .. } => {
                state.flood_with(place.as_ref(), &self.map, FlatSet::<Scalar>::BOTTOM);
            }
            TerminatorKind::Yield { .. } => {
                // They would have an effect, but are not allowed in this phase.
                bug!("encountered disallowed terminator");
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                return self.handle_switch_int(discr, targets, state);
            }
            TerminatorKind::TailCall { .. } => {
                // FIXME(explicit_tail_calls): determine if we need to do something here (probably
                // not)
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Assert { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                // These terminators have no effect on the analysis.
            }
        }
        terminator.edges()
    }

    fn handle_call_return(
        &self,
        return_places: CallReturnPlaces<'_, 'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) {
        return_places.for_each(|place| {
            state.flood(place.as_ref(), &self.map);
        })
    }

    fn handle_set_discriminant(
        &self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
        state: &mut State<FlatSet<Scalar>>,
    ) {
        state.flood_discr(place.as_ref(), &self.map);
        if self.map.find_discr(place.as_ref()).is_some() {
            let enum_ty = place.ty(self.local_decls, self.tcx).ty;
            if let Some(discr) = self.eval_discriminant(enum_ty, variant_index) {
                state.assign_discr(
                    place.as_ref(),
                    ValueOrPlace::Value(FlatSet::Elem(discr)),
                    &self.map,
                );
            }
        }
    }

    fn handle_assign(
        &self,
        target: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) {
        match rvalue {
            Rvalue::Use(operand) => {
                state.flood(target.as_ref(), &self.map);
                if let Some(target) = self.map.find(target.as_ref()) {
                    self.assign_operand(state, target, operand);
                }
            }
            Rvalue::CopyForDeref(rhs) => {
                state.flood(target.as_ref(), &self.map);
                if let Some(target) = self.map.find(target.as_ref()) {
                    self.assign_operand(state, target, &Operand::Copy(*rhs));
                }
            }
            Rvalue::Aggregate(kind, operands) => {
                // If we assign `target = Enum::Variant#0(operand)`,
                // we must make sure that all `target as Variant#i` are `Top`.
                state.flood(target.as_ref(), &self.map);

                let Some(target_idx) = self.map.find(target.as_ref()) else { return };

                let (variant_target, variant_index) = match **kind {
                    AggregateKind::Tuple | AggregateKind::Closure(..) => (Some(target_idx), None),
                    AggregateKind::Adt(def_id, variant_index, ..) => {
                        match self.tcx.def_kind(def_id) {
                            DefKind::Struct => (Some(target_idx), None),
                            DefKind::Enum => (
                                self.map.apply(target_idx, TrackElem::Variant(variant_index)),
                                Some(variant_index),
                            ),
                            _ => return,
                        }
                    }
                    _ => return,
                };
                if let Some(variant_target_idx) = variant_target {
                    for (field_index, operand) in operands.iter_enumerated() {
                        if let Some(field) =
                            self.map.apply(variant_target_idx, TrackElem::Field(field_index))
                        {
                            self.assign_operand(state, field, operand);
                        }
                    }
                }
                if let Some(variant_index) = variant_index
                    && let Some(discr_idx) = self.map.apply(target_idx, TrackElem::Discriminant)
                {
                    // We are assigning the discriminant as part of an aggregate.
                    // This discriminant can only alias a variant field's value if the operand
                    // had an invalid value for that type.
                    // Using invalid values is UB, so we are allowed to perform the assignment
                    // without extra flooding.
                    let enum_ty = target.ty(self.local_decls, self.tcx).ty;
                    if let Some(discr_val) = self.eval_discriminant(enum_ty, variant_index) {
                        state.insert_value_idx(discr_idx, FlatSet::Elem(discr_val), &self.map);
                    }
                }
            }
            Rvalue::BinaryOp(op, box (left, right)) if op.is_overflowing() => {
                // Flood everything now, so we can use `insert_value_idx` directly later.
                state.flood(target.as_ref(), &self.map);

                let Some(target) = self.map.find(target.as_ref()) else { return };

                let value_target = self.map.apply(target, TrackElem::Field(0_u32.into()));
                let overflow_target = self.map.apply(target, TrackElem::Field(1_u32.into()));

                if value_target.is_some() || overflow_target.is_some() {
                    let (val, overflow) = self.binary_op(state, *op, left, right);

                    if let Some(value_target) = value_target {
                        // We have flooded `target` earlier.
                        state.insert_value_idx(value_target, val, &self.map);
                    }
                    if let Some(overflow_target) = overflow_target {
                        // We have flooded `target` earlier.
                        state.insert_value_idx(overflow_target, overflow, &self.map);
                    }
                }
            }
            Rvalue::Cast(
                CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                operand,
                _,
            ) => {
                let pointer = self.handle_operand(operand, state);
                state.assign(target.as_ref(), pointer, &self.map);

                if let Some(target_len) = self.map.find_len(target.as_ref())
                    && let operand_ty = operand.ty(self.local_decls, self.tcx)
                    && let Some(operand_ty) = operand_ty.builtin_deref(true)
                    && let ty::Array(_, len) = operand_ty.kind()
                    && let Some(len) = Const::Ty(self.tcx.types.usize, *len)
                        .try_eval_scalar_int(self.tcx, self.typing_env)
                {
                    state.insert_value_idx(target_len, FlatSet::Elem(len.into()), &self.map);
                }
            }
            _ => {
                let result = self.handle_rvalue(rvalue, state);
                state.assign(target.as_ref(), result, &self.map);
            }
        }
    }

    fn handle_rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) -> ValueOrPlace<FlatSet<Scalar>> {
        let val = match rvalue {
            Rvalue::Len(place) => {
                let place_ty = place.ty(self.local_decls, self.tcx);
                if let ty::Array(_, len) = place_ty.ty.kind() {
                    Const::Ty(self.tcx.types.usize, *len)
                        .try_eval_scalar(self.tcx, self.typing_env)
                        .map_or(FlatSet::Top, FlatSet::Elem)
                } else if let [ProjectionElem::Deref] = place.projection[..] {
                    state.get_len(place.local.into(), &self.map)
                } else {
                    FlatSet::Top
                }
            }
            Rvalue::Cast(CastKind::IntToInt | CastKind::IntToFloat, operand, ty) => {
                let Ok(layout) = self.tcx.layout_of(self.typing_env.as_query_input(*ty)) else {
                    return ValueOrPlace::Value(FlatSet::Top);
                };
                match self.eval_operand(operand, state) {
                    FlatSet::Elem(op) => self
                        .ecx
                        .int_to_int_or_float(&op, layout)
                        .discard_err()
                        .map_or(FlatSet::Top, |result| self.wrap_immediate(*result)),
                    FlatSet::Bottom => FlatSet::Bottom,
                    FlatSet::Top => FlatSet::Top,
                }
            }
            Rvalue::Cast(CastKind::FloatToInt | CastKind::FloatToFloat, operand, ty) => {
                let Ok(layout) = self.tcx.layout_of(self.typing_env.as_query_input(*ty)) else {
                    return ValueOrPlace::Value(FlatSet::Top);
                };
                match self.eval_operand(operand, state) {
                    FlatSet::Elem(op) => self
                        .ecx
                        .float_to_float_or_int(&op, layout)
                        .discard_err()
                        .map_or(FlatSet::Top, |result| self.wrap_immediate(*result)),
                    FlatSet::Bottom => FlatSet::Bottom,
                    FlatSet::Top => FlatSet::Top,
                }
            }
            Rvalue::Cast(CastKind::Transmute, operand, _) => {
                match self.eval_operand(operand, state) {
                    FlatSet::Elem(op) => self.wrap_immediate(*op),
                    FlatSet::Bottom => FlatSet::Bottom,
                    FlatSet::Top => FlatSet::Top,
                }
            }
            Rvalue::BinaryOp(op, box (left, right)) if !op.is_overflowing() => {
                // Overflows must be ignored here.
                // The overflowing operators are handled in `handle_assign`.
                let (val, _overflow) = self.binary_op(state, *op, left, right);
                val
            }
            Rvalue::UnaryOp(op, operand) => match self.eval_operand(operand, state) {
                FlatSet::Elem(value) => self
                    .ecx
                    .unary_op(*op, &value)
                    .discard_err()
                    .map_or(FlatSet::Top, |val| self.wrap_immediate(*val)),
                FlatSet::Bottom => FlatSet::Bottom,
                FlatSet::Top => FlatSet::Top,
            },
            Rvalue::NullaryOp(null_op, ty) => {
                let Ok(layout) = self.tcx.layout_of(self.typing_env.as_query_input(*ty)) else {
                    return ValueOrPlace::Value(FlatSet::Top);
                };
                let val = match null_op {
                    NullOp::SizeOf if layout.is_sized() => layout.size.bytes(),
                    NullOp::AlignOf if layout.is_sized() => layout.align.abi.bytes(),
                    NullOp::OffsetOf(fields) => self
                        .ecx
                        .tcx
                        .offset_of_subfield(self.typing_env, layout, fields.iter())
                        .bytes(),
                    _ => return ValueOrPlace::Value(FlatSet::Top),
                };
                FlatSet::Elem(Scalar::from_target_usize(val, &self.tcx))
            }
            Rvalue::Discriminant(place) => state.get_discr(place.as_ref(), &self.map),
            Rvalue::Use(operand) => return self.handle_operand(operand, state),
            Rvalue::CopyForDeref(place) => {
                return self.handle_operand(&Operand::Copy(*place), state);
            }
            Rvalue::Ref(..) | Rvalue::RawPtr(..) => {
                // We don't track such places.
                return ValueOrPlace::TOP;
            }
            Rvalue::Repeat(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::Aggregate(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::WrapUnsafeBinder(..) => {
                // No modification is possible through these r-values.
                return ValueOrPlace::TOP;
            }
        };
        ValueOrPlace::Value(val)
    }

    fn handle_constant(
        &self,
        constant: &ConstOperand<'tcx>,
        _state: &mut State<FlatSet<Scalar>>,
    ) -> FlatSet<Scalar> {
        constant
            .const_
            .try_eval_scalar(self.tcx, self.typing_env)
            .map_or(FlatSet::Top, FlatSet::Elem)
    }

    fn handle_switch_int<'mir>(
        &self,
        discr: &'mir Operand<'tcx>,
        targets: &'mir SwitchTargets,
        state: &mut State<FlatSet<Scalar>>,
    ) -> TerminatorEdges<'mir, 'tcx> {
        let value = match self.handle_operand(discr, state) {
            ValueOrPlace::Value(value) => value,
            ValueOrPlace::Place(place) => state.get_idx(place, &self.map),
        };
        match value {
            // We are branching on uninitialized data, this is UB, treat it as unreachable.
            // This allows the set of visited edges to grow monotonically with the lattice.
            FlatSet::Bottom => TerminatorEdges::None,
            FlatSet::Elem(scalar) => {
                if let Ok(scalar_int) = scalar.try_to_scalar_int() {
                    TerminatorEdges::Single(
                        targets.target_for_value(scalar_int.to_bits_unchecked()),
                    )
                } else {
                    TerminatorEdges::SwitchInt { discr, targets }
                }
            }
            FlatSet::Top => TerminatorEdges::SwitchInt { discr, targets },
        }
    }

    /// The caller must have flooded `place`.
    fn assign_operand(
        &self,
        state: &mut State<FlatSet<Scalar>>,
        place: PlaceIndex,
        operand: &Operand<'tcx>,
    ) {
        match operand {
            Operand::Copy(rhs) | Operand::Move(rhs) => {
                if let Some(rhs) = self.map.find(rhs.as_ref()) {
                    state.insert_place_idx(place, rhs, &self.map);
                } else if rhs.projection.first() == Some(&PlaceElem::Deref)
                    && let FlatSet::Elem(pointer) = state.get(rhs.local.into(), &self.map)
                    && let rhs_ty = self.local_decls[rhs.local].ty
                    && let Ok(rhs_layout) =
                        self.tcx.layout_of(self.typing_env.as_query_input(rhs_ty))
                {
                    let op = ImmTy::from_scalar(pointer, rhs_layout).into();
                    self.assign_constant(state, place, op, rhs.projection);
                }
            }
            Operand::Constant(box constant) => {
                if let Some(constant) =
                    self.ecx.eval_mir_constant(&constant.const_, constant.span, None).discard_err()
                {
                    self.assign_constant(state, place, constant, &[]);
                }
            }
        }
    }

    /// The caller must have flooded `place`.
    ///
    /// Perform: `place = operand.projection`.
    #[instrument(level = "trace", skip(self, state))]
    fn assign_constant(
        &self,
        state: &mut State<FlatSet<Scalar>>,
        place: PlaceIndex,
        mut operand: OpTy<'tcx>,
        projection: &[PlaceElem<'tcx>],
    ) {
        for &(mut proj_elem) in projection {
            if let PlaceElem::Index(index) = proj_elem {
                if let FlatSet::Elem(index) = state.get(index.into(), &self.map)
                    && let Some(offset) = index.to_target_usize(&self.tcx).discard_err()
                    && let Some(min_length) = offset.checked_add(1)
                {
                    proj_elem = PlaceElem::ConstantIndex { offset, min_length, from_end: false };
                } else {
                    return;
                }
            }
            operand = if let Some(operand) = self.ecx.project(&operand, proj_elem).discard_err() {
                operand
            } else {
                return;
            }
        }

        self.map.for_each_projection_value(
            place,
            operand,
            &mut |elem, op| match elem {
                TrackElem::Field(idx) => self.ecx.project_field(op, idx).discard_err(),
                TrackElem::Variant(idx) => self.ecx.project_downcast(op, idx).discard_err(),
                TrackElem::Discriminant => {
                    let variant = self.ecx.read_discriminant(op).discard_err()?;
                    let discr_value =
                        self.ecx.discriminant_for_variant(op.layout.ty, variant).discard_err()?;
                    Some(discr_value.into())
                }
                TrackElem::DerefLen => {
                    let op: OpTy<'_> = self.ecx.deref_pointer(op).discard_err()?.into();
                    let len_usize = op.len(&self.ecx).discard_err()?;
                    let layout = self
                        .tcx
                        .layout_of(self.typing_env.as_query_input(self.tcx.types.usize))
                        .unwrap();
                    Some(ImmTy::from_uint(len_usize, layout).into())
                }
            },
            &mut |place, op| {
                if let Some(imm) = self.ecx.read_immediate_raw(op).discard_err()
                    && let Some(imm) = imm.right()
                {
                    let elem = self.wrap_immediate(*imm);
                    state.insert_value_idx(place, elem, &self.map);
                }
            },
        );
    }

    fn binary_op(
        &self,
        state: &mut State<FlatSet<Scalar>>,
        op: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
    ) -> (FlatSet<Scalar>, FlatSet<Scalar>) {
        let left = self.eval_operand(left, state);
        let right = self.eval_operand(right, state);

        match (left, right) {
            (FlatSet::Bottom, _) | (_, FlatSet::Bottom) => (FlatSet::Bottom, FlatSet::Bottom),
            // Both sides are known, do the actual computation.
            (FlatSet::Elem(left), FlatSet::Elem(right)) => {
                match self.ecx.binary_op(op, &left, &right).discard_err() {
                    // Ideally this would return an Immediate, since it's sometimes
                    // a pair and sometimes not. But as a hack we always return a pair
                    // and just make the 2nd component `Bottom` when it does not exist.
                    Some(val) => {
                        if matches!(val.layout.backend_repr, BackendRepr::ScalarPair(..)) {
                            let (val, overflow) = val.to_scalar_pair();
                            (FlatSet::Elem(val), FlatSet::Elem(overflow))
                        } else {
                            (FlatSet::Elem(val.to_scalar()), FlatSet::Bottom)
                        }
                    }
                    _ => (FlatSet::Top, FlatSet::Top),
                }
            }
            // Exactly one side is known, attempt some algebraic simplifications.
            (FlatSet::Elem(const_arg), _) | (_, FlatSet::Elem(const_arg)) => {
                let layout = const_arg.layout;
                if !matches!(layout.backend_repr, rustc_abi::BackendRepr::Scalar(..)) {
                    return (FlatSet::Top, FlatSet::Top);
                }

                let arg_scalar = const_arg.to_scalar();
                let Some(arg_value) = arg_scalar.to_bits(layout.size).discard_err() else {
                    return (FlatSet::Top, FlatSet::Top);
                };

                match op {
                    BinOp::BitAnd if arg_value == 0 => (FlatSet::Elem(arg_scalar), FlatSet::Bottom),
                    BinOp::BitOr
                        if arg_value == layout.size.truncate(u128::MAX)
                            || (layout.ty.is_bool() && arg_value == 1) =>
                    {
                        (FlatSet::Elem(arg_scalar), FlatSet::Bottom)
                    }
                    BinOp::Mul if layout.ty.is_integral() && arg_value == 0 => {
                        (FlatSet::Elem(arg_scalar), FlatSet::Elem(Scalar::from_bool(false)))
                    }
                    _ => (FlatSet::Top, FlatSet::Top),
                }
            }
            (FlatSet::Top, FlatSet::Top) => (FlatSet::Top, FlatSet::Top),
        }
    }

    fn eval_operand(
        &self,
        op: &Operand<'tcx>,
        state: &mut State<FlatSet<Scalar>>,
    ) -> FlatSet<ImmTy<'tcx>> {
        let value = match self.handle_operand(op, state) {
            ValueOrPlace::Value(value) => value,
            ValueOrPlace::Place(place) => state.get_idx(place, &self.map),
        };
        match value {
            FlatSet::Top => FlatSet::Top,
            FlatSet::Elem(scalar) => {
                let ty = op.ty(self.local_decls, self.tcx);
                self.tcx
                    .layout_of(self.typing_env.as_query_input(ty))
                    .map_or(FlatSet::Top, |layout| {
                        FlatSet::Elem(ImmTy::from_scalar(scalar, layout))
                    })
            }
            FlatSet::Bottom => FlatSet::Bottom,
        }
    }

    fn eval_discriminant(&self, enum_ty: Ty<'tcx>, variant_index: VariantIdx) -> Option<Scalar> {
        if !enum_ty.is_enum() {
            return None;
        }
        let enum_ty_layout = self.tcx.layout_of(self.typing_env.as_query_input(enum_ty)).ok()?;
        let discr_value =
            self.ecx.discriminant_for_variant(enum_ty_layout.ty, variant_index).discard_err()?;
        Some(discr_value.to_scalar())
    }

    fn wrap_immediate(&self, imm: Immediate) -> FlatSet<Scalar> {
        match imm {
            Immediate::Scalar(scalar) => FlatSet::Elem(scalar),
            Immediate::Uninit => FlatSet::Bottom,
            _ => FlatSet::Top,
        }
    }
}

/// This is used to visualize the dataflow analysis.
impl<'tcx> DebugWithContext<ConstAnalysis<'_, 'tcx>> for State<FlatSet<Scalar>> {
    fn fmt_with(&self, ctxt: &ConstAnalysis<'_, 'tcx>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            State::Reachable(values) => debug_with_context(values, None, &ctxt.map, f),
            State::Unreachable => write!(f, "unreachable"),
        }
    }

    fn fmt_diff_with(
        &self,
        old: &Self,
        ctxt: &ConstAnalysis<'_, 'tcx>,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        match (self, old) {
            (State::Reachable(this), State::Reachable(old)) => {
                debug_with_context(this, Some(old), &ctxt.map, f)
            }
            _ => Ok(()), // Consider printing something here.
        }
    }
}

struct Patch<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// For a given MIR location, this stores the values of the operands used by that location. In
    /// particular, this is before the effect, such that the operands of `_1 = _1 + _2` are
    /// properly captured. (This may become UB soon, but it is currently emitted even by safe code.)
    before_effect: FxHashMap<(Location, Place<'tcx>), Const<'tcx>>,

    /// Stores the assigned values for assignments where the Rvalue is constant.
    assignments: FxHashMap<Location, Const<'tcx>>,
}

impl<'tcx> Patch<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, before_effect: FxHashMap::default(), assignments: FxHashMap::default() }
    }

    fn make_operand(&self, const_: Const<'tcx>) -> Operand<'tcx> {
        Operand::Constant(Box::new(ConstOperand { span: DUMMY_SP, user_ty: None, const_ }))
    }
}

struct Collector<'a, 'tcx> {
    patch: Patch<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

impl<'a, 'tcx> Collector<'a, 'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, local_decls: &'a LocalDecls<'tcx>) -> Self {
        Self { patch: Patch::new(tcx), local_decls }
    }

    #[instrument(level = "trace", skip(self, ecx, map), ret)]
    fn try_make_constant(
        &self,
        ecx: &mut InterpCx<'tcx, DummyMachine>,
        place: Place<'tcx>,
        state: &State<FlatSet<Scalar>>,
        map: &Map<'tcx>,
    ) -> Option<Const<'tcx>> {
        let ty = place.ty(self.local_decls, self.patch.tcx).ty;
        let layout = ecx.layout_of(ty).ok()?;

        if layout.is_zst() {
            return Some(Const::zero_sized(ty));
        }

        if layout.is_unsized() {
            return None;
        }

        let place = map.find(place.as_ref())?;
        if layout.backend_repr.is_scalar()
            && let Some(value) = propagatable_scalar(place, state, map)
        {
            return Some(Const::Val(ConstValue::Scalar(value), ty));
        }

        if matches!(layout.backend_repr, BackendRepr::Scalar(..) | BackendRepr::ScalarPair(..)) {
            let alloc_id = ecx
                .intern_with_temp_alloc(layout, |ecx, dest| {
                    try_write_constant(ecx, dest, place, ty, state, map)
                })
                .discard_err()?;
            return Some(Const::Val(ConstValue::Indirect { alloc_id, offset: Size::ZERO }, ty));
        }

        None
    }
}

#[instrument(level = "trace", skip(map), ret)]
fn propagatable_scalar(
    place: PlaceIndex,
    state: &State<FlatSet<Scalar>>,
    map: &Map<'_>,
) -> Option<Scalar> {
    if let FlatSet::Elem(value) = state.get_idx(place, map)
        && value.try_to_scalar_int().is_ok()
    {
        // Do not attempt to propagate pointers, as we may fail to preserve their identity.
        Some(value)
    } else {
        None
    }
}

#[instrument(level = "trace", skip(ecx, state, map), ret)]
fn try_write_constant<'tcx>(
    ecx: &mut InterpCx<'tcx, DummyMachine>,
    dest: &PlaceTy<'tcx>,
    place: PlaceIndex,
    ty: Ty<'tcx>,
    state: &State<FlatSet<Scalar>>,
    map: &Map<'tcx>,
) -> InterpResult<'tcx> {
    let layout = ecx.layout_of(ty)?;

    // Fast path for ZSTs.
    if layout.is_zst() {
        return interp_ok(());
    }

    // Fast path for scalars.
    if layout.backend_repr.is_scalar()
        && let Some(value) = propagatable_scalar(place, state, map)
    {
        return ecx.write_immediate(Immediate::Scalar(value), dest);
    }

    match ty.kind() {
        // ZSTs. Nothing to do.
        ty::FnDef(..) => {}

        // Those are scalars, must be handled above.
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char =>
            throw_machine_stop_str!("primitive type with provenance"),

        ty::Tuple(elem_tys) => {
            for (i, elem) in elem_tys.iter().enumerate() {
                let i = FieldIdx::from_usize(i);
                let Some(field) = map.apply(place, TrackElem::Field(i)) else {
                    throw_machine_stop_str!("missing field in tuple")
                };
                let field_dest = ecx.project_field(dest, i)?;
                try_write_constant(ecx, &field_dest, field, elem, state, map)?;
            }
        }

        ty::Adt(def, args) => {
            if def.is_union() {
                throw_machine_stop_str!("cannot propagate unions")
            }

            let (variant_idx, variant_def, variant_place, variant_dest) = if def.is_enum() {
                let Some(discr) = map.apply(place, TrackElem::Discriminant) else {
                    throw_machine_stop_str!("missing discriminant for enum")
                };
                let FlatSet::Elem(Scalar::Int(discr)) = state.get_idx(discr, map) else {
                    throw_machine_stop_str!("discriminant with provenance")
                };
                let discr_bits = discr.to_bits(discr.size());
                let Some((variant, _)) = def.discriminants(*ecx.tcx).find(|(_, var)| discr_bits == var.val) else {
                    throw_machine_stop_str!("illegal discriminant for enum")
                };
                let Some(variant_place) = map.apply(place, TrackElem::Variant(variant)) else {
                    throw_machine_stop_str!("missing variant for enum")
                };
                let variant_dest = ecx.project_downcast(dest, variant)?;
                (variant, def.variant(variant), variant_place, variant_dest)
            } else {
                (FIRST_VARIANT, def.non_enum_variant(), place, dest.clone())
            };

            for (i, field) in variant_def.fields.iter_enumerated() {
                let ty = field.ty(*ecx.tcx, args);
                let Some(field) = map.apply(variant_place, TrackElem::Field(i)) else {
                    throw_machine_stop_str!("missing field in ADT")
                };
                let field_dest = ecx.project_field(&variant_dest, i)?;
                try_write_constant(ecx, &field_dest, field, ty, state, map)?;
            }
            ecx.write_discriminant(variant_idx, dest)?;
        }

        // Unsupported for now.
        ty::Array(_, _)
        | ty::Pat(_, _)

        // Do not attempt to support indirection in constants.
        | ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::Str | ty::Slice(_)

        | ty::Never
        | ty::Foreign(..)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::Dynamic(..)
        | ty::UnsafeBinder(_) => throw_machine_stop_str!("unsupported type"),

        ty::Error(_) | ty::Infer(..) | ty::CoroutineWitness(..) => bug!(),
    }

    interp_ok(())
}

impl<'tcx> ResultsVisitor<'tcx, ConstAnalysis<'_, 'tcx>> for Collector<'_, 'tcx> {
    #[instrument(level = "trace", skip(self, analysis, statement))]
    fn visit_after_early_statement_effect(
        &mut self,
        analysis: &mut ConstAnalysis<'_, 'tcx>,
        state: &State<FlatSet<Scalar>>,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        match &statement.kind {
            StatementKind::Assign(box (_, rvalue)) => {
                OperandCollector {
                    state,
                    visitor: self,
                    ecx: &mut analysis.ecx,
                    map: &analysis.map,
                }
                .visit_rvalue(rvalue, location);
            }
            _ => (),
        }
    }

    #[instrument(level = "trace", skip(self, analysis, statement))]
    fn visit_after_primary_statement_effect(
        &mut self,
        analysis: &mut ConstAnalysis<'_, 'tcx>,
        state: &State<FlatSet<Scalar>>,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        match statement.kind {
            StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(_)))) => {
                // Don't overwrite the assignment if it already uses a constant (to keep the span).
            }
            StatementKind::Assign(box (place, _)) => {
                if let Some(value) =
                    self.try_make_constant(&mut analysis.ecx, place, state, &analysis.map)
                {
                    self.patch.assignments.insert(location, value);
                }
            }
            _ => (),
        }
    }

    fn visit_after_early_terminator_effect(
        &mut self,
        analysis: &mut ConstAnalysis<'_, 'tcx>,
        state: &State<FlatSet<Scalar>>,
        terminator: &Terminator<'tcx>,
        location: Location,
    ) {
        OperandCollector { state, visitor: self, ecx: &mut analysis.ecx, map: &analysis.map }
            .visit_terminator(terminator, location);
    }
}

impl<'tcx> MutVisitor<'tcx> for Patch<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let Some(value) = self.assignments.get(&location) {
            match &mut statement.kind {
                StatementKind::Assign(box (_, rvalue)) => {
                    *rvalue = Rvalue::Use(self.make_operand(*value));
                }
                _ => bug!("found assignment info for non-assign statement"),
            }
        } else {
            self.super_statement(statement, location);
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                if let Some(value) = self.before_effect.get(&(location, *place)) {
                    *operand = self.make_operand(*value);
                } else if !place.projection.is_empty() {
                    self.super_operand(operand, location)
                }
            }
            Operand::Constant(_) => {}
        }
    }

    fn process_projection_elem(
        &mut self,
        elem: PlaceElem<'tcx>,
        location: Location,
    ) -> Option<PlaceElem<'tcx>> {
        if let PlaceElem::Index(local) = elem {
            let offset = self.before_effect.get(&(location, local.into()))?;
            let offset = offset.try_to_scalar()?;
            let offset = offset.to_target_usize(&self.tcx).discard_err()?;
            let min_length = offset.checked_add(1)?;
            Some(PlaceElem::ConstantIndex { offset, min_length, from_end: false })
        } else {
            None
        }
    }
}

struct OperandCollector<'a, 'b, 'tcx> {
    state: &'a State<FlatSet<Scalar>>,
    visitor: &'a mut Collector<'b, 'tcx>,
    ecx: &'a mut InterpCx<'tcx, DummyMachine>,
    map: &'a Map<'tcx>,
}

impl<'tcx> Visitor<'tcx> for OperandCollector<'_, '_, 'tcx> {
    fn visit_projection_elem(
        &mut self,
        _: PlaceRef<'tcx>,
        elem: PlaceElem<'tcx>,
        _: PlaceContext,
        location: Location,
    ) {
        if let PlaceElem::Index(local) = elem
            && let Some(value) =
                self.visitor.try_make_constant(self.ecx, local.into(), self.state, self.map)
        {
            self.visitor.patch.before_effect.insert((location, local.into()), value);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        if let Some(place) = operand.place() {
            if let Some(value) =
                self.visitor.try_make_constant(self.ecx, place, self.state, self.map)
            {
                self.visitor.patch.before_effect.insert((location, place), value);
            } else if !place.projection.is_empty() {
                // Try to propagate into `Index` projections.
                self.super_operand(operand, location)
            }
        }
    }
}
