//! A constant propagation optimization pass based on dataflow analysis.
//!
//! Currently, this pass only propagates scalar values.

use rustc_const_eval::const_eval::CheckAlignment;
use rustc_const_eval::interpret::{ConstValue, ImmTy, Immediate, InterpCx, Scalar};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_middle::mir::visit::{MutVisitor, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::value_analysis::{
    Map, State, TrackElem, ValueAnalysis, ValueAnalysisWrapper, ValueOrPlace,
};
use rustc_mir_dataflow::{
    lattice::FlatSet, Analysis, Results, ResultsVisitor, SwitchIntEdgeEffects,
};
use rustc_span::DUMMY_SP;
use rustc_target::abi::{Align, FieldIdx, VariantIdx};

use crate::MirPass;

// These constants are somewhat random guesses and have not been optimized.
// If `tcx.sess.mir_opt_level() >= 4`, we ignore the limits (this can become very expensive).
const BLOCK_LIMIT: usize = 100;
const PLACE_LIMIT: usize = 100;

pub struct DataflowConstProp;

impl<'tcx> MirPass<'tcx> for DataflowConstProp {
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
        let map = Map::from_filter(tcx, body, Ty::is_scalar, place_limit);

        // Perform the actual dataflow analysis.
        let analysis = ConstAnalysis::new(tcx, body, map);
        let mut results = debug_span!("analyze")
            .in_scope(|| analysis.wrap().into_engine(tcx, body).iterate_to_fixpoint());

        // Collect results and patch the body afterwards.
        let mut visitor = CollectAndPatch::new(tcx);
        debug_span!("collect").in_scope(|| results.visit_reachable_with(body, &mut visitor));
        debug_span!("patch").in_scope(|| visitor.visit_body(body));
    }
}

struct ConstAnalysis<'a, 'tcx> {
    map: Map,
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    ecx: InterpCx<'tcx, 'tcx, DummyMachine>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> ValueAnalysis<'tcx> for ConstAnalysis<'_, 'tcx> {
    type Value = FlatSet<ScalarTy<'tcx>>;

    const NAME: &'static str = "ConstAnalysis";

    fn map(&self) -> &Map {
        &self.map
    }

    fn handle_set_discriminant(
        &self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
        state: &mut State<Self::Value>,
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
        state: &mut State<Self::Value>,
    ) {
        match rvalue {
            Rvalue::Aggregate(kind, operands) => {
                // If we assign `target = Enum::Variant#0(operand)`,
                // we must make sure that all `target as Variant#i` are `Top`.
                state.flood(target.as_ref(), self.map());

                let Some(target_idx) = self.map().find(target.as_ref()) else { return };

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
                    for (field_index, operand) in operands.iter().enumerate() {
                        if let Some(field) = self.map().apply(
                            variant_target_idx,
                            TrackElem::Field(FieldIdx::from_usize(field_index)),
                        ) {
                            let result = self.handle_operand(operand, state);
                            state.insert_idx(field, result, self.map());
                        }
                    }
                }
                if let Some(variant_index) = variant_index
                    && let Some(discr_idx) = self.map().apply(target_idx, TrackElem::Discriminant)
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
            Rvalue::CheckedBinaryOp(op, box (left, right)) => {
                // Flood everything now, so we can use `insert_value_idx` directly later.
                state.flood(target.as_ref(), self.map());

                let Some(target) = self.map().find(target.as_ref()) else { return };

                let value_target = self.map().apply(target, TrackElem::Field(0_u32.into()));
                let overflow_target = self.map().apply(target, TrackElem::Field(1_u32.into()));

                if value_target.is_some() || overflow_target.is_some() {
                    let (val, overflow) = self.binary_op(state, *op, left, right);

                    if let Some(value_target) = value_target {
                        // We have flooded `target` earlier.
                        state.insert_value_idx(value_target, val, self.map());
                    }
                    if let Some(overflow_target) = overflow_target {
                        let overflow = match overflow {
                            FlatSet::Top => FlatSet::Top,
                            FlatSet::Elem(overflow) => {
                                self.wrap_scalar(Scalar::from_bool(overflow), self.tcx.types.bool)
                            }
                            FlatSet::Bottom => FlatSet::Bottom,
                        };
                        // We have flooded `target` earlier.
                        state.insert_value_idx(overflow_target, overflow, self.map());
                    }
                }
            }
            _ => self.super_assign(target, rvalue, state),
        }
    }

    fn handle_rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlace<Self::Value> {
        match rvalue {
            Rvalue::Cast(
                kind @ (CastKind::IntToInt
                | CastKind::FloatToInt
                | CastKind::FloatToFloat
                | CastKind::IntToFloat),
                operand,
                ty,
            ) => match self.eval_operand(operand, state) {
                FlatSet::Elem(op) => match kind {
                    CastKind::IntToInt | CastKind::IntToFloat => {
                        self.ecx.int_to_int_or_float(&op, *ty)
                    }
                    CastKind::FloatToInt | CastKind::FloatToFloat => {
                        self.ecx.float_to_float_or_int(&op, *ty)
                    }
                    _ => unreachable!(),
                }
                .map(|result| ValueOrPlace::Value(self.wrap_immediate(result, *ty)))
                .unwrap_or(ValueOrPlace::TOP),
                _ => ValueOrPlace::TOP,
            },
            Rvalue::BinaryOp(op, box (left, right)) => {
                // Overflows must be ignored here.
                let (val, _overflow) = self.binary_op(state, *op, left, right);
                ValueOrPlace::Value(val)
            }
            Rvalue::UnaryOp(op, operand) => match self.eval_operand(operand, state) {
                FlatSet::Elem(value) => self
                    .ecx
                    .unary_op(*op, &value)
                    .map(|val| ValueOrPlace::Value(self.wrap_immty(val)))
                    .unwrap_or(ValueOrPlace::Value(FlatSet::Top)),
                FlatSet::Bottom => ValueOrPlace::Value(FlatSet::Bottom),
                FlatSet::Top => ValueOrPlace::Value(FlatSet::Top),
            },
            Rvalue::Discriminant(place) => {
                ValueOrPlace::Value(state.get_discr(place.as_ref(), self.map()))
            }
            _ => self.super_rvalue(rvalue, state),
        }
    }

    fn handle_constant(
        &self,
        constant: &Constant<'tcx>,
        _state: &mut State<Self::Value>,
    ) -> Self::Value {
        constant
            .literal
            .eval(self.tcx, self.param_env)
            .try_to_scalar()
            .map(|value| FlatSet::Elem(ScalarTy(value, constant.ty())))
            .unwrap_or(FlatSet::Top)
    }

    fn handle_switch_int(
        &self,
        discr: &Operand<'tcx>,
        apply_edge_effects: &mut impl SwitchIntEdgeEffects<State<Self::Value>>,
    ) {
        // FIXME: The dataflow framework only provides the state if we call `apply()`, which makes
        // this more inefficient than it has to be.
        let mut discr_value = None;
        let mut handled = false;
        apply_edge_effects.apply(|state, target| {
            let discr_value = match discr_value {
                Some(value) => value,
                None => {
                    let value = match self.handle_operand(discr, state) {
                        ValueOrPlace::Value(value) => value,
                        ValueOrPlace::Place(place) => state.get_idx(place, self.map()),
                    };
                    let result = match value {
                        FlatSet::Top => FlatSet::Top,
                        FlatSet::Elem(ScalarTy(scalar, _)) => {
                            let int = scalar.assert_int();
                            FlatSet::Elem(int.assert_bits(int.size()))
                        }
                        FlatSet::Bottom => FlatSet::Bottom,
                    };
                    discr_value = Some(result);
                    result
                }
            };

            let FlatSet::Elem(choice) = discr_value else {
                // Do nothing if we don't know which branch will be taken.
                return
            };

            if target.value.map(|n| n == choice).unwrap_or(!handled) {
                // Branch is taken. Has no effect on state.
                handled = true;
            } else {
                // Branch is not taken.
                state.mark_unreachable();
            }
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct ScalarTy<'tcx>(Scalar, Ty<'tcx>);

impl<'tcx> std::fmt::Debug for ScalarTy<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // This is used for dataflow visualization, so we return something more concise.
        std::fmt::Display::fmt(&ConstantKind::Val(ConstValue::Scalar(self.0), self.1), f)
    }
}

impl<'a, 'tcx> ConstAnalysis<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, map: Map) -> Self {
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        Self {
            map,
            tcx,
            local_decls: &body.local_decls,
            ecx: InterpCx::new(tcx, DUMMY_SP, param_env, DummyMachine),
            param_env: param_env,
        }
    }

    fn binary_op(
        &self,
        state: &mut State<FlatSet<ScalarTy<'tcx>>>,
        op: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
    ) -> (FlatSet<ScalarTy<'tcx>>, FlatSet<bool>) {
        let left = self.eval_operand(left, state);
        let right = self.eval_operand(right, state);
        match (left, right) {
            (FlatSet::Elem(left), FlatSet::Elem(right)) => {
                match self.ecx.overflowing_binary_op(op, &left, &right) {
                    Ok((val, overflow, ty)) => (self.wrap_scalar(val, ty), FlatSet::Elem(overflow)),
                    _ => (FlatSet::Top, FlatSet::Top),
                }
            }
            (FlatSet::Bottom, _) | (_, FlatSet::Bottom) => (FlatSet::Bottom, FlatSet::Bottom),
            (_, _) => {
                // Could attempt some algebraic simplifications here.
                (FlatSet::Top, FlatSet::Top)
            }
        }
    }

    fn eval_operand(
        &self,
        op: &Operand<'tcx>,
        state: &mut State<FlatSet<ScalarTy<'tcx>>>,
    ) -> FlatSet<ImmTy<'tcx>> {
        let value = match self.handle_operand(op, state) {
            ValueOrPlace::Value(value) => value,
            ValueOrPlace::Place(place) => state.get_idx(place, &self.map),
        };
        match value {
            FlatSet::Top => FlatSet::Top,
            FlatSet::Elem(ScalarTy(scalar, ty)) => self
                .tcx
                .layout_of(self.param_env.and(ty))
                .map(|layout| FlatSet::Elem(ImmTy::from_scalar(scalar, layout)))
                .unwrap_or(FlatSet::Top),
            FlatSet::Bottom => FlatSet::Bottom,
        }
    }

    fn eval_discriminant(
        &self,
        enum_ty: Ty<'tcx>,
        variant_index: VariantIdx,
    ) -> Option<ScalarTy<'tcx>> {
        if !enum_ty.is_enum() {
            return None;
        }
        let discr = enum_ty.discriminant_for_variant(self.tcx, variant_index)?;
        let discr_layout = self.tcx.layout_of(self.param_env.and(discr.ty)).ok()?;
        let discr_value = Scalar::try_from_uint(discr.val, discr_layout.size)?;
        Some(ScalarTy(discr_value, discr.ty))
    }

    fn wrap_scalar(&self, scalar: Scalar, ty: Ty<'tcx>) -> FlatSet<ScalarTy<'tcx>> {
        FlatSet::Elem(ScalarTy(scalar, ty))
    }

    fn wrap_immediate(&self, imm: Immediate, ty: Ty<'tcx>) -> FlatSet<ScalarTy<'tcx>> {
        match imm {
            Immediate::Scalar(scalar) => self.wrap_scalar(scalar, ty),
            _ => FlatSet::Top,
        }
    }

    fn wrap_immty(&self, val: ImmTy<'tcx>) -> FlatSet<ScalarTy<'tcx>> {
        self.wrap_immediate(*val, val.layout.ty)
    }
}

struct CollectAndPatch<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// For a given MIR location, this stores the values of the operands used by that location. In
    /// particular, this is before the effect, such that the operands of `_1 = _1 + _2` are
    /// properly captured. (This may become UB soon, but it is currently emitted even by safe code.)
    before_effect: FxHashMap<(Location, Place<'tcx>), ScalarTy<'tcx>>,

    /// Stores the assigned values for assignments where the Rvalue is constant.
    assignments: FxHashMap<Location, ScalarTy<'tcx>>,
}

impl<'tcx> CollectAndPatch<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, before_effect: FxHashMap::default(), assignments: FxHashMap::default() }
    }

    fn make_operand(&self, scalar: ScalarTy<'tcx>) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span: DUMMY_SP,
            user_ty: None,
            literal: ConstantKind::Val(ConstValue::Scalar(scalar.0), scalar.1),
        }))
    }
}

impl<'mir, 'tcx>
    ResultsVisitor<'mir, 'tcx, Results<'tcx, ValueAnalysisWrapper<ConstAnalysis<'_, 'tcx>>>>
    for CollectAndPatch<'tcx>
{
    type FlowState = State<FlatSet<ScalarTy<'tcx>>>;

    fn visit_statement_before_primary_effect(
        &mut self,
        results: &Results<'tcx, ValueAnalysisWrapper<ConstAnalysis<'_, 'tcx>>>,
        state: &Self::FlowState,
        statement: &'mir Statement<'tcx>,
        location: Location,
    ) {
        match &statement.kind {
            StatementKind::Assign(box (_, rvalue)) => {
                OperandCollector { state, visitor: self, map: &results.analysis.0.map }
                    .visit_rvalue(rvalue, location);
            }
            _ => (),
        }
    }

    fn visit_statement_after_primary_effect(
        &mut self,
        results: &Results<'tcx, ValueAnalysisWrapper<ConstAnalysis<'_, 'tcx>>>,
        state: &Self::FlowState,
        statement: &'mir Statement<'tcx>,
        location: Location,
    ) {
        match statement.kind {
            StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(_)))) => {
                // Don't overwrite the assignment if it already uses a constant (to keep the span).
            }
            StatementKind::Assign(box (place, _)) => {
                match state.get(place.as_ref(), &results.analysis.0.map) {
                    FlatSet::Top => (),
                    FlatSet::Elem(value) => {
                        self.assignments.insert(location, value);
                    }
                    FlatSet::Bottom => {
                        // This assignment is either unreachable, or an uninitialized value is assigned.
                    }
                }
            }
            _ => (),
        }
    }

    fn visit_terminator_before_primary_effect(
        &mut self,
        results: &Results<'tcx, ValueAnalysisWrapper<ConstAnalysis<'_, 'tcx>>>,
        state: &Self::FlowState,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) {
        OperandCollector { state, visitor: self, map: &results.analysis.0.map }
            .visit_terminator(terminator, location);
    }
}

impl<'tcx> MutVisitor<'tcx> for CollectAndPatch<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let Some(value) = self.assignments.get(&location) {
            match &mut statement.kind {
                StatementKind::Assign(box (_, rvalue)) => {
                    *rvalue = Rvalue::Use(self.make_operand(value.clone()));
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
                    *operand = self.make_operand(value.clone());
                }
            }
            _ => (),
        }
    }
}

struct OperandCollector<'tcx, 'map, 'a> {
    state: &'a State<FlatSet<ScalarTy<'tcx>>>,
    visitor: &'a mut CollectAndPatch<'tcx>,
    map: &'map Map,
}

impl<'tcx, 'map, 'a> Visitor<'tcx> for OperandCollector<'tcx, 'map, 'a> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                match self.state.get(place.as_ref(), self.map) {
                    FlatSet::Top => (),
                    FlatSet::Elem(value) => {
                        self.visitor.before_effect.insert((location, *place), value);
                    }
                    FlatSet::Bottom => (),
                }
            }
            _ => (),
        }
    }
}

struct DummyMachine;

impl<'mir, 'tcx> rustc_const_eval::interpret::Machine<'mir, 'tcx> for DummyMachine {
    rustc_const_eval::interpret::compile_time_machine!(<'mir, 'tcx>);
    type MemoryKind = !;
    const PANIC_ON_ALLOC_FAIL: bool = true;

    fn enforce_alignment(_ecx: &InterpCx<'mir, 'tcx, Self>) -> CheckAlignment {
        unimplemented!()
    }

    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>, _layout: TyAndLayout<'tcx>) -> bool {
        unimplemented!()
    }
    fn alignment_check_failed(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _has: Align,
        _required: Align,
        _check: CheckAlignment,
    ) -> interpret::InterpResult<'tcx, ()> {
        unimplemented!()
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _abi: rustc_target::spec::abi::Abi,
        _args: &[rustc_const_eval::interpret::OpTy<'tcx, Self::Provenance>],
        _destination: &rustc_const_eval::interpret::PlaceTy<'tcx, Self::Provenance>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx, Option<(&'mir Body<'tcx>, ty::Instance<'tcx>)>> {
        unimplemented!()
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[rustc_const_eval::interpret::OpTy<'tcx, Self::Provenance>],
        _destination: &rustc_const_eval::interpret::PlaceTy<'tcx, Self::Provenance>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: &rustc_const_eval::interpret::ImmTy<'tcx, Self::Provenance>,
        _right: &rustc_const_eval::interpret::ImmTy<'tcx, Self::Provenance>,
    ) -> interpret::InterpResult<'tcx, (interpret::Scalar<Self::Provenance>, bool, Ty<'tcx>)> {
        throw_unsup!(Unsupported("".into()))
    }

    fn expose_ptr(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _ptr: interpret::Pointer<Self::Provenance>,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn init_frame_extra(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _frame: rustc_const_eval::interpret::Frame<'mir, 'tcx, Self::Provenance>,
    ) -> interpret::InterpResult<
        'tcx,
        rustc_const_eval::interpret::Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>,
    > {
        unimplemented!()
    }

    fn stack<'a>(
        _ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [rustc_const_eval::interpret::Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>]
    {
        unimplemented!()
    }

    fn stack_mut<'a>(
        _ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<
        rustc_const_eval::interpret::Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>,
    > {
        unimplemented!()
    }
}
