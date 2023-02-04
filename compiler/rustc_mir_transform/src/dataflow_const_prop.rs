//! A constant propagation optimization pass based on dataflow analysis.
//!
//! Currently, this pass only propagates scalar values.

use rustc_const_eval::const_eval::CheckAlignment;
use rustc_const_eval::interpret::{ConstValue, ImmTy, Immediate, InterpCx, Scalar};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_middle::mir::visit::{MutVisitor, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::value_analysis::{Map, State, TrackElem, ValueAnalysis, ValueOrPlace};
use rustc_mir_dataflow::{lattice::FlatSet, Analysis, ResultsVisitor, SwitchIntEdgeEffects};
use rustc_span::DUMMY_SP;
use rustc_target::abi::Align;

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
        if tcx.sess.mir_opt_level() < 4 && body.basic_blocks.len() > BLOCK_LIMIT {
            debug!("aborted dataflow const prop due too many basic blocks");
            return;
        }

        // Decide which places to track during the analysis.
        let map = Map::from_filter(tcx, body, Ty::is_scalar);

        // We want to have a somewhat linear runtime w.r.t. the number of statements/terminators.
        // Let's call this number `n`. Dataflow analysis has `O(h*n)` transfer function
        // applications, where `h` is the height of the lattice. Because the height of our lattice
        // is linear w.r.t. the number of tracked places, this is `O(tracked_places * n)`. However,
        // because every transfer function application could traverse the whole map, this becomes
        // `O(num_nodes * tracked_places * n)` in terms of time complexity. Since the number of
        // map nodes is strongly correlated to the number of tracked places, this becomes more or
        // less `O(n)` if we place a constant limit on the number of tracked places.
        if tcx.sess.mir_opt_level() < 4 && map.tracked_places() > PLACE_LIMIT {
            debug!("aborted dataflow const prop due to too many tracked places");
            return;
        }

        // Perform the actual dataflow analysis.
        let analysis = ConstAnalysis::new(tcx, body, map);
        let results = debug_span!("analyze")
            .in_scope(|| analysis.wrap().into_engine(tcx, body).iterate_to_fixpoint());

        // Collect results and patch the body afterwards.
        let mut visitor = CollectAndPatch::new(tcx, &results.analysis.0.map);
        debug_span!("collect").in_scope(|| results.visit_reachable_with(body, &mut visitor));
        debug_span!("patch").in_scope(|| visitor.visit_body(body));
    }
}

struct ConstAnalysis<'tcx> {
    map: Map,
    tcx: TyCtxt<'tcx>,
    ecx: InterpCx<'tcx, 'tcx, DummyMachine>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> ValueAnalysis<'tcx> for ConstAnalysis<'tcx> {
    type Value = FlatSet<ScalarTy<'tcx>>;

    const NAME: &'static str = "ConstAnalysis";

    fn map(&self) -> &Map {
        &self.map
    }

    fn handle_assign(
        &self,
        target: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) {
        match rvalue {
            Rvalue::Aggregate(kind, operands) => {
                let target = self.map().find(target.as_ref());
                if let Some(target) = target {
                    state.flood_idx_with(target, self.map(), FlatSet::Bottom);
                    let field_based = match **kind {
                        AggregateKind::Tuple | AggregateKind::Closure(..) => true,
                        AggregateKind::Adt(def_id, ..) => {
                            matches!(self.tcx.def_kind(def_id), DefKind::Struct)
                        }
                        _ => false,
                    };
                    if field_based {
                        for (field_index, operand) in operands.iter().enumerate() {
                            if let Some(field) = self
                                .map()
                                .apply(target, TrackElem::Field(Field::from_usize(field_index)))
                            {
                                let result = self.handle_operand(operand, state);
                                state.assign_idx(field, result, self.map());
                            }
                        }
                    }
                }
            }
            Rvalue::CheckedBinaryOp(op, box (left, right)) => {
                let target = self.map().find(target.as_ref());
                if let Some(target) = target {
                    // We should not track any projections other than
                    // what is overwritten below, but just in case...
                    state.flood_idx(target, self.map());
                }

                let value_target = target
                    .and_then(|target| self.map().apply(target, TrackElem::Field(0_u32.into())));
                let overflow_target = target
                    .and_then(|target| self.map().apply(target, TrackElem::Field(1_u32.into())));

                if value_target.is_some() || overflow_target.is_some() {
                    let (val, overflow) = self.binary_op(state, *op, left, right);

                    if let Some(value_target) = value_target {
                        state.assign_idx(value_target, ValueOrPlace::Value(val), self.map());
                    }
                    if let Some(overflow_target) = overflow_target {
                        let overflow = match overflow {
                            FlatSet::Top => FlatSet::Top,
                            FlatSet::Elem(overflow) => {
                                if overflow {
                                    // Overflow cannot be reliably propagated. See: https://github.com/rust-lang/rust/pull/101168#issuecomment-1288091446
                                    FlatSet::Top
                                } else {
                                    self.wrap_scalar(Scalar::from_bool(false), self.tcx.types.bool)
                                }
                            }
                            FlatSet::Bottom => FlatSet::Bottom,
                        };
                        state.assign_idx(
                            overflow_target,
                            ValueOrPlace::Value(overflow),
                            self.map(),
                        );
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
                .unwrap_or(ValueOrPlace::top()),
                _ => ValueOrPlace::top(),
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

impl<'tcx> ConstAnalysis<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, map: Map) -> Self {
        let param_env = tcx.param_env(body.source.def_id());
        Self {
            map,
            tcx,
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
                // Could attempt some algebraic simplifcations here.
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

struct CollectAndPatch<'tcx, 'map> {
    tcx: TyCtxt<'tcx>,
    map: &'map Map,

    /// For a given MIR location, this stores the values of the operands used by that location. In
    /// particular, this is before the effect, such that the operands of `_1 = _1 + _2` are
    /// properly captured. (This may become UB soon, but it is currently emitted even by safe code.)
    before_effect: FxHashMap<(Location, Place<'tcx>), ScalarTy<'tcx>>,

    /// Stores the assigned values for assignments where the Rvalue is constant.
    assignments: FxHashMap<Location, ScalarTy<'tcx>>,
}

impl<'tcx, 'map> CollectAndPatch<'tcx, 'map> {
    fn new(tcx: TyCtxt<'tcx>, map: &'map Map) -> Self {
        Self { tcx, map, before_effect: FxHashMap::default(), assignments: FxHashMap::default() }
    }

    fn make_operand(&self, scalar: ScalarTy<'tcx>) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span: DUMMY_SP,
            user_ty: None,
            literal: ConstantKind::Val(ConstValue::Scalar(scalar.0), scalar.1),
        }))
    }
}

impl<'mir, 'tcx, 'map> ResultsVisitor<'mir, 'tcx> for CollectAndPatch<'tcx, 'map> {
    type FlowState = State<FlatSet<ScalarTy<'tcx>>>;

    fn visit_statement_before_primary_effect(
        &mut self,
        state: &Self::FlowState,
        statement: &'mir Statement<'tcx>,
        location: Location,
    ) {
        match &statement.kind {
            StatementKind::Assign(box (_, rvalue)) => {
                OperandCollector { state, visitor: self }.visit_rvalue(rvalue, location);
            }
            _ => (),
        }
    }

    fn visit_statement_after_primary_effect(
        &mut self,
        state: &Self::FlowState,
        statement: &'mir Statement<'tcx>,
        location: Location,
    ) {
        match statement.kind {
            StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(_)))) => {
                // Don't overwrite the assignment if it already uses a constant (to keep the span).
            }
            StatementKind::Assign(box (place, _)) => match state.get(place.as_ref(), self.map) {
                FlatSet::Top => (),
                FlatSet::Elem(value) => {
                    self.assignments.insert(location, value);
                }
                FlatSet::Bottom => {
                    // This assignment is either unreachable, or an uninitialized value is assigned.
                }
            },
            _ => (),
        }
    }

    fn visit_terminator_before_primary_effect(
        &mut self,
        state: &Self::FlowState,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) {
        OperandCollector { state, visitor: self }.visit_terminator(terminator, location);
    }
}

impl<'tcx, 'map> MutVisitor<'tcx> for CollectAndPatch<'tcx, 'map> {
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
    visitor: &'a mut CollectAndPatch<'tcx, 'map>,
}

impl<'tcx, 'map, 'a> Visitor<'tcx> for OperandCollector<'tcx, 'map, 'a> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                match self.state.get(place.as_ref(), self.visitor.map) {
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

    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
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
        _unwind: rustc_const_eval::interpret::StackPopUnwind,
    ) -> interpret::InterpResult<'tcx, Option<(&'mir Body<'tcx>, ty::Instance<'tcx>)>> {
        unimplemented!()
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[rustc_const_eval::interpret::OpTy<'tcx, Self::Provenance>],
        _destination: &rustc_const_eval::interpret::PlaceTy<'tcx, Self::Provenance>,
        _target: Option<BasicBlock>,
        _unwind: rustc_const_eval::interpret::StackPopUnwind,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: Option<BasicBlock>,
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
