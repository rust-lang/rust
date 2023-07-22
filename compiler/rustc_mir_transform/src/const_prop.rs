//! Propagates constants for early reporting of statically known
//! assertion failures

use either::Right;

use rustc_const_eval::const_eval::CheckAlignment;
use rustc_const_eval::ReportErrorExt;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::BitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{LayoutError, LayoutOf, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{self, GenericArgs, Instance, ParamEnv, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::{def_id::DefId, Span, DUMMY_SP};
use rustc_target::abi::{self, Align, HasDataLayout, Size, TargetDataLayout};
use rustc_target::spec::abi::Abi as CallAbi;

use crate::MirPass;
use rustc_const_eval::interpret::{
    self, compile_time_machine, AllocId, ConstAllocation, ConstValue, FnArg, Frame, ImmTy,
    Immediate, InterpCx, InterpResult, LocalValue, MemoryKind, OpTy, PlaceTy, Pointer, Scalar,
    StackPopCleanup,
};

/// The maximum number of bytes that we'll allocate space for a local or the return value.
/// Needed for #66397, because otherwise we eval into large places and that can cause OOM or just
/// Severely regress performance.
const MAX_ALLOC_LIMIT: u64 = 1024;

/// Macro for machine-specific `InterpError` without allocation.
/// (These will never be shown to the user, but they help diagnose ICEs.)
macro_rules! throw_machine_stop_str {
    ($($tt:tt)*) => {{
        // We make a new local type for it. The type itself does not carry any information,
        // but its vtable (for the `MachineStopType` trait) does.
        #[derive(Debug)]
        struct Zst;
        // Printing this type shows the desired string.
        impl std::fmt::Display for Zst {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $($tt)*)
            }
        }

        impl rustc_middle::mir::interpret::MachineStopType for Zst {
            fn diagnostic_message(&self) -> rustc_errors::DiagnosticMessage {
                self.to_string().into()
            }

            fn add_args(
                self: Box<Self>,
                _: &mut dyn FnMut(std::borrow::Cow<'static, str>, rustc_errors::DiagnosticArgValue<'static>),
            ) {}
        }
        throw_machine_stop!(Zst)
    }};
}

pub struct ConstProp;

impl<'tcx> MirPass<'tcx> for ConstProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[instrument(skip(self, tcx), level = "debug")]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if body.source.promoted.is_some() {
            return;
        }

        let def_id = body.source.def_id().expect_local();
        let def_kind = tcx.def_kind(def_id);
        let is_fn_like = def_kind.is_fn_like();
        let is_assoc_const = def_kind == DefKind::AssocConst;

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("ConstProp skipped for {:?}", def_id);
            return;
        }

        let is_generator = tcx.type_of(def_id.to_def_id()).instantiate_identity().is_generator();
        // FIXME(welseywiser) const prop doesn't work on generators because of query cycles
        // computing their layout.
        if is_generator {
            trace!("ConstProp skipped for generator {:?}", def_id);
            return;
        }

        trace!("ConstProp starting for {:?}", def_id);

        let dummy_body = &Body::new(
            body.source,
            (*body.basic_blocks).to_owned(),
            body.source_scopes.clone(),
            body.local_decls.clone(),
            Default::default(),
            body.arg_count,
            Default::default(),
            body.span,
            body.generator_kind(),
            body.tainted_by_errors,
        );

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require a uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder = ConstPropagator::new(body, dummy_body, tcx);

        // Traverse the body in reverse post-order, to ensure that `FullConstProp` locals are
        // assigned before being read.
        let rpo = body.basic_blocks.reverse_postorder().to_vec();
        for bb in rpo {
            let data = &mut body.basic_blocks.as_mut_preserves_cfg()[bb];
            optimization_finder.visit_basic_block_data(bb, data);
        }

        trace!("ConstProp done for {:?}", def_id);
    }
}

pub struct ConstPropMachine<'mir, 'tcx> {
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx>>,
    pub written_only_inside_own_block_locals: FxHashSet<Local>,
    pub can_const_prop: IndexVec<Local, ConstPropMode>,
}

impl ConstPropMachine<'_, '_> {
    pub fn new(can_const_prop: IndexVec<Local, ConstPropMode>) -> Self {
        Self {
            stack: Vec::new(),
            written_only_inside_own_block_locals: Default::default(),
            can_const_prop,
        }
    }
}

impl<'mir, 'tcx> interpret::Machine<'mir, 'tcx> for ConstPropMachine<'mir, 'tcx> {
    compile_time_machine!(<'mir, 'tcx>);
    const PANIC_ON_ALLOC_FAIL: bool = true; // all allocations are small (see `MAX_ALLOC_LIMIT`)

    type MemoryKind = !;

    #[inline(always)]
    fn enforce_alignment(_ecx: &InterpCx<'mir, 'tcx, Self>) -> CheckAlignment {
        // We do not check for alignment to avoid having to carry an `Align`
        // in `ConstValue::ByRef`.
        CheckAlignment::No
    }

    #[inline(always)]
    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>, _layout: TyAndLayout<'tcx>) -> bool {
        false // for now, we don't enforce validity
    }
    fn alignment_check_failed(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        _has: Align,
        _required: Align,
        _check: CheckAlignment,
    ) -> InterpResult<'tcx, ()> {
        span_bug!(
            ecx.cur_span(),
            "`alignment_check_failed` called when no alignment check requested"
        )
    }

    fn load_mir(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx Body<'tcx>> {
        throw_machine_stop_str!("calling functions isn't supported in ConstProp")
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _abi: CallAbi,
        _args: &[FnArg<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'mir Body<'tcx>, ty::Instance<'tcx>)>> {
        Ok(None)
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> InterpResult<'tcx> {
        throw_machine_stop_str!("calling intrinsics isn't supported in ConstProp")
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: rustc_middle::mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        bug!("panics terminators are not evaluated in ConstProp")
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: &ImmTy<'tcx>,
        _right: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        // We can't do this because aliasing of memory can differ between const eval and llvm
        throw_machine_stop_str!("pointer arithmetic or comparisons aren't supported in ConstProp")
    }

    fn access_local_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
        frame: usize,
        local: Local,
    ) -> InterpResult<'tcx, &'a mut interpret::Operand<Self::Provenance>> {
        assert_eq!(frame, 0);
        match ecx.machine.can_const_prop[local] {
            ConstPropMode::NoPropagation => {
                throw_machine_stop_str!(
                    "tried to write to a local that is marked as not propagatable"
                )
            }
            ConstPropMode::OnlyInsideOwnBlock => {
                ecx.machine.written_only_inside_own_block_locals.insert(local);
            }
            ConstPropMode::FullConstProp => {}
        }
        ecx.machine.stack[frame].locals[local].access_mut()
    }

    fn before_access_global(
        _tcx: TyCtxt<'tcx>,
        _machine: &Self,
        _alloc_id: AllocId,
        alloc: ConstAllocation<'tcx>,
        _static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        if is_write {
            throw_machine_stop_str!("can't write to global");
        }
        // If the static allocation is mutable, then we can't const prop it as its content
        // might be different at runtime.
        if alloc.inner().mutability.is_mut() {
            throw_machine_stop_str!("can't access mutable globals in ConstProp");
        }

        Ok(())
    }

    #[inline(always)]
    fn expose_ptr(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _ptr: Pointer<AllocId>,
    ) -> InterpResult<'tcx> {
        throw_machine_stop_str!("exposing pointers isn't supported in ConstProp")
    }

    #[inline(always)]
    fn init_frame_extra(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx>> {
        Ok(frame)
    }

    #[inline(always)]
    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }
}

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    local_decls: &'mir IndexSlice<Local, LocalDecl<'tcx>>,
}

impl<'tcx> LayoutOfHelpers<'tcx> for ConstPropagator<'_, 'tcx> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}

impl HasDataLayout for ConstPropagator<'_, '_> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for ConstPropagator<'_, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> ty::layout::HasParamEnv<'tcx> for ConstPropagator<'_, 'tcx> {
    #[inline]
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'mir, 'tcx> ConstPropagator<'mir, 'tcx> {
    fn new(
        body: &Body<'tcx>,
        dummy_body: &'mir Body<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> ConstPropagator<'mir, 'tcx> {
        let def_id = body.source.def_id();
        let args = &GenericArgs::identity_for_item(tcx, def_id);
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        let can_const_prop = CanConstProp::check(tcx, param_env, body);
        let mut ecx = InterpCx::new(
            tcx,
            tcx.def_span(def_id),
            param_env,
            ConstPropMachine::new(can_const_prop),
        );

        let ret_layout = ecx
            .layout_of(body.bound_return_ty().instantiate(tcx, args))
            .ok()
            // Don't bother allocating memory for large values.
            // I don't know how return types can seem to be unsized but this happens in the
            // `type/type-unsatisfiable.rs` test.
            .filter(|ret_layout| {
                ret_layout.is_sized() && ret_layout.size < Size::from_bytes(MAX_ALLOC_LIMIT)
            })
            .unwrap_or_else(|| ecx.layout_of(tcx.types.unit).unwrap());

        let ret = ecx
            .allocate(ret_layout, MemoryKind::Stack)
            .expect("couldn't perform small allocation")
            .into();

        ecx.push_stack_frame(
            Instance::new(def_id, args),
            dummy_body,
            &ret,
            StackPopCleanup::Root { cleanup: false },
        )
        .expect("failed to push initial stack frame");

        ConstPropagator { ecx, tcx, param_env, local_decls: &dummy_body.local_decls }
    }

    fn get_const(&self, place: Place<'tcx>) -> Option<OpTy<'tcx>> {
        let op = match self.ecx.eval_place_to_op(place, None) {
            Ok(op) => {
                if matches!(*op, interpret::Operand::Immediate(Immediate::Uninit)) {
                    // Make sure nobody accidentally uses this value.
                    return None;
                }
                op
            }
            Err(e) => {
                trace!("get_const failed: {:?}", e.into_kind().debug());
                return None;
            }
        };

        // Try to read the local as an immediate so that if it is representable as a scalar, we can
        // handle it as such, but otherwise, just return the value as is.
        Some(match self.ecx.read_immediate_raw(&op) {
            Ok(Right(imm)) => imm.into(),
            _ => op,
        })
    }

    /// Remove `local` from the pool of `Locals`. Allows writing to them,
    /// but not reading from them anymore.
    fn remove_const(ecx: &mut InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>, local: Local) {
        ecx.frame_mut().locals[local].value =
            LocalValue::Live(interpret::Operand::Immediate(interpret::Immediate::Uninit));
        ecx.machine.written_only_inside_own_block_locals.remove(&local);
    }

    fn propagate_operand(&mut self, operand: &mut Operand<'tcx>) {
        if let Some(place) = operand.place() && let Some(op) = self.replace_with_const(place) {
            *operand = op;
        }
    }

    fn check_rvalue(&mut self, rvalue: &Rvalue<'tcx>) -> Option<()> {
        // Perform any special handling for specific Rvalue types.
        // Generally, checks here fall into one of two categories:
        //   1. Additional checking to provide useful lints to the user
        //        - In this case, we will do some validation and then fall through to the
        //          end of the function which evals the assignment.
        //   2. Working around bugs in other parts of the compiler
        //        - In this case, we'll return `None` from this function to stop evaluation.
        match rvalue {
            // Do not try creating references (#67862)
            Rvalue::AddressOf(_, place) | Rvalue::Ref(_, _, place) => {
                trace!("skipping AddressOf | Ref for {:?}", place);

                // This may be creating mutable references or immutable references to cells.
                // If that happens, the pointed to value could be mutated via that reference.
                // Since we aren't tracking references, the const propagator loses track of what
                // value the local has right now.
                // Thus, all locals that have their reference taken
                // must not take part in propagation.
                Self::remove_const(&mut self.ecx, place.local);

                return None;
            }
            Rvalue::ThreadLocalRef(def_id) => {
                trace!("skipping ThreadLocalRef({:?})", def_id);

                return None;
            }
            // There's no other checking to do at this time.
            Rvalue::Aggregate(..)
            | Rvalue::Use(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::Repeat(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..) => {}
        }

        // FIXME we need to revisit this for #67176
        if rvalue.has_param() {
            return None;
        }
        if !rvalue
            .ty(&self.ecx.frame().body.local_decls, *self.ecx.tcx)
            .is_sized(*self.ecx.tcx, self.param_env)
        {
            // the interpreter doesn't support unsized locals (only unsized arguments),
            // but rustc does (in a kinda broken way), so we have to skip them here
            return None;
        }

        Some(())
    }

    // Attempt to use algebraic identities to eliminate constant expressions
    fn eval_rvalue_with_identities(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place: Place<'tcx>,
    ) -> Option<()> {
        match rvalue {
            Rvalue::BinaryOp(op, box (left, right))
            | Rvalue::CheckedBinaryOp(op, box (left, right)) => {
                let l = self.ecx.eval_operand(left, None).and_then(|x| self.ecx.read_immediate(&x));
                let r =
                    self.ecx.eval_operand(right, None).and_then(|x| self.ecx.read_immediate(&x));

                let const_arg = match (l, r) {
                    (Ok(x), Err(_)) | (Err(_), Ok(x)) => x, // exactly one side is known
                    (Err(_), Err(_)) => return None,        // neither side is known
                    (Ok(_), Ok(_)) => return self.ecx.eval_rvalue_into_place(rvalue, place).ok(), // both sides are known
                };

                if !matches!(const_arg.layout.abi, abi::Abi::Scalar(..)) {
                    // We cannot handle Scalar Pair stuff.
                    // No point in calling `eval_rvalue_into_place`, since only one side is known
                    return None;
                }

                let arg_value = const_arg.to_scalar().to_bits(const_arg.layout.size).ok()?;
                let dest = self.ecx.eval_place(place).ok()?;

                match op {
                    BinOp::BitAnd if arg_value == 0 => {
                        self.ecx.write_immediate(*const_arg, &dest).ok()
                    }
                    BinOp::BitOr
                        if arg_value == const_arg.layout.size.truncate(u128::MAX)
                            || (const_arg.layout.ty.is_bool() && arg_value == 1) =>
                    {
                        self.ecx.write_immediate(*const_arg, &dest).ok()
                    }
                    BinOp::Mul if const_arg.layout.ty.is_integral() && arg_value == 0 => {
                        if let Rvalue::CheckedBinaryOp(_, _) = rvalue {
                            let val = Immediate::ScalarPair(
                                const_arg.to_scalar(),
                                Scalar::from_bool(false),
                            );
                            self.ecx.write_immediate(val, &dest).ok()
                        } else {
                            self.ecx.write_immediate(*const_arg, &dest).ok()
                        }
                    }
                    _ => None,
                }
            }
            _ => self.ecx.eval_rvalue_into_place(rvalue, place).ok(),
        }
    }

    /// Creates a new `Operand::Constant` from a `Scalar` value
    fn operand_from_scalar(&self, scalar: Scalar, ty: Ty<'tcx>) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span: DUMMY_SP,
            user_ty: None,
            literal: ConstantKind::from_scalar(self.tcx, scalar, ty),
        }))
    }

    fn replace_with_const(&mut self, place: Place<'tcx>) -> Option<Operand<'tcx>> {
        // This will return None if the above `const_prop` invocation only "wrote" a
        // type whose creation requires no write. E.g. a generator whose initial state
        // consists solely of uninitialized memory (so it doesn't capture any locals).
        let value = self.get_const(place)?;
        if !self.tcx.consider_optimizing(|| format!("ConstantPropagation - {value:?}")) {
            return None;
        }
        trace!("replacing {:?} with {:?}", place, value);

        // FIXME> figure out what to do when read_immediate_raw fails
        let imm = self.ecx.read_immediate_raw(&value).ok()?;

        let Right(imm) = imm else { return None };
        match *imm {
            Immediate::Scalar(scalar) if scalar.try_to_int().is_ok() => {
                Some(self.operand_from_scalar(scalar, value.layout.ty))
            }
            Immediate::ScalarPair(l, r) if l.try_to_int().is_ok() && r.try_to_int().is_ok() => {
                let alloc = self
                    .ecx
                    .intern_with_temp_alloc(value.layout, |ecx, dest| {
                        ecx.write_immediate(*imm, dest)
                    })
                    .ok()?;

                let literal = ConstantKind::Val(
                    ConstValue::ByRef { alloc, offset: Size::ZERO },
                    value.layout.ty,
                );
                Some(Operand::Constant(Box::new(Constant {
                    span: DUMMY_SP,
                    user_ty: None,
                    literal,
                })))
            }
            // Scalars or scalar pairs that contain undef values are assumed to not have
            // successfully evaluated and are thus not propagated.
            _ => None,
        }
    }

    fn ensure_not_propagated(&self, local: Local) {
        if cfg!(debug_assertions) {
            assert!(
                self.get_const(local.into()).is_none()
                    || self
                        .layout_of(self.local_decls[local].ty)
                        .map_or(true, |layout| layout.is_zst()),
                "failed to remove values for `{local:?}`, value={:?}",
                self.get_const(local.into()),
            )
        }
    }
}

/// The mode that `ConstProp` is allowed to run in for a given `Local`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstPropMode {
    /// The `Local` can be propagated into and reads of this `Local` can also be propagated.
    FullConstProp,
    /// The `Local` can only be propagated into and from its own block.
    OnlyInsideOwnBlock,
    /// The `Local` cannot be part of propagation at all. Any statement
    /// referencing it either for reading or writing will not get propagated.
    NoPropagation,
}

pub struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // False at the beginning. Once set, no more assignments are allowed to that local.
    found_assignment: BitSet<Local>,
}

impl CanConstProp {
    /// Returns true if `local` can be propagated
    pub fn check<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        body: &Body<'tcx>,
    ) -> IndexVec<Local, ConstPropMode> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(ConstPropMode::FullConstProp, &body.local_decls),
            found_assignment: BitSet::new_empty(body.local_decls.len()),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            let ty = body.local_decls[local].ty;
            match tcx.layout_of(param_env.and(ty)) {
                Ok(layout) if layout.size < Size::from_bytes(MAX_ALLOC_LIMIT) => {}
                // Either the layout fails to compute, then we can't use this local anyway
                // or the local is too large, then we don't want to.
                _ => {
                    *val = ConstPropMode::NoPropagation;
                    continue;
                }
            }
        }
        // Consider that arguments are assigned on entry.
        for arg in body.args_iter() {
            cpv.found_assignment.insert(arg);
        }
        cpv.visit_body(&body);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_place(&mut self, place: &Place<'tcx>, mut context: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;

        // Dereferencing just read the addess of `place.local`.
        if place.projection.first() == Some(&PlaceElem::Deref) {
            context = NonMutatingUse(NonMutatingUseContext::Copy);
        }

        self.visit_local(place.local, context, loc);
        self.visit_projection(place.as_ref(), context, loc);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;
        match context {
            // These are just stores, where the storing is not propagatable, but there may be later
            // mutations of the same local via `Store`
            | MutatingUse(MutatingUseContext::Call)
            | MutatingUse(MutatingUseContext::AsmOutput)
            | MutatingUse(MutatingUseContext::Deinit)
            // Actual store that can possibly even propagate a value
            | MutatingUse(MutatingUseContext::Store)
            | MutatingUse(MutatingUseContext::SetDiscriminant) => {
                if !self.found_assignment.insert(local) {
                    match &mut self.can_const_prop[local] {
                        // If the local can only get propagated in its own block, then we don't have
                        // to worry about multiple assignments, as we'll nuke the const state at the
                        // end of the block anyway, and inside the block we overwrite previous
                        // states as applicable.
                        ConstPropMode::OnlyInsideOwnBlock => {}
                        ConstPropMode::NoPropagation => {}
                        other @ ConstPropMode::FullConstProp => {
                            trace!(
                                "local {:?} can't be propagated because of multiple assignments. Previous state: {:?}",
                                local, other,
                            );
                            *other = ConstPropMode::OnlyInsideOwnBlock;
                        }
                    }
                }
            }
            // Reading constants is allowed an arbitrary number of times
            NonMutatingUse(NonMutatingUseContext::Copy)
            | NonMutatingUse(NonMutatingUseContext::Move)
            | NonMutatingUse(NonMutatingUseContext::Inspect)
            | NonMutatingUse(NonMutatingUseContext::PlaceMention)
            | NonUse(_) => {}

            // These could be propagated with a smarter analysis or just some careful thinking about
            // whether they'd be fine right now.
            MutatingUse(MutatingUseContext::Yield)
            | MutatingUse(MutatingUseContext::Drop)
            | MutatingUse(MutatingUseContext::Retag)
            // These can't ever be propagated under any scheme, as we can't reason about indirect
            // mutation.
            | NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            | NonMutatingUse(NonMutatingUseContext::ShallowBorrow)
            | NonMutatingUse(NonMutatingUseContext::AddressOf)
            | MutatingUse(MutatingUseContext::Borrow)
            | MutatingUse(MutatingUseContext::AddressOf) => {
                trace!("local {:?} can't be propagated because it's used: {:?}", local, context);
                self.can_const_prop[local] = ConstPropMode::NoPropagation;
            }
            MutatingUse(MutatingUseContext::Projection)
            | NonMutatingUse(NonMutatingUseContext::Projection) => bug!("visit_place should not pass {context:?} for {local:?}"),
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for ConstPropagator<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
        self.propagate_operand(operand)
    }

    fn process_projection_elem(
        &mut self,
        elem: PlaceElem<'tcx>,
        _: Location,
    ) -> Option<PlaceElem<'tcx>> {
        if let PlaceElem::Index(local) = elem
            && let Some(value) = self.get_const(local.into())
            && let interpret::Operand::Immediate(Immediate::Scalar(scalar)) = *value
            && let Ok(offset) = scalar.to_target_usize(&self.tcx)
            && let Some(min_length) = offset.checked_add(1)
        {
            Some(PlaceElem::ConstantIndex { offset, min_length, from_end: false })
        } else {
            None
        }
    }

    fn visit_assign(
        &mut self,
        place: &mut Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) {
        self.super_assign(place, rvalue, location);

        let Some(()) = self.check_rvalue(rvalue) else { return };

        match self.ecx.machine.can_const_prop[place.local] {
            // Do nothing if the place is indirect.
            _ if place.is_indirect() => {}
            ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
            ConstPropMode::OnlyInsideOwnBlock | ConstPropMode::FullConstProp => {
                if let Some(()) = self.eval_rvalue_with_identities(rvalue, *place) {
                    // If this was already an evaluated constant, keep it.
                    if let Rvalue::Use(Operand::Constant(c)) = rvalue
                        && let ConstantKind::Val(..) = c.literal
                    {
                        trace!("skipping replace of Rvalue::Use({:?} because it is already a const", c);
                    } else if let Some(operand) = self.replace_with_const(*place) {
                        *rvalue = Rvalue::Use(operand);
                    }
                } else {
                    // Const prop failed, so erase the destination, ensuring that whatever happens
                    // from here on, does not know about the previous value.
                    // This is important in case we have
                    // ```rust
                    // let mut x = 42;
                    // x = SOME_MUTABLE_STATIC;
                    // // x must now be uninit
                    // ```
                    // FIXME: we overzealously erase the entire local, because that's easier to
                    // implement.
                    trace!(
                        "propagation into {:?} failed.
                        Nuking the entire site from orbit, it's the only way to be sure",
                        place,
                    );
                    Self::remove_const(&mut self.ecx, place.local);
                }
            }
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);

        // We want to evaluate operands before any change to the assigned-to value,
        // so we recurse first.
        self.super_statement(statement, location);

        match statement.kind {
            StatementKind::SetDiscriminant { ref place, .. } => {
                match self.ecx.machine.can_const_prop[place.local] {
                    // Do nothing if the place is indirect.
                    _ if place.is_indirect() => {}
                    ConstPropMode::NoPropagation => self.ensure_not_propagated(place.local),
                    ConstPropMode::FullConstProp | ConstPropMode::OnlyInsideOwnBlock => {
                        if self.ecx.statement(statement).is_ok() {
                            trace!("propped discriminant into {:?}", place);
                        } else {
                            Self::remove_const(&mut self.ecx, place.local);
                        }
                    }
                }
            }
            StatementKind::StorageLive(local) => {
                Self::remove_const(&mut self.ecx, local);
            }
            // We do not need to mark dead locals as such. For `FullConstProp` locals,
            // this allows to propagate the single assigned value in this case:
            // ```
            // let x = SOME_CONST;
            // if a {
            //   f(copy x);
            //   StorageDead(x);
            // } else {
            //   g(copy x);
            //   StorageDead(x);
            // }
            // ```
            //
            // This may propagate a constant where the local would be uninit or dead.
            // In both cases, this does not matter, as those reads would be UB anyway.
            _ => {}
        }
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        self.super_basic_block_data(block, data);

        // We remove all Locals which are restricted in propagation to their containing blocks and
        // which were modified in the current block.
        // Take it out of the ecx so we can get a mutable reference to the ecx for `remove_const`.
        let mut written_only_inside_own_block_locals =
            std::mem::take(&mut self.ecx.machine.written_only_inside_own_block_locals);

        // This loop can get very hot for some bodies: it check each local in each bb.
        // To avoid this quadratic behaviour, we only clear the locals that were modified inside
        // the current block.
        for local in written_only_inside_own_block_locals.drain() {
            debug_assert_eq!(
                self.ecx.machine.can_const_prop[local],
                ConstPropMode::OnlyInsideOwnBlock
            );
            Self::remove_const(&mut self.ecx, local);
        }
        self.ecx.machine.written_only_inside_own_block_locals =
            written_only_inside_own_block_locals;

        if cfg!(debug_assertions) {
            for (local, &mode) in self.ecx.machine.can_const_prop.iter_enumerated() {
                match mode {
                    ConstPropMode::FullConstProp => {}
                    ConstPropMode::NoPropagation | ConstPropMode::OnlyInsideOwnBlock => {
                        self.ensure_not_propagated(local);
                    }
                }
            }
        }
    }
}
