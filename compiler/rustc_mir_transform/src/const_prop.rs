//! Propagates constants for early reporting of statically known
//! assertion failures

use std::cell::Cell;

use either::Right;

use rustc_const_eval::const_eval::CheckAlignment;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{
    BasicBlock, BinOp, Body, Constant, ConstantKind, Local, LocalDecl, LocalKind, Location,
    Operand, Place, Rvalue, SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
    RETURN_PLACE,
};
use rustc_middle::ty::layout::{LayoutError, LayoutOf, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::InternalSubsts;
use rustc_middle::ty::{self, ConstKind, Instance, ParamEnv, Ty, TyCtxt, TypeVisitable};
use rustc_span::{def_id::DefId, Span};
use rustc_target::abi::{self, Align, HasDataLayout, Size, TargetDataLayout};
use rustc_target::spec::abi::Abi as CallAbi;
use rustc_trait_selection::traits;

use crate::MirPass;
use rustc_const_eval::interpret::{
    self, compile_time_machine, AllocId, ConstAllocation, ConstValue, CtfeValidationMode, Frame,
    ImmTy, Immediate, InterpCx, InterpResult, LocalState, LocalValue, MemoryKind, OpTy, PlaceTy,
    Pointer, Scalar, StackPopCleanup, StackPopUnwind,
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
        struct Zst;
        // Printing this type shows the desired string.
        impl std::fmt::Display for Zst {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $($tt)*)
            }
        }
        impl rustc_middle::mir::interpret::MachineStopType for Zst {}
        throw_machine_stop!(Zst)
    }};
}

pub struct ConstProp;

impl<'tcx> MirPass<'tcx> for ConstProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
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

        let is_generator = tcx.type_of(def_id.to_def_id()).subst_identity().is_generator();
        // FIXME(welseywiser) const prop doesn't work on generators because of query cycles
        // computing their layout.
        if is_generator {
            trace!("ConstProp skipped for generator {:?}", def_id);
            return;
        }

        // Check if it's even possible to satisfy the 'where' clauses
        // for this item.
        // This branch will never be taken for any normal function.
        // However, it's possible to `#!feature(trivial_bounds)]` to write
        // a function with impossible to satisfy clauses, e.g.:
        // `fn foo() where String: Copy {}`
        //
        // We don't usually need to worry about this kind of case,
        // since we would get a compilation error if the user tried
        // to call it. However, since we can do const propagation
        // even without any calls to the function, we need to make
        // sure that it even makes sense to try to evaluate the body.
        // If there are unsatisfiable where clauses, then all bets are
        // off, and we just give up.
        //
        // We manually filter the predicates, skipping anything that's not
        // "global". We are in a potentially generic context
        // (e.g. we are evaluating a function without substituting generic
        // parameters, so this filtering serves two purposes:
        //
        // 1. We skip evaluating any predicates that we would
        // never be able prove are unsatisfiable (e.g. `<T as Foo>`
        // 2. We avoid trying to normalize predicates involving generic
        // parameters (e.g. `<T as Foo>::MyItem`). This can confuse
        // the normalization code (leading to cycle errors), since
        // it's usually never invoked in this way.
        let predicates = tcx
            .predicates_of(def_id.to_def_id())
            .predicates
            .iter()
            .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None });
        if traits::impossible_predicates(
            tcx,
            traits::elaborate_predicates(tcx, predicates).map(|o| o.predicate).collect(),
        ) {
            trace!("ConstProp skipped for {:?}: found unsatisfiable predicates", def_id);
            return;
        }

        trace!("ConstProp starting for {:?}", def_id);

        let dummy_body = &Body::new(
            body.source,
            (*body.basic_blocks).clone(),
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
        optimization_finder.visit_body(body);

        trace!("ConstProp done for {:?}", def_id);
    }
}

pub struct ConstPropMachine<'mir, 'tcx> {
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx>>,
    /// `OnlyInsideOwnBlock` locals that were written in the current block get erased at the end.
    pub written_only_inside_own_block_locals: FxHashSet<Local>,
    /// Locals that need to be cleared after every block terminates.
    pub only_propagate_inside_block_locals: BitSet<Local>,
    pub can_const_prop: IndexVec<Local, ConstPropMode>,
}

impl ConstPropMachine<'_, '_> {
    pub fn new(
        only_propagate_inside_block_locals: BitSet<Local>,
        can_const_prop: IndexVec<Local, ConstPropMode>,
    ) -> Self {
        Self {
            stack: Vec::new(),
            written_only_inside_own_block_locals: Default::default(),
            only_propagate_inside_block_locals,
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
    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
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
        _args: &[OpTy<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx, Option<(&'mir Body<'tcx>, ty::Instance<'tcx>)>> {
        Ok(None)
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        throw_machine_stop_str!("calling intrinsics isn't supported in ConstProp")
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: Option<rustc_middle::mir::BasicBlock>,
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
        if ecx.machine.can_const_prop[local] == ConstPropMode::NoPropagation {
            throw_machine_stop_str!("tried to write to a local that is marked as not propagatable")
        }
        if frame == 0 && ecx.machine.only_propagate_inside_block_locals.contains(local) {
            trace!(
                "mutating local {:?} which is restricted to its block. \
                Will remove it from const-prop after block is finished.",
                local
            );
            ecx.machine.written_only_inside_own_block_locals.insert(local);
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
    local_decls: &'mir IndexVec<Local, LocalDecl<'tcx>>,
    // Because we have `MutVisitor` we can't obtain the `SourceInfo` from a `Location`. So we store
    // the last known `SourceInfo` here and just keep revisiting it.
    source_info: Option<SourceInfo>,
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
        let substs = &InternalSubsts::identity_for_item(tcx, def_id);
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        let can_const_prop = CanConstProp::check(tcx, param_env, body);
        let mut only_propagate_inside_block_locals = BitSet::new_empty(can_const_prop.len());
        for (l, mode) in can_const_prop.iter_enumerated() {
            if *mode == ConstPropMode::OnlyInsideOwnBlock {
                only_propagate_inside_block_locals.insert(l);
            }
        }
        let mut ecx = InterpCx::new(
            tcx,
            tcx.def_span(def_id),
            param_env,
            ConstPropMachine::new(only_propagate_inside_block_locals, can_const_prop),
        );

        let ret_layout = ecx
            .layout_of(body.bound_return_ty().subst(tcx, substs))
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
            Instance::new(def_id, substs),
            dummy_body,
            &ret,
            StackPopCleanup::Root { cleanup: false },
        )
        .expect("failed to push initial stack frame");

        ConstPropagator {
            ecx,
            tcx,
            param_env,
            local_decls: &dummy_body.local_decls,
            source_info: None,
        }
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
                trace!("get_const failed: {}", e);
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
        ecx.frame_mut().locals[local] = LocalState {
            value: LocalValue::Live(interpret::Operand::Immediate(interpret::Immediate::Uninit)),
            layout: Cell::new(None),
        };
    }

    fn use_ecx<F, T>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        match f(self) {
            Ok(val) => Some(val),
            Err(error) => {
                trace!("InterpCx operation failed: {:?}", error);
                // Some errors shouldn't come up because creating them causes
                // an allocation, which we should avoid. When that happens,
                // dedicated error variants should be introduced instead.
                assert!(
                    !error.kind().formatted_string(),
                    "const-prop encountered formatting error: {}",
                    error
                );
                None
            }
        }
    }

    /// Returns the value, if any, of evaluating `c`.
    fn eval_constant(&mut self, c: &Constant<'tcx>) -> Option<OpTy<'tcx>> {
        // FIXME we need to revisit this for #67176
        if c.needs_subst() {
            return None;
        }

        // No span, we don't want errors to be shown.
        self.ecx.eval_mir_constant(&c.literal, None, None).ok()
    }

    /// Returns the value, if any, of evaluating `place`.
    fn eval_place(&mut self, place: Place<'tcx>) -> Option<OpTy<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(|this| this.ecx.eval_place_to_op(place, None))
    }

    /// Returns the value, if any, of evaluating `op`. Calls upon `eval_constant`
    /// or `eval_place`, depending on the variant of `Operand` used.
    fn eval_operand(&mut self, op: &Operand<'tcx>) -> Option<OpTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c),
            Operand::Move(place) | Operand::Copy(place) => self.eval_place(place),
        }
    }

    fn propagate_operand(&mut self, operand: &mut Operand<'tcx>) {
        match *operand {
            Operand::Copy(l) | Operand::Move(l) => {
                if let Some(value) = self.get_const(l) && self.should_const_prop(&value) {
                    // FIXME(felix91gr): this code only handles `Scalar` cases.
                    // For now, we're not handling `ScalarPair` cases because
                    // doing so here would require a lot of code duplication.
                    // We should hopefully generalize `Operand` handling into a fn,
                    // and use it to do const-prop here and everywhere else
                    // where it makes sense.
                    if let interpret::Operand::Immediate(interpret::Immediate::Scalar(
                        scalar,
                    )) = *value
                    {
                        *operand = self.operand_from_scalar(
                            scalar,
                            value.layout.ty,
                            self.source_info.unwrap().span,
                        );
                    }
                }
            }
            Operand::Constant(_) => (),
        }
    }

    fn const_prop(&mut self, rvalue: &Rvalue<'tcx>, place: Place<'tcx>) -> Option<()> {
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
        if rvalue.needs_subst() {
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

        self.eval_rvalue_with_identities(rvalue, place)
    }

    // Attempt to use algebraic identities to eliminate constant expressions
    fn eval_rvalue_with_identities(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place: Place<'tcx>,
    ) -> Option<()> {
        self.use_ecx(|this| match rvalue {
            Rvalue::BinaryOp(op, box (left, right))
            | Rvalue::CheckedBinaryOp(op, box (left, right)) => {
                let l = this.ecx.eval_operand(left, None).and_then(|x| this.ecx.read_immediate(&x));
                let r =
                    this.ecx.eval_operand(right, None).and_then(|x| this.ecx.read_immediate(&x));

                let const_arg = match (l, r) {
                    (Ok(x), Err(_)) | (Err(_), Ok(x)) => x, // exactly one side is known
                    (Err(e), Err(_)) => return Err(e),      // neither side is known
                    (Ok(_), Ok(_)) => return this.ecx.eval_rvalue_into_place(rvalue, place), // both sides are known
                };

                if !matches!(const_arg.layout.abi, abi::Abi::Scalar(..)) {
                    // We cannot handle Scalar Pair stuff.
                    // No point in calling `eval_rvalue_into_place`, since only one side is known
                    throw_machine_stop_str!("cannot optimize this")
                }

                let arg_value = const_arg.to_scalar().to_bits(const_arg.layout.size)?;
                let dest = this.ecx.eval_place(place)?;

                match op {
                    BinOp::BitAnd if arg_value == 0 => this.ecx.write_immediate(*const_arg, &dest),
                    BinOp::BitOr
                        if arg_value == const_arg.layout.size.truncate(u128::MAX)
                            || (const_arg.layout.ty.is_bool() && arg_value == 1) =>
                    {
                        this.ecx.write_immediate(*const_arg, &dest)
                    }
                    BinOp::Mul if const_arg.layout.ty.is_integral() && arg_value == 0 => {
                        if let Rvalue::CheckedBinaryOp(_, _) = rvalue {
                            let val = Immediate::ScalarPair(
                                const_arg.to_scalar(),
                                Scalar::from_bool(false),
                            );
                            this.ecx.write_immediate(val, &dest)
                        } else {
                            this.ecx.write_immediate(*const_arg, &dest)
                        }
                    }
                    _ => throw_machine_stop_str!("cannot optimize this"),
                }
            }
            _ => this.ecx.eval_rvalue_into_place(rvalue, place),
        })
    }

    /// Creates a new `Operand::Constant` from a `Scalar` value
    fn operand_from_scalar(&self, scalar: Scalar, ty: Ty<'tcx>, span: Span) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ConstantKind::from_scalar(self.tcx, scalar, ty),
        }))
    }

    fn replace_with_const(
        &mut self,
        rval: &mut Rvalue<'tcx>,
        value: &OpTy<'tcx>,
        source_info: SourceInfo,
    ) {
        if let Rvalue::Use(Operand::Constant(c)) = rval {
            match c.literal {
                ConstantKind::Ty(c) if matches!(c.kind(), ConstKind::Unevaluated(..)) => {}
                _ => {
                    trace!("skipping replace of Rvalue::Use({:?} because it is already a const", c);
                    return;
                }
            }
        }

        trace!("attempting to replace {:?} with {:?}", rval, value);
        if let Err(e) = self.ecx.const_validate_operand(
            value,
            vec![],
            // FIXME: is ref tracking too expensive?
            // FIXME: what is the point of ref tracking if we do not even check the tracked refs?
            &mut interpret::RefTracking::empty(),
            CtfeValidationMode::Regular,
        ) {
            trace!("validation error, attempt failed: {:?}", e);
            return;
        }

        // FIXME> figure out what to do when read_immediate_raw fails
        let imm = self.use_ecx(|this| this.ecx.read_immediate_raw(value));

        if let Some(Right(imm)) = imm {
            match *imm {
                interpret::Immediate::Scalar(scalar) => {
                    *rval = Rvalue::Use(self.operand_from_scalar(
                        scalar,
                        value.layout.ty,
                        source_info.span,
                    ));
                }
                Immediate::ScalarPair(..) => {
                    // Found a value represented as a pair. For now only do const-prop if the type
                    // of `rvalue` is also a tuple with two scalars.
                    // FIXME: enable the general case stated above ^.
                    let ty = value.layout.ty;
                    // Only do it for tuples
                    if let ty::Tuple(types) = ty.kind() {
                        // Only do it if tuple is also a pair with two scalars
                        if let [ty1, ty2] = types[..] {
                            let alloc = self.use_ecx(|this| {
                                let ty_is_scalar = |ty| {
                                    this.ecx.layout_of(ty).ok().map(|layout| layout.abi.is_scalar())
                                        == Some(true)
                                };
                                if ty_is_scalar(ty1) && ty_is_scalar(ty2) {
                                    let alloc = this
                                        .ecx
                                        .intern_with_temp_alloc(value.layout, |ecx, dest| {
                                            ecx.write_immediate(*imm, dest)
                                        })
                                        .unwrap();
                                    Ok(Some(alloc))
                                } else {
                                    Ok(None)
                                }
                            });

                            if let Some(Some(alloc)) = alloc {
                                // Assign entire constant in a single statement.
                                // We can't use aggregates, as we run after the aggregate-lowering `MirPhase`.
                                let const_val = ConstValue::ByRef { alloc, offset: Size::ZERO };
                                let literal = ConstantKind::Val(const_val, ty);
                                *rval = Rvalue::Use(Operand::Constant(Box::new(Constant {
                                    span: source_info.span,
                                    user_ty: None,
                                    literal,
                                })));
                            }
                        }
                    }
                }
                // Scalars or scalar pairs that contain undef values are assumed to not have
                // successfully evaluated and are thus not propagated.
                _ => {}
            }
        }
    }

    /// Returns `true` if and only if this `op` should be const-propagated into.
    fn should_const_prop(&mut self, op: &OpTy<'tcx>) -> bool {
        if !self.tcx.consider_optimizing(|| format!("ConstantPropagation - OpTy: {:?}", op)) {
            return false;
        }

        match **op {
            interpret::Operand::Immediate(Immediate::Scalar(s)) => s.try_to_int().is_ok(),
            interpret::Operand::Immediate(Immediate::ScalarPair(l, r)) => {
                l.try_to_int().is_ok() && r.try_to_int().is_ok()
            }
            _ => false,
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
    /// The `Local` can be propagated into but reads cannot be propagated.
    OnlyPropagateInto,
    /// The `Local` cannot be part of propagation at all. Any statement
    /// referencing it either for reading or writing will not get propagated.
    NoPropagation,
}

pub struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // False at the beginning. Once set, no more assignments are allowed to that local.
    found_assignment: BitSet<Local>,
    // Cache of locals' information
    local_kinds: IndexVec<Local, LocalKind>,
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
            local_kinds: IndexVec::from_fn_n(
                |local| body.local_kind(local),
                body.local_decls.len(),
            ),
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
            // Cannot use args at all
            // Cannot use locals because if x < y { y - x } else { x - y } would
            //        lint for x != y
            // FIXME(oli-obk): lint variables until they are used in a condition
            // FIXME(oli-obk): lint if return value is constant
            if cpv.local_kinds[local] == LocalKind::Arg {
                *val = ConstPropMode::OnlyPropagateInto;
                trace!(
                    "local {:?} can't be const propagated because it's a function argument",
                    local
                );
            } else if cpv.local_kinds[local] == LocalKind::Var {
                *val = ConstPropMode::OnlyInsideOwnBlock;
                trace!(
                    "local {:?} will only be propagated inside its block, because it's a user variable",
                    local
                );
            }
        }
        cpv.visit_body(&body);
        cpv.can_const_prop
    }
}

impl Visitor<'_> for CanConstProp {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;
        match context {
            // Projections are fine, because `&mut foo.x` will be caught by
            // `MutatingUseContext::Borrow` elsewhere.
            MutatingUse(MutatingUseContext::Projection)
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
                        ConstPropMode::OnlyPropagateInto => {}
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
            | NonMutatingUse(NonMutatingUseContext::Projection)
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
            | NonMutatingUse(NonMutatingUseContext::UniqueBorrow)
            | NonMutatingUse(NonMutatingUseContext::AddressOf)
            | MutatingUse(MutatingUseContext::Borrow)
            | MutatingUse(MutatingUseContext::AddressOf) => {
                trace!("local {:?} can't be propagaged because it's used: {:?}", local, context);
                self.can_const_prop[local] = ConstPropMode::NoPropagation;
            }
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for ConstPropagator<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_body(&mut self, body: &mut Body<'tcx>) {
        for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            self.visit_basic_block_data(bb, data);
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);

        // Only const prop copies and moves on `mir_opt_level=3` as doing so
        // currently slightly increases compile time in some cases.
        if self.tcx.sess.mir_opt_level() >= 3 {
            self.propagate_operand(operand)
        }
    }

    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, location: Location) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant);
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);
        let source_info = statement.source_info;
        self.source_info = Some(source_info);
        if let StatementKind::Assign(box (place, ref mut rval)) = statement.kind {
            let can_const_prop = self.ecx.machine.can_const_prop[place.local];
            if let Some(()) = self.const_prop(rval, place) {
                // This will return None if the above `const_prop` invocation only "wrote" a
                // type whose creation requires no write. E.g. a generator whose initial state
                // consists solely of uninitialized memory (so it doesn't capture any locals).
                if let Some(ref value) = self.get_const(place) && self.should_const_prop(value) {
                    trace!("replacing {:?} with {:?}", rval, value);
                    self.replace_with_const(rval, value, source_info);
                    if can_const_prop == ConstPropMode::FullConstProp
                        || can_const_prop == ConstPropMode::OnlyInsideOwnBlock
                    {
                        trace!("propagated into {:?}", place);
                    }
                }
                match can_const_prop {
                    ConstPropMode::OnlyInsideOwnBlock => {
                        trace!(
                            "found local restricted to its block. \
                                Will remove it from const-prop after block is finished. Local: {:?}",
                            place.local
                        );
                    }
                    ConstPropMode::OnlyPropagateInto | ConstPropMode::NoPropagation => {
                        trace!("can't propagate into {:?}", place);
                        if place.local != RETURN_PLACE {
                            Self::remove_const(&mut self.ecx, place.local);
                        }
                    }
                    ConstPropMode::FullConstProp => {}
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
        } else {
            match statement.kind {
                StatementKind::SetDiscriminant { ref place, .. } => {
                    match self.ecx.machine.can_const_prop[place.local] {
                        ConstPropMode::FullConstProp | ConstPropMode::OnlyInsideOwnBlock => {
                            if self.use_ecx(|this| this.ecx.statement(statement)).is_some() {
                                trace!("propped discriminant into {:?}", place);
                            } else {
                                Self::remove_const(&mut self.ecx, place.local);
                            }
                        }
                        ConstPropMode::OnlyPropagateInto | ConstPropMode::NoPropagation => {
                            Self::remove_const(&mut self.ecx, place.local);
                        }
                    }
                }
                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                    let frame = self.ecx.frame_mut();
                    frame.locals[local].value =
                        if let StatementKind::StorageLive(_) = statement.kind {
                            LocalValue::Live(interpret::Operand::Immediate(
                                interpret::Immediate::Uninit,
                            ))
                        } else {
                            LocalValue::Dead
                        };
                }
                _ => {}
            }
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, location: Location) {
        let source_info = terminator.source_info;
        self.source_info = Some(source_info);
        self.super_terminator(terminator, location);
        // Do NOT early return in this function, it does some crucial fixup of the state at the end!
        match &mut terminator.kind {
            TerminatorKind::Assert { expected, ref mut cond, .. } => {
                if let Some(ref value) = self.eval_operand(&cond)
                    // FIXME should be used use_ecx rather than a local match... but we have
                    // quite a few of these read_scalar/read_immediate that need fixing.
                    && let Ok(value_const) = self.ecx.read_scalar(&value)
                    && self.should_const_prop(value)
                {
                    trace!("assertion on {:?} should be {:?}", value, expected);
                    *cond = self.operand_from_scalar(
                        value_const,
                        self.tcx.types.bool,
                        source_info.span,
                    );
                }
            }
            TerminatorKind::SwitchInt { ref mut discr, .. } => {
                // FIXME: This is currently redundant with `visit_operand`, but sadly
                // always visiting operands currently causes a perf regression in LLVM codegen, so
                // `visit_operand` currently only runs for propagates places for `mir_opt_level=4`.
                self.propagate_operand(discr)
            }
            // None of these have Operands to const-propagate.
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. } => {}
            // Every argument in our function calls have already been propagated in `visit_operand`.
            //
            // NOTE: because LLVM codegen gives slight performance regressions with it, so this is
            // gated on `mir_opt_level=3`.
            TerminatorKind::Call { .. } => {}
        }

        // We remove all Locals which are restricted in propagation to their containing blocks and
        // which were modified in the current block.
        // Take it out of the ecx so we can get a mutable reference to the ecx for `remove_const`.
        let mut locals = std::mem::take(&mut self.ecx.machine.written_only_inside_own_block_locals);
        for &local in locals.iter() {
            Self::remove_const(&mut self.ecx, local);
        }
        locals.clear();
        // Put it back so we reuse the heap of the storage
        self.ecx.machine.written_only_inside_own_block_locals = locals;
        if cfg!(debug_assertions) {
            // Ensure we are correctly erasing locals with the non-debug-assert logic.
            for local in self.ecx.machine.only_propagate_inside_block_locals.iter() {
                assert!(
                    self.get_const(local.into()).is_none()
                        || self
                            .layout_of(self.local_decls[local].ty)
                            .map_or(true, |layout| layout.is_zst())
                )
            }
        }
    }
}
