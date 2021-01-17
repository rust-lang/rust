//! Propagates constants for early reporting of statically known
//! assertion failures

use std::cell::Cell;

use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::HirId;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{
    AssertKind, BasicBlock, BinOp, Body, ClearCrossCrate, Constant, Local, LocalDecl, LocalKind,
    Location, Operand, Place, Rvalue, SourceInfo, SourceScope, SourceScopeData, Statement,
    StatementKind, Terminator, TerminatorKind, UnOp, RETURN_PLACE,
};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutError, TyAndLayout};
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{
    self, ConstInt, ConstKind, Instance, ParamEnv, ScalarInt, Ty, TyCtxt, TypeFoldable,
};
use rustc_session::lint;
use rustc_span::{def_id::DefId, Span};
use rustc_target::abi::{HasDataLayout, LayoutOf, Size, TargetDataLayout};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;

use crate::const_eval::ConstEvalErr;
use crate::interpret::{
    self, compile_time_machine, AllocId, Allocation, ConstValue, CtfeValidationMode, Frame, ImmTy,
    Immediate, InterpCx, InterpResult, LocalState, LocalValue, MemPlace, Memory, MemoryKind, OpTy,
    Operand as InterpOperand, PlaceTy, Pointer, Scalar, ScalarMaybeUninit, StackPopCleanup,
};
use crate::transform::MirPass;

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
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if body.source.promoted.is_some() {
            return;
        }

        use rustc_middle::hir::map::blocks::FnLikeNode;
        let def_id = body.source.def_id().expect_local();
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();
        let is_assoc_const = tcx.def_kind(def_id.to_def_id()) == DefKind::AssocConst;

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("ConstProp skipped for {:?}", def_id);
            return;
        }

        let is_generator = tcx.type_of(def_id.to_def_id()).is_generator();
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
            body.basic_blocks().clone(),
            body.source_scopes.clone(),
            body.local_decls.clone(),
            Default::default(),
            body.arg_count,
            Default::default(),
            body.span,
            body.generator_kind,
        );

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require an uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder = ConstPropagator::new(body, dummy_body, tcx);
        optimization_finder.visit_body(body);

        trace!("ConstProp done for {:?}", def_id);
    }
}

struct ConstPropMachine<'mir, 'tcx> {
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx, (), ()>>,
    /// `OnlyInsideOwnBlock` locals that were written in the current block get erased at the end.
    written_only_inside_own_block_locals: FxHashSet<Local>,
    /// Locals that need to be cleared after every block terminates.
    only_propagate_inside_block_locals: BitSet<Local>,
    can_const_prop: IndexVec<Local, ConstPropMode>,
}

impl<'mir, 'tcx> ConstPropMachine<'mir, 'tcx> {
    fn new(
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

    type MemoryKind = !;

    type MemoryExtra = ();

    fn load_mir(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx Body<'tcx>> {
        throw_machine_stop_str!("calling functions isn't supported in ConstProp")
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _abi: Abi,
        _args: &[OpTy<'tcx>],
        _ret: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _unwind: Option<BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir Body<'tcx>>> {
        Ok(None)
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _ret: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _unwind: Option<BasicBlock>,
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

    fn ptr_to_int(_mem: &Memory<'mir, 'tcx, Self>, _ptr: Pointer) -> InterpResult<'tcx, u64> {
        throw_unsup!(ReadPointerAsBytes)
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: ImmTy<'tcx>,
        _right: ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        // We can't do this because aliasing of memory can differ between const eval and llvm
        throw_machine_stop_str!("pointer arithmetic or comparisons aren't supported in ConstProp")
    }

    fn box_alloc(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _dest: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_machine_stop_str!("can't const prop heap allocations")
    }

    fn access_local(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        frame: &Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>,
        local: Local,
    ) -> InterpResult<'tcx, InterpOperand<Self::PointerTag>> {
        let l = &frame.locals[local];

        if l.value == LocalValue::Uninitialized {
            throw_machine_stop_str!("tried to access an uninitialized local")
        }

        l.access()
    }

    fn access_local_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
        frame: usize,
        local: Local,
    ) -> InterpResult<'tcx, Result<&'a mut LocalValue<Self::PointerTag>, MemPlace<Self::PointerTag>>>
    {
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
        _memory_extra: &(),
        _alloc_id: AllocId,
        allocation: &Allocation<Self::PointerTag, Self::AllocExtra>,
        _static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        if is_write {
            throw_machine_stop_str!("can't write to global");
        }
        // If the static allocation is mutable, then we can't const prop it as its content
        // might be different at runtime.
        if allocation.mutability == Mutability::Mut {
            throw_machine_stop_str!("can't access mutable globals in ConstProp");
        }

        Ok(())
    }

    #[inline(always)]
    fn init_frame_extra(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx>> {
        Ok(frame)
    }

    #[inline(always)]
    fn stack(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }
}

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    // FIXME(eddyb) avoid cloning these two fields more than once,
    // by accessing them through `ecx` instead.
    source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    // Because we have `MutVisitor` we can't obtain the `SourceInfo` from a `Location`. So we store
    // the last known `SourceInfo` here and just keep revisiting it.
    source_info: Option<SourceInfo>,
}

impl<'mir, 'tcx> LayoutOf for ConstPropagator<'mir, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyAndLayout = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyAndLayout {
        self.tcx.layout_of(self.param_env.and(ty))
    }
}

impl<'mir, 'tcx> HasDataLayout for ConstPropagator<'mir, 'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'mir, 'tcx> HasTyCtxt<'tcx> for ConstPropagator<'mir, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
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

        let span = tcx.def_span(def_id);
        // FIXME: `CanConstProp::check` computes the layout of all locals, return those layouts
        // so we can write them to `ecx.frame_mut().locals.layout, reducing the duplication in
        // `layout_of` query invocations.
        let can_const_prop = CanConstProp::check(tcx, param_env, body);
        let mut only_propagate_inside_block_locals = BitSet::new_empty(can_const_prop.len());
        for (l, mode) in can_const_prop.iter_enumerated() {
            if *mode == ConstPropMode::OnlyInsideOwnBlock {
                only_propagate_inside_block_locals.insert(l);
            }
        }
        let mut ecx = InterpCx::new(
            tcx,
            span,
            param_env,
            ConstPropMachine::new(only_propagate_inside_block_locals, can_const_prop),
            (),
        );

        let ret = ecx
            .layout_of(body.return_ty().subst(tcx, substs))
            .ok()
            // Don't bother allocating memory for ZST types which have no values
            // or for large values.
            .filter(|ret_layout| {
                !ret_layout.is_zst() && ret_layout.size < Size::from_bytes(MAX_ALLOC_LIMIT)
            })
            .map(|ret_layout| ecx.allocate(ret_layout, MemoryKind::Stack));

        ecx.push_stack_frame(
            Instance::new(def_id, substs),
            dummy_body,
            ret.map(Into::into),
            StackPopCleanup::None { cleanup: false },
        )
        .expect("failed to push initial stack frame");

        ConstPropagator {
            ecx,
            tcx,
            param_env,
            // FIXME(eddyb) avoid cloning these two fields more than once,
            // by accessing them through `ecx` instead.
            source_scopes: body.source_scopes.clone(),
            //FIXME(wesleywiser) we can't steal this because `Visitor::super_visit_body()` needs it
            local_decls: body.local_decls.clone(),
            source_info: None,
        }
    }

    fn get_const(&self, place: Place<'tcx>) -> Option<OpTy<'tcx>> {
        let op = match self.ecx.eval_place_to_op(place, None) {
            Ok(op) => op,
            Err(e) => {
                trace!("get_const failed: {}", e);
                return None;
            }
        };

        // Try to read the local as an immediate so that if it is representable as a scalar, we can
        // handle it as such, but otherwise, just return the value as is.
        Some(match self.ecx.try_read_immediate(op) {
            Ok(Ok(imm)) => imm.into(),
            _ => op,
        })
    }

    /// Remove `local` from the pool of `Locals`. Allows writing to them,
    /// but not reading from them anymore.
    fn remove_const(ecx: &mut InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>, local: Local) {
        ecx.frame_mut().locals[local] =
            LocalState { value: LocalValue::Uninitialized, layout: Cell::new(None) };
    }

    fn lint_root(&self, source_info: SourceInfo) -> Option<HirId> {
        match &self.source_scopes[source_info.scope].local_data {
            ClearCrossCrate::Set(data) => Some(data.lint_root),
            ClearCrossCrate::Clear => None,
        }
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
                    !error.kind.allocates(),
                    "const-prop encountered allocating error: {}",
                    error
                );
                None
            }
        }
    }

    /// Returns the value, if any, of evaluating `c`.
    fn eval_constant(&mut self, c: &Constant<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        // FIXME we need to revisit this for #67176
        if c.needs_subst() {
            return None;
        }

        match self.ecx.const_to_op(c.literal, None) {
            Ok(op) => Some(op),
            Err(error) => {
                let tcx = self.ecx.tcx.at(c.span);
                let err = ConstEvalErr::new(&self.ecx, error, Some(c.span));
                if let Some(lint_root) = self.lint_root(source_info) {
                    let lint_only = match c.literal.val {
                        // Promoteds must lint and not error as the user didn't ask for them
                        ConstKind::Unevaluated(_, _, Some(_)) => true,
                        // Out of backwards compatibility we cannot report hard errors in unused
                        // generic functions using associated constants of the generic parameters.
                        _ => c.literal.needs_subst(),
                    };
                    if lint_only {
                        // Out of backwards compatibility we cannot report hard errors in unused
                        // generic functions using associated constants of the generic parameters.
                        err.report_as_lint(tcx, "erroneous constant used", lint_root, Some(c.span));
                    } else {
                        err.report_as_error(tcx, "erroneous constant used");
                    }
                } else {
                    err.report_as_error(tcx, "erroneous constant used");
                }
                None
            }
        }
    }

    /// Returns the value, if any, of evaluating `place`.
    fn eval_place(&mut self, place: Place<'tcx>) -> Option<OpTy<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(|this| this.ecx.eval_place_to_op(place, None))
    }

    /// Returns the value, if any, of evaluating `op`. Calls upon `eval_constant`
    /// or `eval_place`, depending on the variant of `Operand` used.
    fn eval_operand(&mut self, op: &Operand<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c, source_info),
            Operand::Move(place) | Operand::Copy(place) => self.eval_place(place),
        }
    }

    fn report_assert_as_lint(
        &self,
        lint: &'static lint::Lint,
        source_info: SourceInfo,
        message: &'static str,
        panic: AssertKind<impl std::fmt::Debug>,
    ) -> Option<()> {
        let lint_root = self.lint_root(source_info)?;
        self.tcx.struct_span_lint_hir(lint, lint_root, source_info.span, |lint| {
            let mut err = lint.build(message);
            err.span_label(source_info.span, format!("{:?}", panic));
            err.emit()
        });
        None
    }

    fn check_unary_op(
        &mut self,
        op: UnOp,
        arg: &Operand<'tcx>,
        source_info: SourceInfo,
    ) -> Option<()> {
        if let (val, true) = self.use_ecx(|this| {
            let val = this.ecx.read_immediate(this.ecx.eval_operand(arg, None)?)?;
            let (_res, overflow, _ty) = this.ecx.overflowing_unary_op(op, val)?;
            Ok((val, overflow))
        })? {
            // `AssertKind` only has an `OverflowNeg` variant, so make sure that is
            // appropriate to use.
            assert_eq!(op, UnOp::Neg, "Neg is the only UnOp that can overflow");
            self.report_assert_as_lint(
                lint::builtin::ARITHMETIC_OVERFLOW,
                source_info,
                "this arithmetic operation will overflow",
                AssertKind::OverflowNeg(val.to_const_int()),
            )?;
        }

        Some(())
    }

    fn check_binary_op(
        &mut self,
        op: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
        source_info: SourceInfo,
    ) -> Option<()> {
        let r = self.use_ecx(|this| this.ecx.read_immediate(this.ecx.eval_operand(right, None)?));
        let l = self.use_ecx(|this| this.ecx.read_immediate(this.ecx.eval_operand(left, None)?));
        // Check for exceeding shifts *even if* we cannot evaluate the LHS.
        if op == BinOp::Shr || op == BinOp::Shl {
            let r = r?;
            // We need the type of the LHS. We cannot use `place_layout` as that is the type
            // of the result, which for checked binops is not the same!
            let left_ty = left.ty(&self.local_decls, self.tcx);
            let left_size = self.ecx.layout_of(left_ty).ok()?.size;
            let right_size = r.layout.size;
            let r_bits = r.to_scalar().ok();
            // This is basically `force_bits`.
            let r_bits = r_bits.and_then(|r| r.to_bits_or_ptr(right_size, &self.tcx).ok());
            if r_bits.map_or(false, |b| b >= left_size.bits() as u128) {
                debug!("check_binary_op: reporting assert for {:?}", source_info);
                self.report_assert_as_lint(
                    lint::builtin::ARITHMETIC_OVERFLOW,
                    source_info,
                    "this arithmetic operation will overflow",
                    AssertKind::Overflow(
                        op,
                        match l {
                            Some(l) => l.to_const_int(),
                            // Invent a dummy value, the diagnostic ignores it anyway
                            None => ConstInt::new(
                                ScalarInt::try_from_uint(1_u8, left_size).unwrap(),
                                left_ty.is_signed(),
                                left_ty.is_ptr_sized_integral(),
                            ),
                        },
                        r.to_const_int(),
                    ),
                )?;
            }
        }

        if let (Some(l), Some(r)) = (l, r) {
            // The remaining operators are handled through `overflowing_binary_op`.
            if self.use_ecx(|this| {
                let (_res, overflow, _ty) = this.ecx.overflowing_binary_op(op, l, r)?;
                Ok(overflow)
            })? {
                self.report_assert_as_lint(
                    lint::builtin::ARITHMETIC_OVERFLOW,
                    source_info,
                    "this arithmetic operation will overflow",
                    AssertKind::Overflow(op, l.to_const_int(), r.to_const_int()),
                )?;
            }
        }
        Some(())
    }

    fn propagate_operand(&mut self, operand: &mut Operand<'tcx>) {
        match *operand {
            Operand::Copy(l) | Operand::Move(l) => {
                if let Some(value) = self.get_const(l) {
                    if self.should_const_prop(value) {
                        // FIXME(felix91gr): this code only handles `Scalar` cases.
                        // For now, we're not handling `ScalarPair` cases because
                        // doing so here would require a lot of code duplication.
                        // We should hopefully generalize `Operand` handling into a fn,
                        // and use it to do const-prop here and everywhere else
                        // where it makes sense.
                        if let interpret::Operand::Immediate(interpret::Immediate::Scalar(
                            ScalarMaybeUninit::Scalar(scalar),
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
            }
            Operand::Constant(_) => (),
        }
    }

    fn const_prop(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        source_info: SourceInfo,
        place: Place<'tcx>,
    ) -> Option<()> {
        // Perform any special handling for specific Rvalue types.
        // Generally, checks here fall into one of two categories:
        //   1. Additional checking to provide useful lints to the user
        //        - In this case, we will do some validation and then fall through to the
        //          end of the function which evals the assignment.
        //   2. Working around bugs in other parts of the compiler
        //        - In this case, we'll return `None` from this function to stop evaluation.
        match rvalue {
            // Additional checking: give lints to the user if an overflow would occur.
            // We do this here and not in the `Assert` terminator as that terminator is
            // only sometimes emitted (overflow checks can be disabled), but we want to always
            // lint.
            Rvalue::UnaryOp(op, arg) => {
                trace!("checking UnaryOp(op = {:?}, arg = {:?})", op, arg);
                self.check_unary_op(*op, arg, source_info)?;
            }
            Rvalue::BinaryOp(op, left, right) => {
                trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op, left, right);
                self.check_binary_op(*op, left, right, source_info)?;
            }
            Rvalue::CheckedBinaryOp(op, left, right) => {
                trace!(
                    "checking CheckedBinaryOp(op = {:?}, left = {:?}, right = {:?})",
                    op,
                    left,
                    right
                );
                self.check_binary_op(*op, left, right, source_info)?;
            }

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
            | Rvalue::Repeat(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..) => {}
        }

        // FIXME we need to revisit this for #67176
        if rvalue.needs_subst() {
            return None;
        }

        if self.tcx.sess.opts.debugging_opts.mir_opt_level >= 3 {
            self.eval_rvalue_with_identities(rvalue, place)
        } else {
            self.use_ecx(|this| this.ecx.eval_rvalue_into_place(rvalue, place))
        }
    }

    // Attempt to use albegraic identities to eliminate constant expressions
    fn eval_rvalue_with_identities(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place: Place<'tcx>,
    ) -> Option<()> {
        self.use_ecx(|this| {
            match rvalue {
                Rvalue::BinaryOp(op, left, right) | Rvalue::CheckedBinaryOp(op, left, right) => {
                    let l = this.ecx.eval_operand(left, None);
                    let r = this.ecx.eval_operand(right, None);

                    let const_arg = match (l, r) {
                        (Ok(x), Err(_)) | (Err(_), Ok(x)) => this.ecx.read_immediate(x)?,
                        (Err(e), Err(_)) => return Err(e),
                        (Ok(_), Ok(_)) => {
                            this.ecx.eval_rvalue_into_place(rvalue, place)?;
                            return Ok(());
                        }
                    };

                    let arg_value =
                        this.ecx.force_bits(const_arg.to_scalar()?, const_arg.layout.size)?;
                    let dest = this.ecx.eval_place(place)?;

                    match op {
                        BinOp::BitAnd => {
                            if arg_value == 0 {
                                this.ecx.write_immediate(*const_arg, dest)?;
                            }
                        }
                        BinOp::BitOr => {
                            if arg_value == const_arg.layout.size.truncate(u128::MAX)
                                || (const_arg.layout.ty.is_bool() && arg_value == 1)
                            {
                                this.ecx.write_immediate(*const_arg, dest)?;
                            }
                        }
                        BinOp::Mul => {
                            if const_arg.layout.ty.is_integral() && arg_value == 0 {
                                if let Rvalue::CheckedBinaryOp(_, _, _) = rvalue {
                                    let val = Immediate::ScalarPair(
                                        const_arg.to_scalar()?.into(),
                                        Scalar::from_bool(false).into(),
                                    );
                                    this.ecx.write_immediate(val, dest)?;
                                } else {
                                    this.ecx.write_immediate(*const_arg, dest)?;
                                }
                            }
                        }
                        _ => {
                            this.ecx.eval_rvalue_into_place(rvalue, place)?;
                        }
                    }
                }
                _ => {
                    this.ecx.eval_rvalue_into_place(rvalue, place)?;
                }
            }

            Ok(())
        })
    }

    /// Creates a new `Operand::Constant` from a `Scalar` value
    fn operand_from_scalar(&self, scalar: Scalar, ty: Ty<'tcx>, span: Span) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ty::Const::from_scalar(self.tcx, scalar, ty),
        }))
    }

    fn replace_with_const(
        &mut self,
        rval: &mut Rvalue<'tcx>,
        value: OpTy<'tcx>,
        source_info: SourceInfo,
    ) {
        if let Rvalue::Use(Operand::Constant(c)) = rval {
            if !matches!(c.literal.val, ConstKind::Unevaluated(..)) {
                trace!("skipping replace of Rvalue::Use({:?} because it is already a const", c);
                return;
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

        // FIXME> figure out what to do when try_read_immediate fails
        let imm = self.use_ecx(|this| this.ecx.try_read_immediate(value));

        if let Some(Ok(imm)) = imm {
            match *imm {
                interpret::Immediate::Scalar(ScalarMaybeUninit::Scalar(scalar)) => {
                    *rval = Rvalue::Use(self.operand_from_scalar(
                        scalar,
                        value.layout.ty,
                        source_info.span,
                    ));
                }
                Immediate::ScalarPair(
                    ScalarMaybeUninit::Scalar(_),
                    ScalarMaybeUninit::Scalar(_),
                ) => {
                    // Found a value represented as a pair. For now only do const-prop if the type
                    // of `rvalue` is also a tuple with two scalars.
                    // FIXME: enable the general case stated above ^.
                    let ty = &value.layout.ty;
                    // Only do it for tuples
                    if let ty::Tuple(substs) = ty.kind() {
                        // Only do it if tuple is also a pair with two scalars
                        if substs.len() == 2 {
                            let alloc = self.use_ecx(|this| {
                                let ty1 = substs[0].expect_ty();
                                let ty2 = substs[1].expect_ty();
                                let ty_is_scalar = |ty| {
                                    this.ecx.layout_of(ty).ok().map(|layout| layout.abi.is_scalar())
                                        == Some(true)
                                };
                                if ty_is_scalar(ty1) && ty_is_scalar(ty2) {
                                    let alloc = this
                                        .ecx
                                        .intern_with_temp_alloc(value.layout, |ecx, dest| {
                                            ecx.write_immediate_to_mplace(*imm, dest)
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
                                *rval = Rvalue::Use(Operand::Constant(Box::new(Constant {
                                    span: source_info.span,
                                    user_ty: None,
                                    literal: self.ecx.tcx.mk_const(ty::Const {
                                        ty,
                                        val: ty::ConstKind::Value(ConstValue::ByRef {
                                            alloc,
                                            offset: Size::ZERO,
                                        }),
                                    }),
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
    fn should_const_prop(&mut self, op: OpTy<'tcx>) -> bool {
        let mir_opt_level = self.tcx.sess.opts.debugging_opts.mir_opt_level;

        if mir_opt_level == 0 {
            return false;
        }

        if !self.tcx.consider_optimizing(|| format!("ConstantPropagation - OpTy: {:?}", op)) {
            return false;
        }

        match *op {
            interpret::Operand::Immediate(Immediate::Scalar(ScalarMaybeUninit::Scalar(s))) => {
                s.is_bits()
            }
            interpret::Operand::Immediate(Immediate::ScalarPair(
                ScalarMaybeUninit::Scalar(l),
                ScalarMaybeUninit::Scalar(r),
            )) => l.is_bits() && r.is_bits(),
            _ => false,
        }
    }
}

/// The mode that `ConstProp` is allowed to run in for a given `Local`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ConstPropMode {
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

struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // False at the beginning. Once set, no more assignments are allowed to that local.
    found_assignment: BitSet<Local>,
    // Cache of locals' information
    local_kinds: IndexVec<Local, LocalKind>,
}

impl CanConstProp {
    /// Returns true if `local` can be propagated
    fn check(
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

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;
        match context {
            // Projections are fine, because `&mut foo.x` will be caught by
            // `MutatingUseContext::Borrow` elsewhere.
            MutatingUse(MutatingUseContext::Projection)
            // These are just stores, where the storing is not propagatable, but there may be later
            // mutations of the same local via `Store`
            | MutatingUse(MutatingUseContext::Call)
            // Actual store that can possibly even propagate a value
            | MutatingUse(MutatingUseContext::Store) => {
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
            MutatingUse(MutatingUseContext::AsmOutput)
            | MutatingUse(MutatingUseContext::Yield)
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

impl<'mir, 'tcx> MutVisitor<'tcx> for ConstPropagator<'mir, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_body(&mut self, body: &mut Body<'tcx>) {
        for (bb, data) in body.basic_blocks_mut().iter_enumerated_mut() {
            self.visit_basic_block_data(bb, data);
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);

        // Only const prop copies and moves on `mir_opt_level=2` as doing so
        // currently slightly increases compile time in some cases.
        if self.tcx.sess.opts.debugging_opts.mir_opt_level >= 2 {
            self.propagate_operand(operand)
        }
    }

    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, location: Location) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant, self.source_info.unwrap());
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);
        let source_info = statement.source_info;
        self.source_info = Some(source_info);
        if let StatementKind::Assign(box (place, ref mut rval)) = statement.kind {
            let can_const_prop = self.ecx.machine.can_const_prop[place.local];
            if let Some(()) = self.const_prop(rval, source_info, place) {
                // This will return None if the above `const_prop` invocation only "wrote" a
                // type whose creation requires no write. E.g. a generator whose initial state
                // consists solely of uninitialized memory (so it doesn't capture any locals).
                if let Some(value) = self.get_const(place) {
                    if self.should_const_prop(value) {
                        trace!("replacing {:?} with {:?}", rval, value);
                        self.replace_with_const(rval, value, source_info);
                        if can_const_prop == ConstPropMode::FullConstProp
                            || can_const_prop == ConstPropMode::OnlyInsideOwnBlock
                        {
                            trace!("propagated into {:?}", place);
                        }
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
                            LocalValue::Uninitialized
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
        match &mut terminator.kind {
            TerminatorKind::Assert { expected, ref msg, ref mut cond, .. } => {
                if let Some(value) = self.eval_operand(&cond, source_info) {
                    trace!("assertion on {:?} should be {:?}", value, expected);
                    let expected = ScalarMaybeUninit::from(Scalar::from_bool(*expected));
                    let value_const = self.ecx.read_scalar(value).unwrap();
                    if expected != value_const {
                        enum DbgVal<T> {
                            Val(T),
                            Underscore,
                        }
                        impl<T: std::fmt::Debug> std::fmt::Debug for DbgVal<T> {
                            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                                match self {
                                    Self::Val(val) => val.fmt(fmt),
                                    Self::Underscore => fmt.write_str("_"),
                                }
                            }
                        }
                        let mut eval_to_int = |op| {
                            // This can be `None` if the lhs wasn't const propagated and we just
                            // triggered the assert on the value of the rhs.
                            match self.eval_operand(op, source_info) {
                                Some(op) => {
                                    DbgVal::Val(self.ecx.read_immediate(op).unwrap().to_const_int())
                                }
                                None => DbgVal::Underscore,
                            }
                        };
                        let msg = match msg {
                            AssertKind::DivisionByZero(op) => {
                                Some(AssertKind::DivisionByZero(eval_to_int(op)))
                            }
                            AssertKind::RemainderByZero(op) => {
                                Some(AssertKind::RemainderByZero(eval_to_int(op)))
                            }
                            AssertKind::BoundsCheck { ref len, ref index } => {
                                let len = eval_to_int(len);
                                let index = eval_to_int(index);
                                Some(AssertKind::BoundsCheck { len, index })
                            }
                            // Overflow is are already covered by checks on the binary operators.
                            AssertKind::Overflow(..) | AssertKind::OverflowNeg(_) => None,
                            // Need proper const propagator for these.
                            _ => None,
                        };
                        // Poison all places this operand references so that further code
                        // doesn't use the invalid value
                        match cond {
                            Operand::Move(ref place) | Operand::Copy(ref place) => {
                                Self::remove_const(&mut self.ecx, place.local);
                            }
                            Operand::Constant(_) => {}
                        }
                        if let Some(msg) = msg {
                            self.report_assert_as_lint(
                                lint::builtin::UNCONDITIONAL_PANIC,
                                source_info,
                                "this operation will panic at runtime",
                                msg,
                            );
                        }
                    } else {
                        if self.should_const_prop(value) {
                            if let ScalarMaybeUninit::Scalar(scalar) = value_const {
                                *cond = self.operand_from_scalar(
                                    scalar,
                                    self.tcx.types.bool,
                                    source_info.span,
                                );
                            }
                        }
                    }
                }
            }
            TerminatorKind::SwitchInt { ref mut discr, .. } => {
                // FIXME: This is currently redundant with `visit_operand`, but sadly
                // always visiting operands currently causes a perf regression in LLVM codegen, so
                // `visit_operand` currently only runs for propagates places for `mir_opt_level=3`.
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
            // gated on `mir_opt_level=2`.
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
