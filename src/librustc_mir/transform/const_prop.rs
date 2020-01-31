//! Propagates constants for early reporting of statically known
//! assertion failures

use std::borrow::Cow;
use std::cell::Cell;

use rustc::mir::interpret::{InterpResult, PanicInfo, Scalar};
use rustc::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor,
};
use rustc::mir::{
    read_only, AggregateKind, BasicBlock, BinOp, Body, BodyAndCache, ClearCrossCrate, Constant,
    Local, LocalDecl, LocalKind, Location, Operand, Place, ReadOnlyBodyAndCache, Rvalue,
    SourceInfo, SourceScope, SourceScopeData, Statement, StatementKind, Terminator, TerminatorKind,
    UnOp, RETURN_PLACE,
};
use rustc::traits;
use rustc::ty::layout::{
    HasDataLayout, HasTyCtxt, LayoutError, LayoutOf, Size, TargetDataLayout, TyLayout,
};
use rustc::ty::subst::{InternalSubsts, Subst};
use rustc::ty::{self, ConstKind, Instance, ParamEnv, Ty, TyCtxt, TypeFoldable};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_index::vec::IndexVec;
use rustc_span::{Span, DUMMY_SP};
use syntax::ast::Mutability;

use crate::const_eval::error_to_const_error;
use crate::interpret::{
    self, intern_const_alloc_recursive, AllocId, Allocation, Frame, ImmTy, Immediate, InternKind,
    InterpCx, LocalState, LocalValue, Memory, MemoryKind, OpTy, Operand as InterpOperand, PlaceTy,
    Pointer, ScalarMaybeUndef, StackPopCleanup,
};
use crate::transform::{MirPass, MirSource};

/// The maximum number of bytes that we'll allocate space for a return value.
const MAX_ALLOC_LIMIT: u64 = 1024;

pub struct ConstProp;

impl<'tcx> MirPass<'tcx> for ConstProp {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if source.promoted.is_some() {
            return;
        }

        use rustc::hir::map::blocks::FnLikeNode;
        let hir_id = tcx
            .hir()
            .as_local_hir_id(source.def_id())
            .expect("Non-local call to local provider is_const_fn");

        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();
        let is_assoc_const = match tcx.def_kind(source.def_id()) {
            Some(DefKind::AssocConst) => true,
            _ => false,
        };

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("ConstProp skipped for {:?}", source.def_id());
            return;
        }

        let is_generator = tcx.type_of(source.def_id()).is_generator();
        // FIXME(welseywiser) const prop doesn't work on generators because of query cycles
        // computing their layout.
        if is_generator {
            trace!("ConstProp skipped for generator {:?}", source.def_id());
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
            .predicates_of(source.def_id())
            .predicates
            .iter()
            .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None })
            .collect();
        if !traits::normalize_and_test_predicates(
            tcx,
            traits::elaborate_predicates(tcx, predicates).collect(),
        ) {
            trace!("ConstProp skipped for {:?}: found unsatisfiable predicates", source.def_id());
            return;
        }

        trace!("ConstProp starting for {:?}", source.def_id());

        let dummy_body = &Body::new(
            body.basic_blocks().clone(),
            body.source_scopes.clone(),
            body.local_decls.clone(),
            Default::default(),
            body.arg_count,
            Default::default(),
            tcx.def_span(source.def_id()),
            Default::default(),
            body.generator_kind,
        );

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require an uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder =
            ConstPropagator::new(read_only!(body), dummy_body, tcx, source);
        optimization_finder.visit_body(body);

        trace!("ConstProp done for {:?}", source.def_id());
    }
}

struct ConstPropMachine;

impl<'mir, 'tcx> interpret::Machine<'mir, 'tcx> for ConstPropMachine {
    type MemoryKinds = !;
    type PointerTag = ();
    type ExtraFnVal = !;

    type FrameExtra = ();
    type MemoryExtra = ();
    type AllocExtra = ();

    type MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation)>;

    const STATIC_KIND: Option<!> = None;

    const CHECK_ALIGN: bool = false;

    #[inline(always)]
    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        false
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _span: Span,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _ret: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _unwind: Option<BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir Body<'tcx>>> {
        Ok(None)
    }

    fn call_extra_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: !,
        _args: &[OpTy<'tcx>],
        _ret: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _unwind: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        match fn_val {}
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _span: Span,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _ret: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _unwind: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        throw_unsup!(ConstPropUnsupported("calling intrinsics isn't supported in ConstProp"));
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _span: Span,
        _msg: &rustc::mir::interpret::AssertMessage<'tcx>,
        _unwind: Option<rustc::mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        bug!("panics terminators are not evaluated in ConstProp");
    }

    fn ptr_to_int(_mem: &Memory<'mir, 'tcx, Self>, _ptr: Pointer) -> InterpResult<'tcx, u64> {
        throw_unsup!(ConstPropUnsupported("ptr-to-int casts aren't supported in ConstProp"));
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: ImmTy<'tcx>,
        _right: ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        // We can't do this because aliasing of memory can differ between const eval and llvm
        throw_unsup!(ConstPropUnsupported(
            "pointer arithmetic or comparisons aren't supported \
            in ConstProp"
        ));
    }

    fn find_foreign_static(
        _tcx: TyCtxt<'tcx>,
        _def_id: DefId,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<Self::PointerTag>>> {
        throw_unsup!(ReadForeignStatic)
    }

    #[inline(always)]
    fn init_allocation_extra<'b>(
        _memory_extra: &(),
        _id: AllocId,
        alloc: Cow<'b, Allocation>,
        _kind: Option<MemoryKind<!>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag>>, Self::PointerTag) {
        // We do not use a tag so we can just cheaply forward the allocation
        (alloc, ())
    }

    #[inline(always)]
    fn tag_static_base_pointer(_memory_extra: &(), _id: AllocId) -> Self::PointerTag {
        ()
    }

    fn box_alloc(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _dest: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup!(ConstPropUnsupported("can't const prop `box` keyword"));
    }

    fn access_local(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        frame: &Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>,
        local: Local,
    ) -> InterpResult<'tcx, InterpOperand<Self::PointerTag>> {
        let l = &frame.locals[local];

        if l.value == LocalValue::Uninitialized {
            throw_unsup!(ConstPropUnsupported("tried to access an uninitialized local"));
        }

        l.access()
    }

    fn before_access_static(
        _memory_extra: &(),
        allocation: &Allocation<Self::PointerTag, Self::AllocExtra>,
    ) -> InterpResult<'tcx> {
        // if the static allocation is mutable or if it has relocations (it may be legal to mutate
        // the memory behind that in the future), then we can't const prop it
        if allocation.mutability == Mutability::Mut || allocation.relocations().len() > 0 {
            throw_unsup!(ConstPropUnsupported("can't eval mutable statics in ConstProp"));
        }

        Ok(())
    }

    fn before_terminator(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    #[inline(always)]
    fn stack_push(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }
}

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine>,
    tcx: TyCtxt<'tcx>,
    source: MirSource<'tcx>,
    can_const_prop: IndexVec<Local, ConstPropMode>,
    param_env: ParamEnv<'tcx>,
    // FIXME(eddyb) avoid cloning these two fields more than once,
    // by accessing them through `ecx` instead.
    source_scopes: IndexVec<SourceScope, SourceScopeData>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    ret: Option<OpTy<'tcx, ()>>,
    // Because we have `MutVisitor` we can't obtain the `SourceInfo` from a `Location`. So we store
    // the last known `SourceInfo` here and just keep revisiting it.
    source_info: Option<SourceInfo>,
}

impl<'mir, 'tcx> LayoutOf for ConstPropagator<'mir, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
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
        body: ReadOnlyBodyAndCache<'_, 'tcx>,
        dummy_body: &'mir Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        source: MirSource<'tcx>,
    ) -> ConstPropagator<'mir, 'tcx> {
        let def_id = source.def_id();
        let substs = &InternalSubsts::identity_for_item(tcx, def_id);
        let mut param_env = tcx.param_env(def_id);

        // If we're evaluating inside a monomorphic function, then use `Reveal::All` because
        // we want to see the same instances that codegen will see. This allows us to `resolve()`
        // specializations.
        if !substs.needs_subst() {
            param_env = param_env.with_reveal_all();
        }

        let span = tcx.def_span(def_id);
        let mut ecx = InterpCx::new(tcx.at(span), param_env, ConstPropMachine, ());
        let can_const_prop = CanConstProp::check(body);

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
            span,
            dummy_body,
            ret.map(Into::into),
            StackPopCleanup::None { cleanup: false },
        )
        .expect("failed to push initial stack frame");

        ConstPropagator {
            ecx,
            tcx,
            source,
            param_env,
            can_const_prop,
            // FIXME(eddyb) avoid cloning these two fields more than once,
            // by accessing them through `ecx` instead.
            source_scopes: body.source_scopes.clone(),
            //FIXME(wesleywiser) we can't steal this because `Visitor::super_visit_body()` needs it
            local_decls: body.local_decls.clone(),
            ret: ret.map(Into::into),
            source_info: None,
        }
    }

    fn get_const(&self, local: Local) -> Option<OpTy<'tcx>> {
        if local == RETURN_PLACE {
            // Try to read the return place as an immediate so that if it is representable as a
            // scalar, we can handle it as such, but otherwise, just return the value as is.
            return match self.ret.map(|ret| self.ecx.try_read_immediate(ret)) {
                Some(Ok(Ok(imm))) => Some(imm.into()),
                _ => self.ret,
            };
        }

        self.ecx.access_local(self.ecx.frame(), local, None).ok()
    }

    fn remove_const(&mut self, local: Local) {
        self.ecx.frame_mut().locals[local] =
            LocalState { value: LocalValue::Uninitialized, layout: Cell::new(None) };
    }

    fn lint_root(&self, source_info: SourceInfo) -> Option<HirId> {
        match &self.source_scopes[source_info.scope].local_data {
            ClearCrossCrate::Set(data) => Some(data.lint_root),
            ClearCrossCrate::Clear => None,
        }
    }

    fn use_ecx<F, T>(&mut self, source_info: SourceInfo, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        self.ecx.tcx.span = source_info.span;
        // FIXME(eddyb) move this to the `Panic(_)` error case, so that
        // `f(self)` is always called, and that the only difference when the
        // scope's `local_data` is missing, is that the lint isn't emitted.
        let lint_root = self.lint_root(source_info)?;
        let r = match f(self) {
            Ok(val) => Some(val),
            Err(error) => {
                use rustc::mir::interpret::{
                    InterpError::*, UndefinedBehaviorInfo, UnsupportedOpInfo,
                };
                match error.kind {
                    MachineStop(_) => bug!("ConstProp does not stop"),

                    // Some error shouldn't come up because creating them causes
                    // an allocation, which we should avoid. When that happens,
                    // dedicated error variants should be introduced instead.
                    // Only test this in debug builds though to avoid disruptions.
                    Unsupported(UnsupportedOpInfo::Unsupported(_))
                    | Unsupported(UnsupportedOpInfo::ValidationFailure(_))
                    | UndefinedBehavior(UndefinedBehaviorInfo::Ub(_))
                    | UndefinedBehavior(UndefinedBehaviorInfo::UbExperimental(_))
                        if cfg!(debug_assertions) =>
                    {
                        bug!("const-prop encountered allocating error: {:?}", error.kind);
                    }

                    Unsupported(_)
                    | UndefinedBehavior(_)
                    | InvalidProgram(_)
                    | ResourceExhaustion(_) => {
                        // Ignore these errors.
                    }
                    Panic(_) => {
                        let diagnostic = error_to_const_error(&self.ecx, error);
                        diagnostic.report_as_lint(
                            self.ecx.tcx,
                            "this expression will panic at runtime",
                            lint_root,
                            None,
                        );
                    }
                }
                None
            }
        };
        self.ecx.tcx.span = DUMMY_SP;
        r
    }

    fn eval_constant(&mut self, c: &Constant<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        self.ecx.tcx.span = c.span;

        // FIXME we need to revisit this for #67176
        if c.needs_subst() {
            return None;
        }

        match self.ecx.eval_const_to_op(c.literal, None) {
            Ok(op) => Some(op),
            Err(error) => {
                let err = error_to_const_error(&self.ecx, error);
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
                        err.report_as_lint(
                            self.ecx.tcx,
                            "erroneous constant used",
                            lint_root,
                            Some(c.span),
                        );
                    } else {
                        err.report_as_error(self.ecx.tcx, "erroneous constant used");
                    }
                } else {
                    err.report_as_error(self.ecx.tcx, "erroneous constant used");
                }
                None
            }
        }
    }

    fn eval_place(&mut self, place: &Place<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(source_info, |this| this.ecx.eval_place_to_op(place, None))
    }

    fn eval_operand(&mut self, op: &Operand<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c, source_info),
            Operand::Move(ref place) | Operand::Copy(ref place) => {
                self.eval_place(place, source_info)
            }
        }
    }

    fn check_unary_op(&mut self, arg: &Operand<'tcx>, source_info: SourceInfo) -> Option<()> {
        self.use_ecx(source_info, |this| {
            let ty = arg.ty(&this.local_decls, this.tcx);

            if ty.is_integral() {
                let arg = this.ecx.eval_operand(arg, None)?;
                let prim = this.ecx.read_immediate(arg)?;
                // Need to do overflow check here: For actual CTFE, MIR
                // generation emits code that does this before calling the op.
                if prim.to_bits()? == (1 << (prim.layout.size.bits() - 1)) {
                    throw_panic!(OverflowNeg)
                }
            }

            Ok(())
        })?;

        Some(())
    }

    fn check_binary_op(
        &mut self,
        op: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
        source_info: SourceInfo,
        place_layout: TyLayout<'tcx>,
        overflow_check: bool,
    ) -> Option<()> {
        let r = self.use_ecx(source_info, |this| {
            this.ecx.read_immediate(this.ecx.eval_operand(right, None)?)
        })?;
        if op == BinOp::Shr || op == BinOp::Shl {
            let left_bits = place_layout.size.bits();
            let right_size = r.layout.size;
            let r_bits = r.to_scalar().and_then(|r| r.to_bits(right_size));
            if r_bits.map_or(false, |b| b >= left_bits as u128) {
                let lint_root = self.lint_root(source_info)?;
                let dir = if op == BinOp::Shr { "right" } else { "left" };
                self.tcx.struct_span_lint_hir(
                    ::rustc::lint::builtin::EXCEEDING_BITSHIFTS,
                    lint_root,
                    source_info.span,
                    |lint| lint.build(&format!("attempt to shift {} with overflow", dir)).emit(),
                );
                return None;
            }
        }

        // If overflow checking is enabled (like in debug mode by default),
        // then we'll already catch overflow when we evaluate the `Assert` statement
        // in MIR. However, if overflow checking is disabled, then there won't be any
        // `Assert` statement and so we have to do additional checking here.
        if !overflow_check {
            self.use_ecx(source_info, |this| {
                let l = this.ecx.read_immediate(this.ecx.eval_operand(left, None)?)?;
                let (_, overflow, _ty) = this.ecx.overflowing_binary_op(op, l, r)?;

                if overflow {
                    let err = err_panic!(Overflow(op)).into();
                    return Err(err);
                }

                Ok(())
            })?;
        }

        Some(())
    }

    fn const_prop(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place_layout: TyLayout<'tcx>,
        source_info: SourceInfo,
        place: &Place<'tcx>,
    ) -> Option<()> {
        // #66397: Don't try to eval into large places as that can cause an OOM
        if place_layout.size >= Size::from_bytes(MAX_ALLOC_LIMIT) {
            return None;
        }

        // FIXME we need to revisit this for #67176
        if rvalue.needs_subst() {
            return None;
        }

        let overflow_check = self.tcx.sess.overflow_checks();

        // Perform any special handling for specific Rvalue types.
        // Generally, checks here fall into one of two categories:
        //   1. Additional checking to provide useful lints to the user
        //        - In this case, we will do some validation and then fall through to the
        //          end of the function which evals the assignment.
        //   2. Working around bugs in other parts of the compiler
        //        - In this case, we'll return `None` from this function to stop evaluation.
        match rvalue {
            // Additional checking: if overflow checks are disabled (which is usually the case in
            // release mode), then we need to do additional checking here to give lints to the user
            // if an overflow would occur.
            Rvalue::UnaryOp(UnOp::Neg, arg) if !overflow_check => {
                trace!("checking UnaryOp(op = Neg, arg = {:?})", arg);
                self.check_unary_op(arg, source_info)?;
            }

            // Additional checking: check for overflows on integer binary operations and report
            // them to the user as lints.
            Rvalue::BinaryOp(op, left, right) => {
                trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op, left, right);
                self.check_binary_op(*op, left, right, source_info, place_layout, overflow_check)?;
            }

            // Do not try creating references (#67862)
            Rvalue::Ref(_, _, place_ref) => {
                trace!("skipping Ref({:?})", place_ref);

                return None;
            }

            _ => {}
        }

        self.use_ecx(source_info, |this| {
            trace!("calling eval_rvalue_into_place(rvalue = {:?}, place = {:?})", rvalue, place);
            this.ecx.eval_rvalue_into_place(rvalue, place)?;
            Ok(())
        })
    }

    fn operand_from_scalar(&self, scalar: Scalar, ty: Ty<'tcx>, span: Span) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: self.tcx.mk_const(*ty::Const::from_scalar(self.tcx, scalar, ty)),
        }))
    }

    fn replace_with_const(
        &mut self,
        rval: &mut Rvalue<'tcx>,
        value: OpTy<'tcx>,
        source_info: SourceInfo,
    ) {
        trace!("attepting to replace {:?} with {:?}", rval, value);
        if let Err(e) = self.ecx.validate_operand(
            value,
            vec![],
            // FIXME: is ref tracking too expensive?
            Some(&mut interpret::RefTracking::empty()),
        ) {
            trace!("validation error, attempt failed: {:?}", e);
            return;
        }

        // FIXME> figure out what tho do when try_read_immediate fails
        let imm = self.use_ecx(source_info, |this| this.ecx.try_read_immediate(value));

        if let Some(Ok(imm)) = imm {
            match *imm {
                interpret::Immediate::Scalar(ScalarMaybeUndef::Scalar(scalar)) => {
                    *rval = Rvalue::Use(self.operand_from_scalar(
                        scalar,
                        value.layout.ty,
                        source_info.span,
                    ));
                }
                Immediate::ScalarPair(
                    ScalarMaybeUndef::Scalar(one),
                    ScalarMaybeUndef::Scalar(two),
                ) => {
                    // Found a value represented as a pair. For now only do cont-prop if type of
                    // Rvalue is also a pair with two scalars. The more general case is more
                    // complicated to implement so we'll do it later.
                    let ty = &value.layout.ty.kind;
                    // Only do it for tuples
                    if let ty::Tuple(substs) = ty {
                        // Only do it if tuple is also a pair with two scalars
                        if substs.len() == 2 {
                            let opt_ty1_ty2 = self.use_ecx(source_info, |this| {
                                let ty1 = substs[0].expect_ty();
                                let ty2 = substs[1].expect_ty();
                                let ty_is_scalar = |ty| {
                                    this.ecx.layout_of(ty).ok().map(|ty| ty.details.abi.is_scalar())
                                        == Some(true)
                                };
                                if ty_is_scalar(ty1) && ty_is_scalar(ty2) {
                                    Ok(Some((ty1, ty2)))
                                } else {
                                    Ok(None)
                                }
                            });

                            if let Some(Some((ty1, ty2))) = opt_ty1_ty2 {
                                *rval = Rvalue::Aggregate(
                                    Box::new(AggregateKind::Tuple),
                                    vec![
                                        self.operand_from_scalar(one, ty1, source_info.span),
                                        self.operand_from_scalar(two, ty2, source_info.span),
                                    ],
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn should_const_prop(&mut self, op: OpTy<'tcx>) -> bool {
        let mir_opt_level = self.tcx.sess.opts.debugging_opts.mir_opt_level;

        if mir_opt_level == 0 {
            return false;
        }

        match *op {
            interpret::Operand::Immediate(Immediate::Scalar(ScalarMaybeUndef::Scalar(s))) => {
                s.is_bits()
            }
            interpret::Operand::Immediate(Immediate::ScalarPair(
                ScalarMaybeUndef::Scalar(l),
                ScalarMaybeUndef::Scalar(r),
            )) => l.is_bits() && r.is_bits(),
            interpret::Operand::Indirect(_) if mir_opt_level >= 2 => {
                let mplace = op.assert_mem_place(&self.ecx);
                intern_const_alloc_recursive(&mut self.ecx, InternKind::ConstProp, mplace, false)
                    .expect("failed to intern alloc");
                true
            }
            _ => false,
        }
    }
}

/// The mode that `ConstProp` is allowed to run in for a given `Local`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ConstPropMode {
    /// The `Local` can be propagated into and reads of this `Local` can also be propagated.
    FullConstProp,
    /// The `Local` can be propagated into but reads cannot be propagated.
    OnlyPropagateInto,
    /// No propagation is allowed at all.
    NoPropagation,
}

struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // false at the beginning, once set, there are not allowed to be any more assignments
    found_assignment: IndexVec<Local, bool>,
}

impl CanConstProp {
    /// returns true if `local` can be propagated
    fn check(body: ReadOnlyBodyAndCache<'_, '_>) -> IndexVec<Local, ConstPropMode> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(ConstPropMode::FullConstProp, &body.local_decls),
            found_assignment: IndexVec::from_elem(false, &body.local_decls),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            // cannot use args at all
            // cannot use locals because if x < y { y - x } else { x - y } would
            //        lint for x != y
            // FIXME(oli-obk): lint variables until they are used in a condition
            // FIXME(oli-obk): lint if return value is constant
            let local_kind = body.local_kind(local);

            if local_kind == LocalKind::Arg || local_kind == LocalKind::Var {
                *val = ConstPropMode::OnlyPropagateInto;
                trace!("local {:?} can't be const propagated because it's not a temporary", local);
            }
        }
        cpv.visit_body(body);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
        use rustc::mir::visit::PlaceContext::*;
        match context {
            // Constants must have at most one write
            // FIXME(oli-obk): we could be more powerful here, if the multiple writes
            // only occur in independent execution paths
            MutatingUse(MutatingUseContext::Store) => {
                if self.found_assignment[local] {
                    trace!("local {:?} can't be propagated because of multiple assignments", local);
                    self.can_const_prop[local] = ConstPropMode::NoPropagation;
                } else {
                    self.found_assignment[local] = true
                }
            }
            // Reading constants is allowed an arbitrary number of times
            NonMutatingUse(NonMutatingUseContext::Copy)
            | NonMutatingUse(NonMutatingUseContext::Move)
            | NonMutatingUse(NonMutatingUseContext::Inspect)
            | NonMutatingUse(NonMutatingUseContext::Projection)
            | MutatingUse(MutatingUseContext::Projection)
            | NonUse(_) => {}
            _ => {
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

    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, location: Location) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant, self.source_info.unwrap());
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);
        let source_info = statement.source_info;
        self.source_info = Some(source_info);
        if let StatementKind::Assign(box (ref place, ref mut rval)) = statement.kind {
            let place_ty: Ty<'tcx> = place.ty(&self.local_decls, self.tcx).ty;
            if let Ok(place_layout) = self.tcx.layout_of(self.param_env.and(place_ty)) {
                if let Some(local) = place.as_local() {
                    let can_const_prop = self.can_const_prop[local];
                    if let Some(()) = self.const_prop(rval, place_layout, source_info, place) {
                        if can_const_prop == ConstPropMode::FullConstProp
                            || can_const_prop == ConstPropMode::OnlyPropagateInto
                        {
                            if let Some(value) = self.get_const(local) {
                                if self.should_const_prop(value) {
                                    trace!("replacing {:?} with {:?}", rval, value);
                                    self.replace_with_const(rval, value, statement.source_info);

                                    if can_const_prop == ConstPropMode::FullConstProp {
                                        trace!("propagated into {:?}", local);
                                    }
                                }
                            }
                        }
                    }
                    if self.can_const_prop[local] != ConstPropMode::FullConstProp {
                        trace!("can't propagate into {:?}", local);
                        if local != RETURN_PLACE {
                            self.remove_const(local);
                        }
                    }
                }
            }
        } else {
            match statement.kind {
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
                    let expected = ScalarMaybeUndef::from(Scalar::from_bool(*expected));
                    let value_const = self.ecx.read_scalar(value).unwrap();
                    if expected != value_const {
                        // poison all places this operand references so that further code
                        // doesn't use the invalid value
                        match cond {
                            Operand::Move(ref place) | Operand::Copy(ref place) => {
                                self.remove_const(place.local);
                            }
                            Operand::Constant(_) => {}
                        }
                        let span = terminator.source_info.span;
                        let hir_id = self
                            .tcx
                            .hir()
                            .as_local_hir_id(self.source.def_id())
                            .expect("some part of a failing const eval must be local");
                        self.tcx.struct_span_lint_hir(
                            ::rustc::lint::builtin::CONST_ERR,
                            hir_id,
                            span,
                            |lint| {
                                let msg = match msg {
                                    PanicInfo::Overflow(_)
                                    | PanicInfo::OverflowNeg
                                    | PanicInfo::DivisionByZero
                                    | PanicInfo::RemainderByZero => msg.description().to_owned(),
                                    PanicInfo::BoundsCheck { ref len, ref index } => {
                                        let len =
                                            self.eval_operand(len, source_info).expect("len must be const");
                                        let len = match self.ecx.read_scalar(len) {
                                            Ok(ScalarMaybeUndef::Scalar(Scalar::Raw { data, .. })) => data,
                                            other => bug!("const len not primitive: {:?}", other),
                                        };
                                        let index = self
                                            .eval_operand(index, source_info)
                                            .expect("index must be const");
                                        let index = match self.ecx.read_scalar(index) {
                                            Ok(ScalarMaybeUndef::Scalar(Scalar::Raw { data, .. })) => data,
                                            other => bug!("const index not primitive: {:?}", other),
                                        };
                                        format!(
                                            "index out of bounds: \
                                            the len is {} but the index is {}",
                                            len, index,
                                        )
                                    }
                                    // Need proper const propagator for these
                                    _ => return,
                                };
                                lint.build(&msg).emit()
                            },
                        );
                    } else {
                        if self.should_const_prop(value) {
                            if let ScalarMaybeUndef::Scalar(scalar) = value_const {
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
            TerminatorKind::SwitchInt { ref mut discr, switch_ty, .. } => {
                if let Some(value) = self.eval_operand(&discr, source_info) {
                    if self.should_const_prop(value) {
                        if let ScalarMaybeUndef::Scalar(scalar) =
                            self.ecx.read_scalar(value).unwrap()
                        {
                            *discr = self.operand_from_scalar(scalar, switch_ty, source_info.span);
                        }
                    }
                }
            }
            //none of these have Operands to const-propagate
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. } => {}
            //FIXME(wesleywiser) Call does have Operands that could be const-propagated
            TerminatorKind::Call { .. } => {}
        }
    }
}
