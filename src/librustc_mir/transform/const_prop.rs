//! Propagates constants for early reporting of statically known
//! assertion failures

use std::borrow::Cow;
use std::cell::Cell;

use rustc::hir::def::DefKind;
use rustc::hir::def_id::DefId;
use rustc::mir::{
    AggregateKind, Constant, Location, Place, PlaceBase, Body, Operand, Rvalue,
    Local, UnOp, StatementKind, Statement, LocalKind,
    TerminatorKind, Terminator,  ClearCrossCrate, SourceInfo, BinOp,
    SourceScope, SourceScopeLocalData, LocalDecl, BasicBlock,
};
use rustc::mir::visit::{
    Visitor, PlaceContext, MutatingUseContext, MutVisitor, NonMutatingUseContext,
};
use rustc::mir::interpret::{Scalar, InterpResult, PanicInfo};
use rustc::ty::{self, Instance, ParamEnv, Ty, TyCtxt};
use syntax::ast::Mutability;
use syntax_pos::{Span, DUMMY_SP};
use rustc::ty::subst::InternalSubsts;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc::ty::layout::{
    LayoutOf, TyLayout, LayoutError, HasTyCtxt, TargetDataLayout, HasDataLayout,
};

use crate::interpret::{
    self, InterpCx, ScalarMaybeUndef, Immediate, OpTy,
    StackPopCleanup, LocalValue, LocalState, AllocId, Frame,
    Allocation, MemoryKind, ImmTy, Pointer, Memory, PlaceTy,
    Operand as InterpOperand,
};
use crate::const_eval::error_to_const_error;
use crate::transform::{MirPass, MirSource};

pub struct ConstProp;

impl<'tcx> MirPass<'tcx> for ConstProp {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if source.promoted.is_some() {
            return;
        }

        use rustc::hir::map::blocks::FnLikeNode;
        let hir_id = tcx.hir().as_local_hir_id(source.def_id())
                              .expect("Non-local call to local provider is_const_fn");

        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();
        let is_assoc_const = match tcx.def_kind(source.def_id()) {
            Some(DefKind::AssocConst) => true,
            _ => false,
        };

        // Only run const prop on functions, methods, closures and associated constants
        if !is_fn_like && !is_assoc_const  {
            // skip anon_const/statics/consts because they'll be evaluated by miri anyway
            trace!("ConstProp skipped for {:?}", source.def_id());
            return
        }

        let is_generator = tcx.type_of(source.def_id()).is_generator();
        // FIXME(welseywiser) const prop doesn't work on generators because of query cycles
        // computing their layout.
        if is_generator {
            trace!("ConstProp skipped for generator {:?}", source.def_id());
            return
        }

        trace!("ConstProp starting for {:?}", source.def_id());

        // Steal some data we need from `body`.
        let source_scope_local_data = std::mem::replace(
            &mut body.source_scope_local_data,
            ClearCrossCrate::Clear
        );

        let dummy_body =
            &Body::new(
                body.basic_blocks().clone(),
                Default::default(),
                ClearCrossCrate::Clear,
                None,
                body.local_decls.clone(),
                Default::default(),
                body.arg_count,
                Default::default(),
                tcx.def_span(source.def_id()),
                Default::default(),
            );

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require an uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder = ConstPropagator::new(
            body,
            dummy_body,
            source_scope_local_data,
            tcx,
            source
        );
        optimization_finder.visit_body(body);

        // put back the data we stole from `mir`
        let source_scope_local_data = optimization_finder.release_stolen_data();
        std::mem::replace(
            &mut body.source_scope_local_data,
            source_scope_local_data
        );

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

    fn find_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _dest: Option<PlaceTy<'tcx>>,
        _ret: Option<BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir Body<'tcx>>> {
        Ok(None)
    }

    fn call_extra_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: !,
        _args: &[OpTy<'tcx>],
        _dest: Option<PlaceTy<'tcx>>,
        _ret: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        match fn_val {}
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _dest: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("calling intrinsics isn't supported in ConstProp");
    }

    fn ptr_to_int(
        _mem: &Memory<'mir, 'tcx, Self>,
        _ptr: Pointer,
    ) -> InterpResult<'tcx, u64> {
        throw_unsup_format!("ptr-to-int casts aren't supported in ConstProp");
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: ImmTy<'tcx>,
        _right: ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        // We can't do this because aliasing of memory can differ between const eval and llvm
        throw_unsup_format!("pointer arithmetic or comparisons aren't supported in ConstProp");
    }

    fn find_foreign_static(
        _tcx: TyCtxt<'tcx>,
        _def_id: DefId,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<Self::PointerTag>>> {
        throw_unsup!(ReadForeignStatic)
    }

    #[inline(always)]
    fn tag_allocation<'b>(
        _memory_extra: &(),
        _id: AllocId,
        alloc: Cow<'b, Allocation>,
        _kind: Option<MemoryKind<!>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag>>, Self::PointerTag) {
        // We do not use a tag so we can just cheaply forward the allocation
        (alloc, ())
    }

    #[inline(always)]
    fn tag_static_base_pointer(
        _memory_extra: &(),
        _id: AllocId,
    ) -> Self::PointerTag {
        ()
    }

    fn box_alloc(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _dest: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        throw_unsup_format!("can't const prop `box` keyword");
    }

    fn access_local(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        frame: &Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>,
        local: Local,
    ) -> InterpResult<'tcx, InterpOperand<Self::PointerTag>> {
        let l = &frame.locals[local];

        if l.value == LocalValue::Uninitialized {
            throw_unsup_format!("tried to access an uninitialized local");
        }

        l.access()
    }

    fn before_access_static(
        allocation: &Allocation<Self::PointerTag, Self::AllocExtra>,
    ) -> InterpResult<'tcx> {
        // if the static allocation is mutable or if it has relocations (it may be legal to mutate
        // the memory behind that in the future), then we can't const prop it
        if allocation.mutability == Mutability::Mutable || allocation.relocations().len() > 0 {
            throw_unsup_format!("can't eval mutable statics in ConstProp");
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

    /// Called immediately before a stack frame gets popped.
    #[inline(always)]
    fn stack_pop(_ecx: &mut InterpCx<'mir, 'tcx, Self>, _extra: ()) -> InterpResult<'tcx> {
        Ok(())
    }
}

type Const<'tcx> = OpTy<'tcx>;

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine>,
    tcx: TyCtxt<'tcx>,
    source: MirSource<'tcx>,
    can_const_prop: IndexVec<Local, bool>,
    param_env: ParamEnv<'tcx>,
    source_scope_local_data: ClearCrossCrate<IndexVec<SourceScope, SourceScopeLocalData>>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
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
        body: &Body<'tcx>,
        dummy_body: &'mir Body<'tcx>,
        source_scope_local_data: ClearCrossCrate<IndexVec<SourceScope, SourceScopeLocalData>>,
        tcx: TyCtxt<'tcx>,
        source: MirSource<'tcx>,
    ) -> ConstPropagator<'mir, 'tcx> {
        let def_id = source.def_id();
        let param_env = tcx.param_env(def_id);
        let span = tcx.def_span(def_id);
        let mut ecx = InterpCx::new(tcx.at(span), param_env, ConstPropMachine, ());
        let can_const_prop = CanConstProp::check(body);

        ecx.push_stack_frame(
            Instance::new(def_id, &InternalSubsts::identity_for_item(tcx, def_id)),
            span,
            dummy_body,
            None,
            StackPopCleanup::None {
                cleanup: false,
            },
        ).expect("failed to push initial stack frame");

        ConstPropagator {
            ecx,
            tcx,
            source,
            param_env,
            can_const_prop,
            source_scope_local_data,
            //FIXME(wesleywiser) we can't steal this because `Visitor::super_visit_body()` needs it
            local_decls: body.local_decls.clone(),
        }
    }

    fn release_stolen_data(self) -> ClearCrossCrate<IndexVec<SourceScope, SourceScopeLocalData>> {
        self.source_scope_local_data
    }

    fn get_const(&self, local: Local) -> Option<Const<'tcx>> {
        self.ecx.access_local(self.ecx.frame(), local, None).ok()
    }

    fn remove_const(&mut self, local: Local) {
        self.ecx.frame_mut().locals[local] = LocalState {
            value: LocalValue::Uninitialized,
            layout: Cell::new(None),
        };
    }

    fn use_ecx<F, T>(
        &mut self,
        source_info: SourceInfo,
        f: F
    ) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        self.ecx.tcx.span = source_info.span;
        let lint_root = match self.source_scope_local_data {
            ClearCrossCrate::Set(ref ivs) => {
                //FIXME(#51314): remove this check
                if source_info.scope.index() >= ivs.len() {
                    return None;
                }
                ivs[source_info.scope].lint_root
            },
            ClearCrossCrate::Clear => return None,
        };
        let r = match f(self) {
            Ok(val) => Some(val),
            Err(error) => {
                use rustc::mir::interpret::InterpError::*;
                match error.kind {
                    Exit(_) => bug!("the CTFE program cannot exit"),
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
            },
        };
        self.ecx.tcx.span = DUMMY_SP;
        r
    }

    fn eval_constant(
        &mut self,
        c: &Constant<'tcx>,
    ) -> Option<Const<'tcx>> {
        self.ecx.tcx.span = c.span;
        match self.ecx.eval_const_to_op(c.literal, None) {
            Ok(op) => {
                Some(op)
            },
            Err(error) => {
                let err = error_to_const_error(&self.ecx, error);
                err.report_as_error(self.ecx.tcx, "erroneous constant used");
                None
            },
        }
    }

    fn eval_place(&mut self, place: &Place<'tcx>, source_info: SourceInfo) -> Option<Const<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(source_info, |this| {
            this.ecx.eval_place_to_op(place, None)
        })
    }

    fn eval_operand(&mut self, op: &Operand<'tcx>, source_info: SourceInfo) -> Option<Const<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c),
            | Operand::Move(ref place)
            | Operand::Copy(ref place) => self.eval_place(place, source_info),
        }
    }

    fn const_prop(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place_layout: TyLayout<'tcx>,
        source_info: SourceInfo,
        place: &Place<'tcx>,
    ) -> Option<Const<'tcx>> {
        let span = source_info.span;

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
            }

            // Additional checking: check for overflows on integer binary operations and report
            // them to the user as lints.
            Rvalue::BinaryOp(op, left, right) => {
                trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op, left, right);

                let r = self.use_ecx(source_info, |this| {
                    this.ecx.read_immediate(this.ecx.eval_operand(right, None)?)
                })?;
                if *op == BinOp::Shr || *op == BinOp::Shl {
                    let left_bits = place_layout.size.bits();
                    let right_size = r.layout.size;
                    let r_bits = r.to_scalar().and_then(|r| r.to_bits(right_size));
                    if r_bits.ok().map_or(false, |b| b >= left_bits as u128) {
                        let source_scope_local_data = match self.source_scope_local_data {
                            ClearCrossCrate::Set(ref data) => data,
                            ClearCrossCrate::Clear => return None,
                        };
                        let dir = if *op == BinOp::Shr {
                            "right"
                        } else {
                            "left"
                        };
                        let hir_id = source_scope_local_data[source_info.scope].lint_root;
                        self.tcx.lint_hir(
                            ::rustc::lint::builtin::EXCEEDING_BITSHIFTS,
                            hir_id,
                            span,
                            &format!("attempt to shift {} with overflow", dir));
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
                        let (_, overflow, _ty) = this.ecx.overflowing_binary_op(*op, l, r)?;

                        if overflow {
                            let err = err_panic!(Overflow(*op)).into();
                            return Err(err);
                        }

                        Ok(())
                    })?;
                }
            }

            // Work around: avoid ICE in miri. FIXME(wesleywiser)
            // The Miri engine ICEs when taking a reference to an uninitialized unsized
            // local. There's nothing it can do here: taking a reference needs an allocation
            // which needs to know the size. Normally that's okay as during execution
            // (e.g. for CTFE) it can never happen. But here in const_prop
            // unknown data is uninitialized, so if e.g. a function argument is unsized
            // and has a reference taken, we get an ICE.
            Rvalue::Ref(_, _, Place { base: PlaceBase::Local(local), projection: box [] }) => {
                trace!("checking Ref({:?})", place);
                let alive =
                    if let LocalValue::Live(_) = self.ecx.frame().locals[*local].value {
                        true
                    } else {
                        false
                    };

                if !alive {
                    trace!("skipping Ref({:?}) to uninitialized local", place);
                    return None;
                }
            }

            // Work around: avoid extra unnecessary locals. FIXME(wesleywiser)
            // Const eval will turn this into a `const Scalar(<ZST>)` that
            // `SimplifyLocals` doesn't know it can remove.
            Rvalue::Aggregate(_, operands) if operands.len() == 0 => {
                return None;
            }

            _ => { }
        }

        self.use_ecx(source_info, |this| {
            trace!("calling eval_rvalue_into_place(rvalue = {:?}, place = {:?})", rvalue, place);
            this.ecx.eval_rvalue_into_place(rvalue, place)?;
            this.ecx.eval_place_to_op(place, Some(place_layout))
        })
    }

    fn operand_from_scalar(&self, scalar: Scalar, ty: Ty<'tcx>, span: Span) -> Operand<'tcx> {
        Operand::Constant(Box::new(
            Constant {
                span,
                user_ty: None,
                literal: self.tcx.mk_const(*ty::Const::from_scalar(
                    self.tcx,
                    scalar,
                    ty,
                ))
            }
        ))
    }

    fn replace_with_const(
        &mut self,
        rval: &mut Rvalue<'tcx>,
        value: Const<'tcx>,
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
        let imm = self.use_ecx(source_info, |this| {
            this.ecx.try_read_immediate(value)
        });

        if let Some(Ok(imm)) = imm {
            match *imm {
                interpret::Immediate::Scalar(ScalarMaybeUndef::Scalar(scalar)) => {
                    *rval = Rvalue::Use(
                        self.operand_from_scalar(scalar, value.layout.ty, source_info.span));
                },
                Immediate::ScalarPair(
                    ScalarMaybeUndef::Scalar(one),
                    ScalarMaybeUndef::Scalar(two)
                ) => {
                    let ty = &value.layout.ty.kind;
                    if let ty::Tuple(substs) = ty {
                        *rval = Rvalue::Aggregate(
                            Box::new(AggregateKind::Tuple),
                            vec![
                                self.operand_from_scalar(
                                    one, substs[0].expect_ty(), source_info.span
                                ),
                                self.operand_from_scalar(
                                    two, substs[1].expect_ty(), source_info.span
                                ),
                            ],
                        );
                    }
                },
                _ => { }
            }
        }
    }

    fn should_const_prop(&self) -> bool {
        self.tcx.sess.opts.debugging_opts.mir_opt_level >= 2
    }
}

struct CanConstProp {
    can_const_prop: IndexVec<Local, bool>,
    // false at the beginning, once set, there are not allowed to be any more assignments
    found_assignment: IndexVec<Local, bool>,
}

impl CanConstProp {
    /// returns true if `local` can be propagated
    fn check(body: &Body<'_>) -> IndexVec<Local, bool> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(true, &body.local_decls),
            found_assignment: IndexVec::from_elem(false, &body.local_decls),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            // cannot use args at all
            // cannot use locals because if x < y { y - x } else { x - y } would
            //        lint for x != y
            // FIXME(oli-obk): lint variables until they are used in a condition
            // FIXME(oli-obk): lint if return value is constant
            *val = body.local_kind(local) == LocalKind::Temp;

            if !*val {
                trace!("local {:?} can't be propagated because it's not a temporary", local);
            }
        }
        cpv.visit_body(body);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_local(
        &mut self,
        &local: &Local,
        context: PlaceContext,
        _: Location,
    ) {
        use rustc::mir::visit::PlaceContext::*;
        match context {
            // Constants must have at most one write
            // FIXME(oli-obk): we could be more powerful here, if the multiple writes
            // only occur in independent execution paths
            MutatingUse(MutatingUseContext::Store) => if self.found_assignment[local] {
                trace!("local {:?} can't be propagated because of multiple assignments", local);
                self.can_const_prop[local] = false;
            } else {
                self.found_assignment[local] = true
            },
            // Reading constants is allowed an arbitrary number of times
            NonMutatingUse(NonMutatingUseContext::Copy) |
            NonMutatingUse(NonMutatingUseContext::Move) |
            NonMutatingUse(NonMutatingUseContext::Inspect) |
            NonMutatingUse(NonMutatingUseContext::Projection) |
            MutatingUse(MutatingUseContext::Projection) |
            NonUse(_) => {},
            _ => {
                trace!("local {:?} can't be propagaged because it's used: {:?}", local, context);
                self.can_const_prop[local] = false;
            },
        }
    }
}

impl<'mir, 'tcx> MutVisitor<'tcx> for ConstPropagator<'mir, 'tcx> {
    fn visit_constant(
        &mut self,
        constant: &mut Constant<'tcx>,
        location: Location,
    ) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant);
    }

    fn visit_statement(
        &mut self,
        statement: &mut Statement<'tcx>,
        location: Location,
    ) {
        trace!("visit_statement: {:?}", statement);
        if let StatementKind::Assign(box(ref place, ref mut rval)) = statement.kind {
            let place_ty: Ty<'tcx> = place
                .ty(&self.local_decls, self.tcx)
                .ty;
            if let Ok(place_layout) = self.tcx.layout_of(self.param_env.and(place_ty)) {
                if let Place {
                    base: PlaceBase::Local(local),
                    projection: box [],
                } = *place {
                    if let Some(value) = self.const_prop(rval,
                                                         place_layout,
                                                         statement.source_info,
                                                         place) {
                        trace!("checking whether {:?} can be stored to {:?}", value, local);
                        if self.can_const_prop[local] {
                            trace!("stored {:?} to {:?}", value, local);
                            assert_eq!(self.get_const(local), Some(value));

                            if self.should_const_prop() {
                                self.replace_with_const(
                                    rval,
                                    value,
                                    statement.source_info,
                                );
                            }
                        } else {
                            trace!("can't propagate {:?} to {:?}", value, local);
                            self.remove_const(local);
                        }
                    }
                }
            }
        } else {
            match statement.kind {
                StatementKind::StorageLive(local) |
                StatementKind::StorageDead(local) if self.can_const_prop[local] => {
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

    fn visit_terminator(
        &mut self,
        terminator: &mut Terminator<'tcx>,
        location: Location,
    ) {
        self.super_terminator(terminator, location);
        let source_info = terminator.source_info;
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
                                if let PlaceBase::Local(local) = place.base {
                                    self.remove_const(local);
                                }
                            },
                            Operand::Constant(_) => {}
                        }
                        let span = terminator.source_info.span;
                        let hir_id = self
                            .tcx
                            .hir()
                            .as_local_hir_id(self.source.def_id())
                            .expect("some part of a failing const eval must be local");
                        let msg = match msg {
                            PanicInfo::Overflow(_) |
                            PanicInfo::OverflowNeg |
                            PanicInfo::DivisionByZero |
                            PanicInfo::RemainderByZero =>
                                msg.description().to_owned(),
                            PanicInfo::BoundsCheck { ref len, ref index } => {
                                let len = self
                                    .eval_operand(len, source_info)
                                    .expect("len must be const");
                                let len = match self.ecx.read_scalar(len) {
                                    Ok(ScalarMaybeUndef::Scalar(Scalar::Raw {
                                        data, ..
                                    })) => data,
                                    other => bug!("const len not primitive: {:?}", other),
                                };
                                let index = self
                                    .eval_operand(index, source_info)
                                    .expect("index must be const");
                                let index = match self.ecx.read_scalar(index) {
                                    Ok(ScalarMaybeUndef::Scalar(Scalar::Raw {
                                        data, ..
                                    })) => data,
                                    other => bug!("const index not primitive: {:?}", other),
                                };
                                format!(
                                    "index out of bounds: \
                                    the len is {} but the index is {}",
                                    len,
                                    index,
                                )
                            },
                            // Need proper const propagator for these
                            _ => return,
                        };
                        self.tcx.lint_hir(
                            ::rustc::lint::builtin::CONST_ERR,
                            hir_id,
                            span,
                            &msg,
                        );
                    } else {
                        if self.should_const_prop() {
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
            },
            TerminatorKind::SwitchInt { ref mut discr, switch_ty, .. } => {
                if self.should_const_prop() {
                    if let Some(value) = self.eval_operand(&discr, source_info) {
                        if let ScalarMaybeUndef::Scalar(scalar) =
                                self.ecx.read_scalar(value).unwrap() {
                            *discr = self.operand_from_scalar(scalar, switch_ty, source_info.span);
                        }
                    }
                }
            },
            //none of these have Operands to const-propagate
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::Abort |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::Drop { .. } |
            TerminatorKind::DropAndReplace { .. } |
            TerminatorKind::Yield { .. } |
            TerminatorKind::GeneratorDrop |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => { }
            //FIXME(wesleywiser) Call does have Operands that could be const-propagated
            TerminatorKind::Call { .. } => { }
        }
    }
}
