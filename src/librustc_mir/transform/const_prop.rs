//! Propagates constants for early reporting of statically known
//! assertion failures

use std::cell::Cell;

use rustc::hir::def::DefKind;
use rustc::mir::{
    AggregateKind, Constant, Location, Place, PlaceBase, Body, Operand, Rvalue,
    Local, NullOp, UnOp, StatementKind, Statement, LocalKind,
    TerminatorKind, Terminator,  ClearCrossCrate, SourceInfo, BinOp,
    SourceScope, SourceScopeLocalData, LocalDecl,
};
use rustc::mir::visit::{
    Visitor, PlaceContext, MutatingUseContext, MutVisitor, NonMutatingUseContext,
};
use rustc::mir::interpret::{Scalar, InterpResult, PanicInfo};
use rustc::ty::{self, Instance, ParamEnv, Ty, TyCtxt};
use syntax_pos::{Span, DUMMY_SP};
use rustc::ty::subst::InternalSubsts;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::ty::layout::{
    LayoutOf, TyLayout, LayoutError, HasTyCtxt, TargetDataLayout, HasDataLayout,
};

use crate::interpret::{
    self, InterpCx, ScalarMaybeUndef, Immediate, OpTy,
    ImmTy, StackPopCleanup, LocalValue, LocalState,
};
use crate::const_eval::{
    CompileTimeInterpreter, error_to_const_error, mk_eval_cx,
};
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

type Const<'tcx> = OpTy<'tcx>;

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>,
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
        let mut ecx = mk_eval_cx(tcx, span, param_env);
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
        let l = &self.ecx.frame().locals[local];

        // If the local is `Unitialized` or `Dead` then we haven't propagated a value into it.
        //
        // `InterpCx::access_local()` mostly takes care of this for us however, for ZSTs,
        // it will synthesize a value for us. In doing so, that will cause the
        // `get_const(l).is_empty()` assert right before we call `set_const()` in `visit_statement`
        // to fail.
        if let LocalValue::Uninitialized | LocalValue::Dead = l.value {
            return None;
        }

        self.ecx.access_local(self.ecx.frame(), local, None).ok()
    }

    fn set_const(&mut self, local: Local, c: Const<'tcx>) {
        let frame = self.ecx.frame_mut();

        if let Some(layout) = frame.locals[local].layout.get() {
            debug_assert_eq!(c.layout, layout);
        }

        frame.locals[local] = LocalState {
            value: LocalValue::Live(*c),
            layout: Cell::new(Some(c.layout)),
        };
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
        match *rvalue {
            Rvalue::Repeat(..) |
            Rvalue::Aggregate(..) |
            Rvalue::NullaryOp(NullOp::Box, _) |
            Rvalue::Discriminant(..) => None,

            Rvalue::Use(_) |
            Rvalue::Len(_) |
            Rvalue::Cast(..) |
            Rvalue::NullaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::Ref(..) => {
                self.use_ecx(source_info, |this| {
                    this.ecx.eval_rvalue_into_place(rvalue, place)?;
                    this.ecx.eval_place_to_op(place, Some(place_layout))
                })
            },

            Rvalue::UnaryOp(op, ref arg) => {
                let overflow_check = self.tcx.sess.overflow_checks();

                self.use_ecx(source_info, |this| {
                    // We check overflow in debug mode already
                    // so should only check in release mode.
                    if op == UnOp::Neg && !overflow_check {
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
                    }

                    this.ecx.eval_rvalue_into_place(rvalue, place)?;
                    this.ecx.eval_place_to_op(place, Some(place_layout))
                })
            }
            Rvalue::BinaryOp(op, ref left, ref right) => {
                trace!("rvalue binop {:?} for {:?} and {:?}", op, left, right);

                let r = self.use_ecx(source_info, |this| {
                    this.ecx.read_immediate(this.ecx.eval_operand(right, None)?)
                })?;
                if op == BinOp::Shr || op == BinOp::Shl {
                    let left_bits = place_layout.size.bits();
                    let right_size = r.layout.size;
                    let r_bits = r.to_scalar().and_then(|r| r.to_bits(right_size));
                    if r_bits.ok().map_or(false, |b| b >= left_bits as u128) {
                        let source_scope_local_data = match self.source_scope_local_data {
                            ClearCrossCrate::Set(ref data) => data,
                            ClearCrossCrate::Clear => return None,
                        };
                        let dir = if op == BinOp::Shr {
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
                trace!("const evaluating {:?} for {:?} and {:?}", op, left, right);
                let val = self.use_ecx(source_info, |this| {
                    let l = this.ecx.read_immediate(this.ecx.eval_operand(left, None)?)?;
                    let (val, overflow, _ty) = this.ecx.overflowing_binary_op(op, l, r)?;

                    // We check overflow in debug mode already
                    // so should only check in release mode.
                    if !this.tcx.sess.overflow_checks() && overflow {
                        let err = err_panic!(Overflow(op)).into();
                        return Err(err);
                    }

                    let val = ImmTy {
                        imm: Immediate::Scalar(val.into()),
                        layout: place_layout,
                    };

                    let dest = this.ecx.eval_place(place)?;
                    this.ecx.write_immediate(*val, dest)?;

                    Ok(val)
                })?;
                Some(val.into())
            },
        }
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
                    let ty = &value.layout.ty.sty;
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
                    if let Some(value) = self.const_prop(rval, place_layout, statement.source_info, place) {
                        trace!("checking whether {:?} can be stored to {:?}", value, local);
                        if self.can_const_prop[local] {
                            trace!("storing {:?} to {:?}", value, local);
                            assert!(self.get_const(local).is_none() || self.get_const(local) == Some(value));
                            self.set_const(local, value);

                            if self.should_const_prop() {
                                self.replace_with_const(
                                    rval,
                                    value,
                                    statement.source_info,
                                );
                            }
                        }
                    }
                }
            }
        } else if let StatementKind::StorageLive(local) = statement.kind {
            if self.can_const_prop[local] {
                let frame = self.ecx.frame_mut();

                frame.locals[local].value = LocalValue::Uninitialized;
            }
        } else if let StatementKind::StorageDead(local) = statement.kind {
            if self.can_const_prop[local] {
                let frame = self.ecx.frame_mut();

                frame.locals[local].value = LocalValue::Dead;
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
