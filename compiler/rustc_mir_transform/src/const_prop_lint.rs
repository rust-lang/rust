//! Propagates constants for early reporting of statically known
//! assertion failures

use crate::const_prop::CanConstProp;
use crate::const_prop::ConstPropMachine;
use crate::const_prop::ConstPropMode;
use crate::MirLint;
use rustc_const_eval::const_eval::ConstEvalErr;
use rustc_const_eval::interpret::{
    self, InterpCx, InterpResult, LocalState, LocalValue, MemoryKind, OpTy, Scalar, StackPopCleanup,
};
use rustc_hir::def::DefKind;
use rustc_hir::HirId;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    AssertKind, BinOp, Body, Constant, ConstantKind, Local, LocalDecl, Location, Operand, Place,
    Rvalue, SourceInfo, SourceScope, SourceScopeData, Statement, StatementKind, Terminator,
    TerminatorKind, UnOp, RETURN_PLACE,
};
use rustc_middle::ty::layout::{LayoutError, LayoutOf, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{
    self, ConstInt, ConstKind, Instance, ParamEnv, ScalarInt, Ty, TyCtxt, TypeVisitable,
};
use rustc_session::lint;
use rustc_span::Span;
use rustc_target::abi::{HasDataLayout, Size, TargetDataLayout};
use rustc_trait_selection::traits;
use std::cell::Cell;

/// The maximum number of bytes that we'll allocate space for a local or the return value.
/// Needed for #66397, because otherwise we eval into large places and that can cause OOM or just
/// Severely regress performance.
const MAX_ALLOC_LIMIT: u64 = 1024;
pub struct ConstProp;

impl<'tcx> MirLint<'tcx> for ConstProp {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if body.source.promoted.is_some() {
            return;
        }

        let def_id = body.source.def_id().expect_local();
        let is_fn_like = tcx.def_kind(def_id).is_fn_like();
        let is_assoc_const = tcx.def_kind(def_id) == DefKind::AssocConst;

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

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'mir, 'tcx> {
    ecx: InterpCx<'mir, 'tcx, ConstPropMachine<'mir, 'tcx>>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    source_scopes: &'mir IndexVec<SourceScope, SourceScopeData<'tcx>>,
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
                !ret_layout.is_unsized() && ret_layout.size < Size::from_bytes(MAX_ALLOC_LIMIT)
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
            source_scopes: &dummy_body.source_scopes,
            local_decls: &dummy_body.local_decls,
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
        Some(match self.ecx.read_immediate_raw(&op) {
            Ok(Ok(imm)) => imm.into(),
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

    fn lint_root(&self, source_info: SourceInfo) -> Option<HirId> {
        source_info.scope.lint_root(self.source_scopes)
    }

    fn use_ecx<F, T>(&mut self, source_info: SourceInfo, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> InterpResult<'tcx, T>,
    {
        // Overwrite the PC -- whatever the interpreter does to it does not make any sense anyway.
        self.ecx.frame_mut().loc = Err(source_info.span);
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
    fn eval_constant(&mut self, c: &Constant<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        // FIXME we need to revisit this for #67176
        if c.needs_subst() {
            return None;
        }

        match self.ecx.mir_const_to_op(&c.literal, None) {
            Ok(op) => Some(op),
            Err(error) => {
                let tcx = self.ecx.tcx.at(c.span);
                let err = ConstEvalErr::new(&self.ecx, error, Some(c.span));
                if let Some(lint_root) = self.lint_root(source_info) {
                    let lint_only = match c.literal {
                        ConstantKind::Ty(ct) => match ct.kind() {
                            // Promoteds must lint and not error as the user didn't ask for them
                            ConstKind::Unevaluated(ty::Unevaluated {
                                def: _,
                                substs: _,
                                promoted: Some(_),
                            }) => true,
                            // Out of backwards compatibility we cannot report hard errors in unused
                            // generic functions using associated constants of the generic parameters.
                            _ => c.literal.needs_subst(),
                        },
                        ConstantKind::Val(_, ty) => ty.needs_subst(),
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
    fn eval_place(&mut self, place: Place<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        trace!("eval_place(place={:?})", place);
        self.use_ecx(source_info, |this| this.ecx.eval_place_to_op(place, None))
    }

    /// Returns the value, if any, of evaluating `op`. Calls upon `eval_constant`
    /// or `eval_place`, depending on the variant of `Operand` used.
    fn eval_operand(&mut self, op: &Operand<'tcx>, source_info: SourceInfo) -> Option<OpTy<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c, source_info),
            Operand::Move(place) | Operand::Copy(place) => self.eval_place(place, source_info),
        }
    }

    fn report_assert_as_lint(
        &self,
        lint: &'static lint::Lint,
        source_info: SourceInfo,
        message: &'static str,
        panic: AssertKind<impl std::fmt::Debug>,
    ) {
        if let Some(lint_root) = self.lint_root(source_info) {
            self.tcx.struct_span_lint_hir(lint, lint_root, source_info.span, |lint| {
                let mut err = lint.build(message);
                err.span_label(source_info.span, format!("{:?}", panic));
                err.emit();
            });
        }
    }

    fn check_unary_op(
        &mut self,
        op: UnOp,
        arg: &Operand<'tcx>,
        source_info: SourceInfo,
    ) -> Option<()> {
        if let (val, true) = self.use_ecx(source_info, |this| {
            let val = this.ecx.read_immediate(&this.ecx.eval_operand(arg, None)?)?;
            let (_res, overflow, _ty) = this.ecx.overflowing_unary_op(op, &val)?;
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
            );
            return None;
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
        let r = self.use_ecx(source_info, |this| {
            this.ecx.read_immediate(&this.ecx.eval_operand(right, None)?)
        });
        let l = self.use_ecx(source_info, |this| {
            this.ecx.read_immediate(&this.ecx.eval_operand(left, None)?)
        });
        // Check for exceeding shifts *even if* we cannot evaluate the LHS.
        if op == BinOp::Shr || op == BinOp::Shl {
            let r = r.clone()?;
            // We need the type of the LHS. We cannot use `place_layout` as that is the type
            // of the result, which for checked binops is not the same!
            let left_ty = left.ty(self.local_decls, self.tcx);
            let left_size = self.ecx.layout_of(left_ty).ok()?.size;
            let right_size = r.layout.size;
            let r_bits = r.to_scalar().to_bits(right_size).ok();
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
                );
                return None;
            }
        }

        if let (Some(l), Some(r)) = (l, r) {
            // The remaining operators are handled through `overflowing_binary_op`.
            if self.use_ecx(source_info, |this| {
                let (_res, overflow, _ty) = this.ecx.overflowing_binary_op(op, &l, &r)?;
                Ok(overflow)
            })? {
                self.report_assert_as_lint(
                    lint::builtin::ARITHMETIC_OVERFLOW,
                    source_info,
                    "this arithmetic operation will overflow",
                    AssertKind::Overflow(op, l.to_const_int(), r.to_const_int()),
                );
                return None;
            }
        }
        Some(())
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
            Rvalue::BinaryOp(op, box (left, right)) => {
                trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op, left, right);
                self.check_binary_op(*op, left, right, source_info)?;
            }
            Rvalue::CheckedBinaryOp(op, box (left, right)) => {
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
            | Rvalue::CopyForDeref(..)
            | Rvalue::Repeat(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..) => {}
        }

        // FIXME we need to revisit this for #67176
        if rvalue.needs_subst() {
            return None;
        }

        self.use_ecx(source_info, |this| this.ecx.eval_rvalue_into_place(rvalue, place))
    }
}

impl<'tcx> Visitor<'tcx> for ConstPropagator<'_, 'tcx> {
    fn visit_body(&mut self, body: &Body<'tcx>) {
        for (bb, data) in body.basic_blocks().iter_enumerated() {
            self.visit_basic_block_data(bb, data);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant, self.source_info.unwrap());
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        trace!("visit_statement: {:?}", statement);
        let source_info = statement.source_info;
        self.source_info = Some(source_info);
        if let StatementKind::Assign(box (place, ref rval)) = statement.kind {
            let can_const_prop = self.ecx.machine.can_const_prop[place.local];
            if let Some(()) = self.const_prop(rval, source_info, place) {
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
                            if self
                                .use_ecx(source_info, |this| this.ecx.statement(statement))
                                .is_some()
                            {
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

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let source_info = terminator.source_info;
        self.source_info = Some(source_info);
        self.super_terminator(terminator, location);
        match &terminator.kind {
            TerminatorKind::Assert { expected, ref msg, ref cond, .. } => {
                if let Some(ref value) = self.eval_operand(&cond, source_info) {
                    trace!("assertion on {:?} should be {:?}", value, expected);
                    let expected = Scalar::from_bool(*expected);
                    let value_const = self.ecx.read_scalar(&value).unwrap();
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
                            self.eval_operand(op, source_info).map_or(DbgVal::Underscore, |op| {
                                DbgVal::Val(self.ecx.read_immediate(&op).unwrap().to_const_int())
                            })
                        };
                        let msg = match msg {
                            AssertKind::DivisionByZero(op) => {
                                Some(AssertKind::DivisionByZero(eval_to_int(op)))
                            }
                            AssertKind::RemainderByZero(op) => {
                                Some(AssertKind::RemainderByZero(eval_to_int(op)))
                            }
                            AssertKind::Overflow(bin_op @ (BinOp::Div | BinOp::Rem), op1, op2) => {
                                // Division overflow is *UB* in the MIR, and different than the
                                // other overflow checks.
                                Some(AssertKind::Overflow(
                                    *bin_op,
                                    eval_to_int(op1),
                                    eval_to_int(op2),
                                ))
                            }
                            AssertKind::BoundsCheck { ref len, ref index } => {
                                let len = eval_to_int(len);
                                let index = eval_to_int(index);
                                Some(AssertKind::BoundsCheck { len, index })
                            }
                            // Remaining overflow errors are already covered by checks on the binary operators.
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
                    }
                }
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
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::InlineAsm { .. } => {}
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
