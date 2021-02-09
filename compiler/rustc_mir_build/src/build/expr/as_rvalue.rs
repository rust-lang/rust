//! See docs in `build/expr/mod.rs`.

use rustc_index::vec::Idx;

use crate::build::expr::category::{Category, RvalueFunc};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::build::expr::as_place::PlaceBase;
use crate::thir::*;
use rustc_middle::middle::region;
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, UpvarSubsts};
use rustc_span::Span;

use std::slice;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Returns an rvalue suitable for use until the end of the current
    /// scope expression.
    ///
    /// The operand returned from this function will *not be valid* after
    /// an ExprKind::Scope is passed, so please do *not* return it from
    /// functions to avoid bad miscompiles.
    crate fn as_local_rvalue<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Rvalue<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let local_scope = self.local_scope();
        self.as_rvalue(block, Some(local_scope), expr)
    }

    /// Compile `expr`, yielding an rvalue.
    fn as_rvalue<M>(
        &mut self,
        block: BasicBlock,
        scope: Option<region::Scope>,
        expr: M,
    ) -> BlockAnd<Rvalue<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_rvalue(block, scope, expr)
    }

    fn expr_as_rvalue(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: Expr<'tcx>,
    ) -> BlockAnd<Rvalue<'tcx>> {
        debug!("expr_as_rvalue(block={:?}, scope={:?}, expr={:?})", block, scope, expr);

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::ThreadLocalRef(did) => block.and(Rvalue::ThreadLocalRef(did)),
            ExprKind::Scope { region_scope, lint_level, value } => {
                let region_scope = (region_scope, source_info);
                this.in_scope(region_scope, lint_level, |this| this.as_rvalue(block, scope, value))
            }
            ExprKind::Repeat { value, count } => {
                let value_operand = unpack!(block = this.as_operand(block, scope, value));
                block.and(Rvalue::Repeat(value_operand, count))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs = unpack!(block = this.as_operand(block, scope, lhs));
                let rhs = unpack!(block = this.as_operand(block, scope, rhs));
                this.build_binary_op(block, op, expr_span, expr.ty, lhs, rhs)
            }
            ExprKind::Unary { op, arg } => {
                let arg = unpack!(block = this.as_operand(block, scope, arg));
                // Check for -MIN on signed integers
                if this.hir.check_overflow() && op == UnOp::Neg && expr.ty.is_signed() {
                    let bool_ty = this.hir.bool_ty();

                    let minval = this.minval_literal(expr_span, expr.ty);
                    let is_min = this.temp(bool_ty, expr_span);

                    this.cfg.push_assign(
                        block,
                        source_info,
                        is_min,
                        Rvalue::BinaryOp(BinOp::Eq, arg.to_copy(), minval),
                    );

                    block = this.assert(
                        block,
                        Operand::Move(is_min),
                        false,
                        AssertKind::OverflowNeg(arg.to_copy()),
                        expr_span,
                    );
                }
                block.and(Rvalue::UnaryOp(op, arg))
            }
            ExprKind::Box { value } => {
                let value = this.hir.mirror(value);
                // The `Box<T>` temporary created here is not a part of the HIR,
                // and therefore is not considered during generator auto-trait
                // determination. See the comment about `box` at `yield_in_scope`.
                let result = this.local_decls.push(LocalDecl::new(expr.ty, expr_span).internal());
                this.cfg.push(
                    block,
                    Statement { source_info, kind: StatementKind::StorageLive(result) },
                );
                if let Some(scope) = scope {
                    // schedule a shallow free of that memory, lest we unwind:
                    this.schedule_drop_storage_and_value(expr_span, scope, result);
                }

                // malloc some memory of suitable type (thus far, uninitialized):
                let box_ = Rvalue::NullaryOp(NullOp::Box, value.ty);
                this.cfg.push_assign(block, source_info, Place::from(result), box_);

                // Initialize the box contents. No scope is needed since the
                // `Box` is already scheduled to be dropped.
                unpack!(
                    block = this.into(
                        this.hir.tcx().mk_place_deref(Place::from(result)),
                        None,
                        block,
                        value,
                    )
                );
                let result_operand = Operand::Move(Place::from(result));
                this.record_operands_moved(slice::from_ref(&result_operand));
                block.and(Rvalue::Use(result_operand))
            }
            ExprKind::Cast { source } => {
                let source = unpack!(block = this.as_operand(block, scope, source));
                block.and(Rvalue::Cast(CastKind::Misc, source, expr.ty))
            }
            ExprKind::Pointer { cast, source } => {
                let source = unpack!(block = this.as_operand(block, scope, source));
                block.and(Rvalue::Cast(CastKind::Pointer(cast), source, expr.ty))
            }
            ExprKind::Array { fields } => {
                // (*) We would (maybe) be closer to codegen if we
                // handled this and other aggregate cases via
                // `into()`, not `as_rvalue` -- in that case, instead
                // of generating
                //
                //     let tmp1 = ...1;
                //     let tmp2 = ...2;
                //     dest = Rvalue::Aggregate(Foo, [tmp1, tmp2])
                //
                // we could just generate
                //
                //     dest.f = ...1;
                //     dest.g = ...2;
                //
                // The problem is that then we would need to:
                //
                // (a) have a more complex mechanism for handling
                //     partial cleanup;
                // (b) distinguish the case where the type `Foo` has a
                //     destructor, in which case creating an instance
                //     as a whole "arms" the destructor, and you can't
                //     write individual fields; and,
                // (c) handle the case where the type Foo has no
                //     fields. We don't want `let x: ();` to compile
                //     to the same MIR as `let x = ();`.

                // first process the set of fields
                let el_ty = expr.ty.sequence_element_type(this.hir.tcx());
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|f| unpack!(block = this.as_operand(block, scope, f)))
                    .collect();

                this.record_operands_moved(&fields);
                block.and(Rvalue::Aggregate(box AggregateKind::Array(el_ty), fields))
            }
            ExprKind::Tuple { fields } => {
                // see (*) above
                // first process the set of fields
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|f| unpack!(block = this.as_operand(block, scope, f)))
                    .collect();

                this.record_operands_moved(&fields);
                block.and(Rvalue::Aggregate(box AggregateKind::Tuple, fields))
            }
            ExprKind::Closure { closure_id, substs, upvars, movability } => {
                // see (*) above
                let operands: Vec<_> = upvars
                    .into_iter()
                    .map(|upvar| {
                        let upvar = this.hir.mirror(upvar);
                        match Category::of(&upvar.kind) {
                            // Use as_place to avoid creating a temporary when
                            // moving a variable into a closure, so that
                            // borrowck knows which variables to mark as being
                            // used as mut. This is OK here because the upvar
                            // expressions have no side effects and act on
                            // disjoint places.
                            // This occurs when capturing by copy/move, while
                            // by reference captures use as_operand
                            Some(Category::Place) => {
                                let place = unpack!(block = this.as_place(block, upvar));
                                this.consume_by_copy_or_move(place)
                            }
                            _ => {
                                // Turn mutable borrow captures into unique
                                // borrow captures when capturing an immutable
                                // variable. This is sound because the mutation
                                // that caused the capture will cause an error.
                                match upvar.kind {
                                    ExprKind::Borrow {
                                        borrow_kind:
                                            BorrowKind::Mut { allow_two_phase_borrow: false },
                                        arg,
                                    } => unpack!(
                                        block = this.limit_capture_mutability(
                                            upvar.span, upvar.ty, scope, block, arg,
                                        )
                                    ),
                                    _ => unpack!(block = this.as_operand(block, scope, upvar)),
                                }
                            }
                        }
                    })
                    .collect();
                let result = match substs {
                    UpvarSubsts::Generator(substs) => {
                        // We implicitly set the discriminant to 0. See
                        // librustc_mir/transform/deaggregator.rs for details.
                        let movability = movability.unwrap();
                        box AggregateKind::Generator(closure_id, substs, movability)
                    }
                    UpvarSubsts::Closure(substs) => box AggregateKind::Closure(closure_id, substs),
                };
                this.record_operands_moved(&operands);
                block.and(Rvalue::Aggregate(result, operands))
            }
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                block = unpack!(this.stmt_expr(block, expr, None));
                block.and(Rvalue::Use(Operand::Constant(box Constant {
                    span: expr_span,
                    user_ty: None,
                    literal: ty::Const::zero_sized(this.hir.tcx(), this.hir.tcx().types.unit),
                })))
            }
            ExprKind::Yield { .. }
            | ExprKind::Literal { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::Block { .. }
            | ExprKind::Match { .. }
            | ExprKind::If { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::Use { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::AddressOf { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Loop { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Call { .. }
            | ExprKind::Field { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Index { .. }
            | ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::LlvmInlineAsm { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. } => {
                // these do not have corresponding `Rvalue` variants,
                // so make an operand and then return that
                debug_assert!(!matches!(Category::of(&expr.kind), Some(Category::Rvalue(RvalueFunc::AsRvalue))));
                let operand = unpack!(block = this.as_operand(block, scope, expr));
                block.and(Rvalue::Use(operand))
            }
        }
    }

    crate fn build_binary_op(
        &mut self,
        mut block: BasicBlock,
        op: BinOp,
        span: Span,
        ty: Ty<'tcx>,
        lhs: Operand<'tcx>,
        rhs: Operand<'tcx>,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let source_info = self.source_info(span);
        let bool_ty = self.hir.bool_ty();
        if self.hir.check_overflow() && op.is_checkable() && ty.is_integral() {
            let result_tup = self.hir.tcx().intern_tup(&[ty, bool_ty]);
            let result_value = self.temp(result_tup, span);

            self.cfg.push_assign(
                block,
                source_info,
                result_value,
                Rvalue::CheckedBinaryOp(op, lhs.to_copy(), rhs.to_copy()),
            );
            let val_fld = Field::new(0);
            let of_fld = Field::new(1);

            let tcx = self.hir.tcx();
            let val = tcx.mk_place_field(result_value, val_fld, ty);
            let of = tcx.mk_place_field(result_value, of_fld, bool_ty);

            let err = AssertKind::Overflow(op, lhs, rhs);

            block = self.assert(block, Operand::Move(of), false, err, span);

            block.and(Rvalue::Use(Operand::Move(val)))
        } else {
            if ty.is_integral() && (op == BinOp::Div || op == BinOp::Rem) {
                // Checking division and remainder is more complex, since we 1. always check
                // and 2. there are two possible failure cases, divide-by-zero and overflow.

                let zero_err = if op == BinOp::Div {
                    AssertKind::DivisionByZero(lhs.to_copy())
                } else {
                    AssertKind::RemainderByZero(lhs.to_copy())
                };
                let overflow_err = AssertKind::Overflow(op, lhs.to_copy(), rhs.to_copy());

                // Check for / 0
                let is_zero = self.temp(bool_ty, span);
                let zero = self.zero_literal(span, ty);
                self.cfg.push_assign(
                    block,
                    source_info,
                    is_zero,
                    Rvalue::BinaryOp(BinOp::Eq, rhs.to_copy(), zero),
                );

                block = self.assert(block, Operand::Move(is_zero), false, zero_err, span);

                // We only need to check for the overflow in one case:
                // MIN / -1, and only for signed values.
                if ty.is_signed() {
                    let neg_1 = self.neg_1_literal(span, ty);
                    let min = self.minval_literal(span, ty);

                    let is_neg_1 = self.temp(bool_ty, span);
                    let is_min = self.temp(bool_ty, span);
                    let of = self.temp(bool_ty, span);

                    // this does (rhs == -1) & (lhs == MIN). It could short-circuit instead

                    self.cfg.push_assign(
                        block,
                        source_info,
                        is_neg_1,
                        Rvalue::BinaryOp(BinOp::Eq, rhs.to_copy(), neg_1),
                    );
                    self.cfg.push_assign(
                        block,
                        source_info,
                        is_min,
                        Rvalue::BinaryOp(BinOp::Eq, lhs.to_copy(), min),
                    );

                    let is_neg_1 = Operand::Move(is_neg_1);
                    let is_min = Operand::Move(is_min);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        of,
                        Rvalue::BinaryOp(BinOp::BitAnd, is_neg_1, is_min),
                    );

                    block = self.assert(block, Operand::Move(of), false, overflow_err, span);
                }
            }

            block.and(Rvalue::BinaryOp(op, lhs, rhs))
        }
    }

    fn limit_capture_mutability(
        &mut self,
        upvar_span: Span,
        upvar_ty: Ty<'tcx>,
        temp_lifetime: Option<region::Scope>,
        mut block: BasicBlock,
        arg: ExprRef<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        let this = self;

        let source_info = this.source_info(upvar_span);
        let temp = this.local_decls.push(LocalDecl::new(upvar_ty, upvar_span));

        this.cfg.push(block, Statement { source_info, kind: StatementKind::StorageLive(temp) });

        let arg_place_builder = unpack!(block = this.as_place_builder(block, arg));

        let mutability = match arg_place_builder.base() {
            // We are capturing a path that starts off a local variable in the parent.
            // The mutability of the current capture is same as the mutability
            // of the local declaration in the parent.
            PlaceBase::Local(local) =>  this.local_decls[local].mutability,
            // Parent is a closure and we are capturing a path that is captured
            // by the parent itself. The mutability of the current capture
            // is same as that of the capture in the parent closure.
            PlaceBase::Upvar { .. } => {
                let enclosing_upvars_resolved = arg_place_builder.clone().into_place(
                    this.hir.tcx(),
                    this.hir.typeck_results());

                match enclosing_upvars_resolved.as_ref() {
                    PlaceRef { local, projection: &[ProjectionElem::Field(upvar_index, _), ..] }
                    | PlaceRef {
                        local,
                        projection: &[ProjectionElem::Deref, ProjectionElem::Field(upvar_index, _), ..] } => {
                            // Not in a closure
                            debug_assert!(
                                local == Local::new(1),
                                "Expected local to be Local(1), found {:?}",
                                local
                            );
                            // Not in a closure
                            debug_assert!(
                                this.upvar_mutbls.len() > upvar_index.index(),
                                "Unexpected capture place, upvar_mutbls={:#?}, upvar_index={:?}",
                                this.upvar_mutbls, upvar_index
                            );
                            this.upvar_mutbls[upvar_index.index()]
                        }
                    _ => bug!("Unexpected capture place"),
                }
            }
        };

        let borrow_kind = match mutability {
            Mutability::Not => BorrowKind::Unique,
            Mutability::Mut => BorrowKind::Mut { allow_two_phase_borrow: false },
        };

        let arg_place = arg_place_builder.into_place(
                    this.hir.tcx(),
                    this.hir.typeck_results());

        this.cfg.push_assign(
            block,
            source_info,
            Place::from(temp),
            Rvalue::Ref(this.hir.tcx().lifetimes.re_erased, borrow_kind, arg_place),
        );

        // See the comment in `expr_as_temp` and on the `rvalue_scopes` field for why
        // this can be `None`.
        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop_storage_and_value(upvar_span, temp_lifetime, temp);
        }

        block.and(Operand::Move(Place::from(temp)))
    }

    // Helper to get a `-1` value of the appropriate type
    fn neg_1_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        let param_ty = ty::ParamEnv::empty().and(ty);
        let bits = self.hir.tcx().layout_of(param_ty).unwrap().size.bits();
        let n = (!0u128) >> (128 - bits);
        let literal = ty::Const::from_bits(self.hir.tcx(), n, param_ty);

        self.literal_operand(span, literal)
    }

    // Helper to get the minimum value of the appropriate type
    fn minval_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        assert!(ty.is_signed());
        let param_ty = ty::ParamEnv::empty().and(ty);
        let bits = self.hir.tcx().layout_of(param_ty).unwrap().size.bits();
        let n = 1 << (bits - 1);
        let literal = ty::Const::from_bits(self.hir.tcx(), n, param_ty);

        self.literal_operand(span, literal)
    }
}
