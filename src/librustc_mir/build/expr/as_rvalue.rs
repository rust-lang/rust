//! See docs in `build/expr/mod.rs`.

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;

use crate::build::expr::category::{Category, RvalueFunc};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::hair::*;
use rustc::middle::region;
use rustc::mir::interpret::InterpError;
use rustc::mir::*;
use rustc::ty::{self, CanonicalUserTypeAnnotation, Ty, UpvarSubsts};
use syntax_pos::Span;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// See comment on `as_local_operand`
    pub fn as_local_rvalue<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Rvalue<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let local_scope = self.local_scope();
        self.as_rvalue(block, local_scope, expr)
    }

    /// Compile `expr`, yielding an rvalue.
    pub fn as_rvalue<M>(
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
        debug!(
            "expr_as_rvalue(block={:?}, scope={:?}, expr={:?})",
            block, scope, expr
        );

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::Scope {
                region_scope,
                lint_level,
                value,
            } => {
                let region_scope = (region_scope, source_info);
                this.in_scope(region_scope, lint_level, |this| {
                    this.as_rvalue(block, scope, value)
                })
            }
            ExprKind::Repeat { value, count } => {
                let value_operand = unpack!(block = this.as_operand(block, scope, value));
                block.and(Rvalue::Repeat(value_operand, count))
            }
            ExprKind::Borrow {
                borrow_kind,
                arg,
            } => {
                let arg_place = match borrow_kind {
                    BorrowKind::Shared => unpack!(block = this.as_read_only_place(block, arg)),
                    _ => unpack!(block = this.as_place(block, arg)),
                };
                block.and(Rvalue::Ref(this.hir.tcx().lifetimes.re_erased, borrow_kind, arg_place))
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
                        &is_min,
                        Rvalue::BinaryOp(BinOp::Eq, arg.to_copy(), minval),
                    );

                    block = this.assert(
                        block,
                        Operand::Move(is_min),
                        false,
                        InterpError::OverflowNeg,
                        expr_span,
                    );
                }
                block.and(Rvalue::UnaryOp(op, arg))
            }
            ExprKind::Box { value } => {
                let value = this.hir.mirror(value);
                // The `Box<T>` temporary created here is not a part of the HIR,
                // and therefore is not considered during generator OIBIT
                // determination. See the comment about `box` at `yield_in_scope`.
                let result = this
                    .local_decls
                    .push(LocalDecl::new_internal(expr.ty, expr_span));
                this.cfg.push(
                    block,
                    Statement {
                        source_info,
                        kind: StatementKind::StorageLive(result),
                    },
                );
                if let Some(scope) = scope {
                    // schedule a shallow free of that memory, lest we unwind:
                    this.schedule_drop_storage_and_value(
                        expr_span,
                        scope,
                        result,
                        value.ty,
                    );
                }

                // malloc some memory of suitable type (thus far, uninitialized):
                let box_ = Rvalue::NullaryOp(NullOp::Box, value.ty);
                this.cfg
                    .push_assign(block, source_info, &Place::from(result), box_);

                // initialize the box contents:
                unpack!(
                    block = this.into(
                        &Place::from(result).deref(),
                        block, value
                    )
                );
                block.and(Rvalue::Use(Operand::Move(Place::from(result))))
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

                block.and(Rvalue::Aggregate(box AggregateKind::Array(el_ty), fields))
            }
            ExprKind::Tuple { fields } => {
                // see (*) above
                // first process the set of fields
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|f| unpack!(block = this.as_operand(block, scope, f)))
                    .collect();

                block.and(Rvalue::Aggregate(box AggregateKind::Tuple, fields))
            }
            ExprKind::Closure {
                closure_id,
                substs,
                upvars,
                movability,
            } => {
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
                                            BorrowKind::Mut {
                                                allow_two_phase_borrow: false,
                                            },
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
                    }).collect();
                let result = match substs {
                    UpvarSubsts::Generator(substs) => {
                        // We implicitly set the discriminant to 0. See
                        // librustc_mir/transform/deaggregator.rs for details.
                        let movability = movability.unwrap();
                        box AggregateKind::Generator(closure_id, substs, movability)
                    }
                    UpvarSubsts::Closure(substs) => box AggregateKind::Closure(closure_id, substs),
                };
                block.and(Rvalue::Aggregate(result, operands))
            }
            ExprKind::Adt {
                adt_def,
                variant_index,
                substs,
                user_ty,
                fields,
                base,
            } => {
                // see (*) above
                let is_union = adt_def.is_union();
                let active_field_index = if is_union {
                    Some(fields[0].name.index())
                } else {
                    None
                };

                // first process the set of fields that were provided
                // (evaluating them in order given by user)
                let fields_map: FxHashMap<_, _> = fields
                    .into_iter()
                    .map(|f| {
                        (
                            f.name,
                            unpack!(block = this.as_operand(block, scope, f.expr)),
                        )
                    }).collect();

                let field_names = this.hir.all_fields(adt_def, variant_index);

                let fields = if let Some(FruInfo { base, field_types }) = base {
                    let base = unpack!(block = this.as_place(block, base));

                    // MIR does not natively support FRU, so for each
                    // base-supplied field, generate an operand that
                    // reads it from the base.
                    field_names
                        .into_iter()
                        .zip(field_types.into_iter())
                        .map(|(n, ty)| match fields_map.get(&n) {
                            Some(v) => v.clone(),
                            None => this.consume_by_copy_or_move(base.clone().field(n, ty)),
                        }).collect()
                } else {
                    field_names
                        .iter()
                        .filter_map(|n| fields_map.get(n).cloned())
                        .collect()
                };

                let inferred_ty = expr.ty;
                let user_ty = user_ty.map(|ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span: source_info.span,
                        user_ty: ty,
                        inferred_ty,
                    })
                });
                let adt = box AggregateKind::Adt(
                    adt_def,
                    variant_index,
                    substs,
                    user_ty,
                    active_field_index,
                );
                block.and(Rvalue::Aggregate(adt, fields))
            }
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                block = unpack!(this.stmt_expr(block, expr, None));
                block.and(this.unit_rvalue())
            }
            ExprKind::Yield { value } => {
                let value = unpack!(block = this.as_operand(block, scope, value));
                let resume = this.cfg.start_new_block();
                let cleanup = this.generator_drop_cleanup();
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Yield {
                        value: value,
                        resume: resume,
                        drop: cleanup,
                    },
                );
                resume.and(this.unit_rvalue())
            }
            ExprKind::Literal { .. }
            | ExprKind::Block { .. }
            | ExprKind::Match { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::Use { .. }
            | ExprKind::Loop { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Call { .. }
            | ExprKind::Field { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Index { .. }
            | ExprKind::VarRef { .. }
            | ExprKind::SelfRef
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. } => {
                // these do not have corresponding `Rvalue` variants,
                // so make an operand and then return that
                debug_assert!(match Category::of(&expr.kind) {
                    Some(Category::Rvalue(RvalueFunc::AsRvalue)) => false,
                    _ => true,
                });
                let operand = unpack!(block = this.as_operand(block, scope, expr));
                block.and(Rvalue::Use(operand))
            }
        }
    }

    pub fn build_binary_op(
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
                &result_value,
                Rvalue::CheckedBinaryOp(op, lhs, rhs),
            );
            let val_fld = Field::new(0);
            let of_fld = Field::new(1);

            let val = result_value.clone().field(val_fld, ty);
            let of = result_value.field(of_fld, bool_ty);

            let err = InterpError::Overflow(op);

            block = self.assert(block, Operand::Move(of), false, err, span);

            block.and(Rvalue::Use(Operand::Move(val)))
        } else {
            if ty.is_integral() && (op == BinOp::Div || op == BinOp::Rem) {
                // Checking division and remainder is more complex, since we 1. always check
                // and 2. there are two possible failure cases, divide-by-zero and overflow.

                let (zero_err, overflow_err) = if op == BinOp::Div {
                    (InterpError::DivisionByZero, InterpError::Overflow(op))
                } else {
                    (InterpError::RemainderByZero, InterpError::Overflow(op))
                };

                // Check for / 0
                let is_zero = self.temp(bool_ty, span);
                let zero = self.zero_literal(span, ty);
                self.cfg.push_assign(
                    block,
                    source_info,
                    &is_zero,
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
                        &is_neg_1,
                        Rvalue::BinaryOp(BinOp::Eq, rhs.to_copy(), neg_1),
                    );
                    self.cfg.push_assign(
                        block,
                        source_info,
                        &is_min,
                        Rvalue::BinaryOp(BinOp::Eq, lhs.to_copy(), min),
                    );

                    let is_neg_1 = Operand::Move(is_neg_1);
                    let is_min = Operand::Move(is_min);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        &of,
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
        let temp = this
            .local_decls
            .push(LocalDecl::new_temp(upvar_ty, upvar_span));

        this.cfg.push(
            block,
            Statement {
                source_info,
                kind: StatementKind::StorageLive(temp),
            },
        );

        let arg_place = unpack!(block = this.as_place(block, arg));

        let mutability = match arg_place {
            Place::Base(PlaceBase::Local(local)) => this.local_decls[local].mutability,
            Place::Projection(box Projection {
                base: Place::Base(PlaceBase::Local(local)),
                elem: ProjectionElem::Deref,
            }) => {
                debug_assert!(
                    this.local_decls[local].is_ref_for_guard(),
                    "Unexpected capture place",
                );
                this.local_decls[local].mutability
            }
            Place::Projection(box Projection {
                ref base,
                elem: ProjectionElem::Field(upvar_index, _),
            })
            | Place::Projection(box Projection {
                base:
                    Place::Projection(box Projection {
                        ref base,
                        elem: ProjectionElem::Field(upvar_index, _),
                    }),
                elem: ProjectionElem::Deref,
            }) => {
                // Not projected from the implicit `self` in a closure.
                debug_assert!(
                    match base.local_or_deref_local() {
                        Some(local) => local == Local::new(1),
                        None => false,
                    },
                    "Unexpected capture place"
                );
                // Not in a closure
                debug_assert!(
                    this.upvar_mutbls.len() > upvar_index.index(),
                    "Unexpected capture place"
                );
                this.upvar_mutbls[upvar_index.index()]
            }
            _ => bug!("Unexpected capture place"),
        };

        let borrow_kind = match mutability {
            Mutability::Not => BorrowKind::Unique,
            Mutability::Mut => BorrowKind::Mut {
                allow_two_phase_borrow: false,
            },
        };

        this.cfg.push_assign(
            block,
            source_info,
            &Place::from(temp),
            Rvalue::Ref(this.hir.tcx().lifetimes.re_erased, borrow_kind, arg_place),
        );

        // In constants, temp_lifetime is None. We should not need to drop
        // anything because no values with a destructor can be created in
        // a constant at this time, even if the type may need dropping.
        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop_storage_and_value(
                upvar_span,
                temp_lifetime,
                temp,
                upvar_ty,
            );
        }

        block.and(Operand::Move(Place::from(temp)))
    }

    // Helper to get a `-1` value of the appropriate type
    fn neg_1_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        let param_ty = ty::ParamEnv::empty().and(ty);
        let bits = self.hir.tcx().layout_of(param_ty).unwrap().size.bits();
        let n = (!0u128) >> (128 - bits);
        let literal = ty::Const::from_bits(self.hir.tcx(), n, param_ty);

        self.literal_operand(span, ty, literal)
    }

    // Helper to get the minimum value of the appropriate type
    fn minval_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        assert!(ty.is_signed());
        let param_ty = ty::ParamEnv::empty().and(ty);
        let bits = self.hir.tcx().layout_of(param_ty).unwrap().size.bits();
        let n = 1 << (bits - 1);
        let literal = ty::Const::from_bits(self.hir.tcx(), n, param_ty);

        self.literal_operand(span, ty, literal)
    }
}
