//! See docs in `build/expr/mod.rs`.

use rustc_index::vec::Idx;

use crate::build::expr::as_place::PlaceBase;
use crate::build::expr::category::{Category, RvalueFunc};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use rustc_hir::lang_items::LangItem;
use rustc_middle::middle::region;
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::Place;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty, UpvarSubsts};
use rustc_span::Span;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Returns an rvalue suitable for use until the end of the current
    /// scope expression.
    ///
    /// The operand returned from this function will *not be valid* after
    /// an ExprKind::Scope is passed, so please do *not* return it from
    /// functions to avoid bad miscompiles.
    crate fn as_local_rvalue(
        &mut self,
        block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let local_scope = self.local_scope();
        self.as_rvalue(block, Some(local_scope), expr)
    }

    /// Compile `expr`, yielding an rvalue.
    crate fn as_rvalue(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Rvalue<'tcx>> {
        debug!("expr_as_rvalue(block={:?}, scope={:?}, expr={:?})", block, scope, expr);

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::ThreadLocalRef(did) => block.and(Rvalue::ThreadLocalRef(did)),
            ExprKind::Scope { region_scope, lint_level, value } => {
                let region_scope = (region_scope, source_info);
                this.in_scope(region_scope, lint_level, |this| {
                    this.as_rvalue(block, scope, &this.thir[value])
                })
            }
            ExprKind::Repeat { value, count } => {
                let value_operand =
                    unpack!(block = this.as_operand(block, scope, &this.thir[value], None));
                block.and(Rvalue::Repeat(value_operand, count))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs = unpack!(block = this.as_operand(block, scope, &this.thir[lhs], None));
                let rhs = unpack!(block = this.as_operand(block, scope, &this.thir[rhs], None));
                this.build_binary_op(block, op, expr_span, expr.ty, lhs, rhs)
            }
            ExprKind::Unary { op, arg } => {
                let arg = unpack!(block = this.as_operand(block, scope, &this.thir[arg], None));
                // Check for -MIN on signed integers
                if this.check_overflow && op == UnOp::Neg && expr.ty.is_signed() {
                    let bool_ty = this.tcx.types.bool;

                    let minval = this.minval_literal(expr_span, expr.ty);
                    let is_min = this.temp(bool_ty, expr_span);

                    this.cfg.push_assign(
                        block,
                        source_info,
                        is_min,
                        Rvalue::BinaryOp(BinOp::Eq, Box::new((arg.to_copy(), minval))),
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
                let value = &this.thir[value];
                let tcx = this.tcx;

                // `exchange_malloc` is unsafe but box is safe, so need a new scope.
                let synth_scope = this.new_source_scope(
                    expr_span,
                    LintLevel::Inherited,
                    Some(Safety::BuiltinUnsafe),
                );
                let synth_info = SourceInfo { span: expr_span, scope: synth_scope };

                let size = this.temp(tcx.types.usize, expr_span);
                this.cfg.push_assign(
                    block,
                    synth_info,
                    size,
                    Rvalue::NullaryOp(NullOp::SizeOf, value.ty),
                );

                let align = this.temp(tcx.types.usize, expr_span);
                this.cfg.push_assign(
                    block,
                    synth_info,
                    align,
                    Rvalue::NullaryOp(NullOp::AlignOf, value.ty),
                );

                // malloc some memory of suitable size and align:
                let exchange_malloc = Operand::function_handle(
                    tcx,
                    tcx.require_lang_item(LangItem::ExchangeMalloc, Some(expr_span)),
                    ty::List::empty(),
                    expr_span,
                );
                let storage = this.temp(tcx.mk_mut_ptr(tcx.types.u8), expr_span);
                let success = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    synth_info,
                    TerminatorKind::Call {
                        func: exchange_malloc,
                        args: vec![Operand::Move(size), Operand::Move(align)],
                        destination: Some((Place::from(storage), success)),
                        cleanup: None,
                        from_hir_call: false,
                        fn_span: expr_span,
                    },
                );
                this.diverge_from(block);
                block = success;

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

                // Transmute `*mut u8` to the box (thus far, uninitialized):
                let box_ = Rvalue::ShallowInitBox(Operand::Move(Place::from(storage)), value.ty);
                this.cfg.push_assign(block, source_info, Place::from(result), box_);

                // initialize the box contents:
                unpack!(
                    block = this.expr_into_dest(
                        this.tcx.mk_place_deref(Place::from(result)),
                        block,
                        value
                    )
                );
                block.and(Rvalue::Use(Operand::Move(Place::from(result))))
            }
            ExprKind::Cast { source } => {
                let source =
                    unpack!(block = this.as_operand(block, scope, &this.thir[source], None));
                block.and(Rvalue::Cast(CastKind::Misc, source, expr.ty))
            }
            ExprKind::Pointer { cast, source } => {
                let source =
                    unpack!(block = this.as_operand(block, scope, &this.thir[source], None));
                block.and(Rvalue::Cast(CastKind::Pointer(cast), source, expr.ty))
            }
            ExprKind::Array { ref fields } => {
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
                let el_ty = expr.ty.sequence_element_type(this.tcx);
                let fields: Vec<_> = fields
                    .into_iter()
                    .copied()
                    .map(|f| unpack!(block = this.as_operand(block, scope, &this.thir[f], None)))
                    .collect();

                block.and(Rvalue::Aggregate(Box::new(AggregateKind::Array(el_ty)), fields))
            }
            ExprKind::Tuple { ref fields } => {
                // see (*) above
                // first process the set of fields
                let fields: Vec<_> = fields
                    .into_iter()
                    .copied()
                    .map(|f| unpack!(block = this.as_operand(block, scope, &this.thir[f], None)))
                    .collect();

                block.and(Rvalue::Aggregate(Box::new(AggregateKind::Tuple), fields))
            }
            ExprKind::Closure { closure_id, substs, ref upvars, movability, ref fake_reads } => {
                // Convert the closure fake reads, if any, from `ExprRef` to mir `Place`
                // and push the fake reads.
                // This must come before creating the operands. This is required in case
                // there is a fake read and a borrow of the same path, since otherwise the
                // fake read might interfere with the borrow. Consider an example like this
                // one:
                // ```
                // let mut x = 0;
                // let c = || {
                //     &mut x; // mutable borrow of `x`
                //     match x { _ => () } // fake read of `x`
                // };
                // ```
                //
                for (thir_place, cause, hir_id) in fake_reads.into_iter() {
                    let place_builder =
                        unpack!(block = this.as_place_builder(block, &this.thir[*thir_place]));

                    if let Ok(place_builder_resolved) =
                        place_builder.try_upvars_resolved(this.tcx, this.typeck_results)
                    {
                        let mir_place =
                            place_builder_resolved.into_place(this.tcx, this.typeck_results);
                        this.cfg.push_fake_read(
                            block,
                            this.source_info(this.tcx.hir().span(*hir_id)),
                            *cause,
                            mir_place,
                        );
                    }
                }

                // see (*) above
                let operands: Vec<_> = upvars
                    .into_iter()
                    .copied()
                    .map(|upvar| {
                        let upvar = &this.thir[upvar];
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
                                            upvar.span,
                                            upvar.ty,
                                            scope,
                                            block,
                                            &this.thir[arg],
                                        )
                                    ),
                                    _ => {
                                        unpack!(block = this.as_operand(block, scope, upvar, None))
                                    }
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
                        Box::new(AggregateKind::Generator(closure_id, substs, movability))
                    }
                    UpvarSubsts::Closure(substs) => {
                        Box::new(AggregateKind::Closure(closure_id, substs))
                    }
                };
                block.and(Rvalue::Aggregate(result, operands))
            }
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                block = unpack!(this.stmt_expr(block, expr, None));
                block.and(Rvalue::Use(Operand::Constant(Box::new(Constant {
                    span: expr_span,
                    user_ty: None,
                    literal: ty::Const::zero_sized(this.tcx, this.tcx.types.unit).into(),
                }))))
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
            | ExprKind::Let { .. }
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
                debug_assert!(!matches!(
                    Category::of(&expr.kind),
                    Some(Category::Rvalue(RvalueFunc::AsRvalue))
                ));
                let operand = unpack!(block = this.as_operand(block, scope, expr, None));
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
        let bool_ty = self.tcx.types.bool;
        if self.check_overflow && op.is_checkable() && ty.is_integral() {
            let result_tup = self.tcx.intern_tup(&[ty, bool_ty]);
            let result_value = self.temp(result_tup, span);

            self.cfg.push_assign(
                block,
                source_info,
                result_value,
                Rvalue::CheckedBinaryOp(op, Box::new((lhs.to_copy(), rhs.to_copy()))),
            );
            let val_fld = Field::new(0);
            let of_fld = Field::new(1);

            let tcx = self.tcx;
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
                    Rvalue::BinaryOp(BinOp::Eq, Box::new((rhs.to_copy(), zero))),
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
                        Rvalue::BinaryOp(BinOp::Eq, Box::new((rhs.to_copy(), neg_1))),
                    );
                    self.cfg.push_assign(
                        block,
                        source_info,
                        is_min,
                        Rvalue::BinaryOp(BinOp::Eq, Box::new((lhs.to_copy(), min))),
                    );

                    let is_neg_1 = Operand::Move(is_neg_1);
                    let is_min = Operand::Move(is_min);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        of,
                        Rvalue::BinaryOp(BinOp::BitAnd, Box::new((is_neg_1, is_min))),
                    );

                    block = self.assert(block, Operand::Move(of), false, overflow_err, span);
                }
            }

            block.and(Rvalue::BinaryOp(op, Box::new((lhs, rhs))))
        }
    }

    fn limit_capture_mutability(
        &mut self,
        upvar_span: Span,
        upvar_ty: Ty<'tcx>,
        temp_lifetime: Option<region::Scope>,
        mut block: BasicBlock,
        arg: &Expr<'tcx>,
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
            PlaceBase::Local(local) => this.local_decls[local].mutability,
            // Parent is a closure and we are capturing a path that is captured
            // by the parent itself. The mutability of the current capture
            // is same as that of the capture in the parent closure.
            PlaceBase::Upvar { .. } => {
                let enclosing_upvars_resolved =
                    arg_place_builder.clone().into_place(this.tcx, this.typeck_results);

                match enclosing_upvars_resolved.as_ref() {
                    PlaceRef {
                        local,
                        projection: &[ProjectionElem::Field(upvar_index, _), ..],
                    }
                    | PlaceRef {
                        local,
                        projection:
                            &[ProjectionElem::Deref, ProjectionElem::Field(upvar_index, _), ..],
                    } => {
                        // Not in a closure
                        debug_assert!(
                            local == ty::CAPTURE_STRUCT_LOCAL,
                            "Expected local to be Local(1), found {:?}",
                            local
                        );
                        // Not in a closure
                        debug_assert!(
                            this.upvar_mutbls.len() > upvar_index.index(),
                            "Unexpected capture place, upvar_mutbls={:#?}, upvar_index={:?}",
                            this.upvar_mutbls,
                            upvar_index
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

        let arg_place = arg_place_builder.into_place(this.tcx, this.typeck_results);

        this.cfg.push_assign(
            block,
            source_info,
            Place::from(temp),
            Rvalue::Ref(this.tcx.lifetimes.re_erased, borrow_kind, arg_place),
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
        let size = self.tcx.layout_of(param_ty).unwrap().size;
        let literal = ty::Const::from_bits(self.tcx, size.unsigned_int_max(), param_ty);

        self.literal_operand(span, literal)
    }

    // Helper to get the minimum value of the appropriate type
    fn minval_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        assert!(ty.is_signed());
        let param_ty = ty::ParamEnv::empty().and(ty);
        let bits = self.tcx.layout_of(param_ty).unwrap().size.bits();
        let n = 1 << (bits - 1);
        let literal = ty::Const::from_bits(self.tcx, n, param_ty);

        self.literal_operand(span, literal)
    }
}
