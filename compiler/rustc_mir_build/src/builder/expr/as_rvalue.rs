//! See docs in `build/expr/mod.rs`.

use rustc_abi::{BackendRepr, FieldIdx, Primitive};
use rustc_hir::lang_items::LangItem;
use rustc_index::{Idx, IndexVec};
use rustc_middle::bug;
use rustc_middle::middle::region;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::cast::{CastTy, mir_cast_kind};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, Ty, UpvarArgs};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span};
use tracing::debug;

use crate::builder::expr::as_place::PlaceBase;
use crate::builder::expr::category::{Category, RvalueFunc};
use crate::builder::{BlockAnd, BlockAndExtension, Builder, NeedsTemporary};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Returns an rvalue suitable for use until the end of the current
    /// scope expression.
    ///
    /// The operand returned from this function will *not be valid* after
    /// an ExprKind::Scope is passed, so please do *not* return it from
    /// functions to avoid bad miscompiles.
    pub(crate) fn as_local_rvalue(
        &mut self,
        block: BasicBlock,
        expr_id: ExprId,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let local_scope = self.local_scope();
        self.as_rvalue(
            block,
            TempLifetime { temp_lifetime: Some(local_scope), backwards_incompatible: None },
            expr_id,
        )
    }

    /// Compile `expr`, yielding an rvalue.
    pub(crate) fn as_rvalue(
        &mut self,
        mut block: BasicBlock,
        scope: TempLifetime,
        expr_id: ExprId,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let this = self;
        let expr = &this.thir[expr_id];
        debug!("expr_as_rvalue(block={:?}, scope={:?}, expr={:?})", block, scope, expr);

        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::ThreadLocalRef(did) => block.and(Rvalue::ThreadLocalRef(did)),
            ExprKind::Scope { region_scope, lint_level, value } => {
                let region_scope = (region_scope, source_info);
                this.in_scope(region_scope, lint_level, |this| this.as_rvalue(block, scope, value))
            }
            ExprKind::Repeat { value, count } => {
                if Some(0) == count.try_to_target_usize(this.tcx) {
                    this.build_zero_repeat(block, value, scope, source_info)
                } else {
                    let value_operand = unpack!(
                        block = this.as_operand(
                            block,
                            scope,
                            value,
                            LocalInfo::Boring,
                            NeedsTemporary::No
                        )
                    );
                    block.and(Rvalue::Repeat(value_operand, count))
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs = unpack!(
                    block = this.as_operand(
                        block,
                        scope,
                        lhs,
                        LocalInfo::Boring,
                        NeedsTemporary::Maybe
                    )
                );
                let rhs = unpack!(
                    block =
                        this.as_operand(block, scope, rhs, LocalInfo::Boring, NeedsTemporary::No)
                );
                this.build_binary_op(block, op, expr_span, expr.ty, lhs, rhs)
            }
            ExprKind::Unary { op, arg } => {
                let arg = unpack!(
                    block =
                        this.as_operand(block, scope, arg, LocalInfo::Boring, NeedsTemporary::No)
                );
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
                let value_ty = this.thir[value].ty;
                let tcx = this.tcx;
                let source_info = this.source_info(expr_span);

                let size = this.temp(tcx.types.usize, expr_span);
                this.cfg.push_assign(
                    block,
                    source_info,
                    size,
                    Rvalue::NullaryOp(NullOp::SizeOf, value_ty),
                );

                let align = this.temp(tcx.types.usize, expr_span);
                this.cfg.push_assign(
                    block,
                    source_info,
                    align,
                    Rvalue::NullaryOp(NullOp::AlignOf, value_ty),
                );

                // malloc some memory of suitable size and align:
                let exchange_malloc = Operand::function_handle(
                    tcx,
                    tcx.require_lang_item(LangItem::ExchangeMalloc, expr_span),
                    [],
                    expr_span,
                );
                let storage = this.temp(Ty::new_mut_ptr(tcx, tcx.types.u8), expr_span);
                let success = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Call {
                        func: exchange_malloc,
                        args: [
                            Spanned { node: Operand::Move(size), span: DUMMY_SP },
                            Spanned { node: Operand::Move(align), span: DUMMY_SP },
                        ]
                        .into(),
                        destination: storage,
                        target: Some(success),
                        unwind: UnwindAction::Continue,
                        call_source: CallSource::Misc,
                        fn_span: expr_span,
                    },
                );
                this.diverge_from(block);
                block = success;

                // The `Box<T>` temporary created here is not a part of the HIR,
                // and therefore is not considered during coroutine auto-trait
                // determination. See the comment about `box` at `yield_in_scope`.
                let result = this.local_decls.push(LocalDecl::new(expr.ty, expr_span));
                this.cfg
                    .push(block, Statement::new(source_info, StatementKind::StorageLive(result)));
                if let Some(scope) = scope.temp_lifetime {
                    // schedule a shallow free of that memory, lest we unwind:
                    this.schedule_drop_storage_and_value(expr_span, scope, result);
                }

                // Transmute `*mut u8` to the box (thus far, uninitialized):
                let box_ = Rvalue::ShallowInitBox(Operand::Move(storage), value_ty);
                this.cfg.push_assign(block, source_info, Place::from(result), box_);

                // initialize the box contents:
                block = this
                    .expr_into_dest(this.tcx.mk_place_deref(Place::from(result)), block, value)
                    .into_block();
                block.and(Rvalue::Use(Operand::Move(Place::from(result))))
            }
            ExprKind::Cast { source } => {
                let source_expr = &this.thir[source];

                // Casting an enum to an integer is equivalent to computing the discriminant and casting the
                // discriminant. Previously every backend had to repeat the logic for this operation. Now we
                // create all the steps directly in MIR with operations all backends need to support anyway.
                let (source, ty) = if let ty::Adt(adt_def, ..) = source_expr.ty.kind()
                    && adt_def.is_enum()
                {
                    let discr_ty = adt_def.repr().discr_type().to_ty(this.tcx);
                    let temp = unpack!(block = this.as_temp(block, scope, source, Mutability::Not));
                    let layout =
                        this.tcx.layout_of(this.typing_env().as_query_input(source_expr.ty));
                    let discr = this.temp(discr_ty, source_expr.span);
                    this.cfg.push_assign(
                        block,
                        source_info,
                        discr,
                        Rvalue::Discriminant(temp.into()),
                    );
                    let (op, ty) = (Operand::Move(discr), discr_ty);

                    if let BackendRepr::Scalar(scalar) = layout.unwrap().backend_repr
                        && !scalar.is_always_valid(&this.tcx)
                        && let Primitive::Int(int_width, _signed) = scalar.primitive()
                    {
                        let unsigned_ty = int_width.to_ty(this.tcx, false);
                        let unsigned_place = this.temp(unsigned_ty, expr_span);
                        this.cfg.push_assign(
                            block,
                            source_info,
                            unsigned_place,
                            Rvalue::Cast(CastKind::IntToInt, Operand::Copy(discr), unsigned_ty),
                        );

                        let bool_ty = this.tcx.types.bool;
                        let range = scalar.valid_range(&this.tcx);
                        let merge_op =
                            if range.start <= range.end { BinOp::BitAnd } else { BinOp::BitOr };

                        let mut comparer = |range: u128, bin_op: BinOp| -> Place<'tcx> {
                            // We can use `ty::TypingEnv::fully_monomorphized()` here
                            // as we only need it to compute the layout of a primitive.
                            let range_val = Const::from_bits(
                                this.tcx,
                                range,
                                ty::TypingEnv::fully_monomorphized(),
                                unsigned_ty,
                            );
                            let lit_op = this.literal_operand(expr.span, range_val);
                            let is_bin_op = this.temp(bool_ty, expr_span);
                            this.cfg.push_assign(
                                block,
                                source_info,
                                is_bin_op,
                                Rvalue::BinaryOp(
                                    bin_op,
                                    Box::new((Operand::Copy(unsigned_place), lit_op)),
                                ),
                            );
                            is_bin_op
                        };
                        let assert_place = if range.start == 0 {
                            comparer(range.end, BinOp::Le)
                        } else {
                            let start_place = comparer(range.start, BinOp::Ge);
                            let end_place = comparer(range.end, BinOp::Le);
                            let merge_place = this.temp(bool_ty, expr_span);
                            this.cfg.push_assign(
                                block,
                                source_info,
                                merge_place,
                                Rvalue::BinaryOp(
                                    merge_op,
                                    Box::new((
                                        Operand::Move(start_place),
                                        Operand::Move(end_place),
                                    )),
                                ),
                            );
                            merge_place
                        };
                        this.cfg.push(
                            block,
                            Statement::new(
                                source_info,
                                StatementKind::Intrinsic(Box::new(NonDivergingIntrinsic::Assume(
                                    Operand::Move(assert_place),
                                ))),
                            ),
                        );
                    }

                    (op, ty)
                } else {
                    let ty = source_expr.ty;
                    let source = unpack!(
                        block = this.as_operand(
                            block,
                            scope,
                            source,
                            LocalInfo::Boring,
                            NeedsTemporary::No
                        )
                    );
                    (source, ty)
                };
                let from_ty = CastTy::from_ty(ty);
                let cast_ty = CastTy::from_ty(expr.ty);
                debug!("ExprKind::Cast from_ty={from_ty:?}, cast_ty={:?}/{cast_ty:?}", expr.ty);
                let cast_kind = mir_cast_kind(ty, expr.ty);
                block.and(Rvalue::Cast(cast_kind, source, expr.ty))
            }
            ExprKind::PointerCoercion { cast, source, is_from_as_cast } => {
                let source = unpack!(
                    block = this.as_operand(
                        block,
                        scope,
                        source,
                        LocalInfo::Boring,
                        NeedsTemporary::No
                    )
                );
                let origin =
                    if is_from_as_cast { CoercionSource::AsCast } else { CoercionSource::Implicit };
                block.and(Rvalue::Cast(CastKind::PointerCoercion(cast, origin), source, expr.ty))
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
                let fields: IndexVec<FieldIdx, _> = fields
                    .into_iter()
                    .copied()
                    .map(|f| {
                        unpack!(
                            block = this.as_operand(
                                block,
                                scope,
                                f,
                                LocalInfo::Boring,
                                NeedsTemporary::Maybe
                            )
                        )
                    })
                    .collect();

                block.and(Rvalue::Aggregate(Box::new(AggregateKind::Array(el_ty)), fields))
            }
            ExprKind::Tuple { ref fields } => {
                // see (*) above
                // first process the set of fields
                let fields: IndexVec<FieldIdx, _> = fields
                    .into_iter()
                    .copied()
                    .map(|f| {
                        unpack!(
                            block = this.as_operand(
                                block,
                                scope,
                                f,
                                LocalInfo::Boring,
                                NeedsTemporary::Maybe
                            )
                        )
                    })
                    .collect();

                block.and(Rvalue::Aggregate(Box::new(AggregateKind::Tuple), fields))
            }
            ExprKind::Closure(box ClosureExpr {
                closure_id,
                args,
                ref upvars,
                ref fake_reads,
                movability: _,
            }) => {
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
                    let place_builder = unpack!(block = this.as_place_builder(block, *thir_place));

                    if let Some(mir_place) = place_builder.try_to_place(this) {
                        this.cfg.push_fake_read(
                            block,
                            this.source_info(this.tcx.hir_span(*hir_id)),
                            *cause,
                            mir_place,
                        );
                    }
                }

                // see (*) above
                let operands: IndexVec<FieldIdx, _> = upvars
                    .into_iter()
                    .copied()
                    .map(|upvar| {
                        let upvar_expr = &this.thir[upvar];
                        match Category::of(&upvar_expr.kind) {
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
                                match upvar_expr.kind {
                                    ExprKind::Borrow {
                                        borrow_kind:
                                            BorrowKind::Mut { kind: MutBorrowKind::Default },
                                        arg,
                                    } => unpack!(
                                        block = this.limit_capture_mutability(
                                            upvar_expr.span,
                                            upvar_expr.ty,
                                            scope.temp_lifetime,
                                            block,
                                            arg,
                                        )
                                    ),
                                    _ => {
                                        unpack!(
                                            block = this.as_operand(
                                                block,
                                                scope,
                                                upvar,
                                                LocalInfo::Boring,
                                                NeedsTemporary::Maybe
                                            )
                                        )
                                    }
                                }
                            }
                        }
                    })
                    .collect();

                let result = match args {
                    UpvarArgs::Coroutine(args) => {
                        Box::new(AggregateKind::Coroutine(closure_id.to_def_id(), args))
                    }
                    UpvarArgs::Closure(args) => {
                        Box::new(AggregateKind::Closure(closure_id.to_def_id(), args))
                    }
                    UpvarArgs::CoroutineClosure(args) => {
                        Box::new(AggregateKind::CoroutineClosure(closure_id.to_def_id(), args))
                    }
                };
                block.and(Rvalue::Aggregate(result, operands))
            }
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                block = this.stmt_expr(block, expr_id, None).into_block();
                block.and(Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
                    span: expr_span,
                    user_ty: None,
                    const_: Const::zero_sized(this.tcx.types.unit),
                }))))
            }

            ExprKind::OffsetOf { container, fields } => {
                block.and(Rvalue::NullaryOp(NullOp::OffsetOf(fields), container))
            }

            ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::StaticRef { .. } => {
                let constant = this.as_constant(expr);
                block.and(Rvalue::Use(Operand::Constant(Box::new(constant))))
            }

            ExprKind::WrapUnsafeBinder { source } => {
                let source = unpack!(
                    block = this.as_operand(
                        block,
                        scope,
                        source,
                        LocalInfo::Boring,
                        NeedsTemporary::Maybe
                    )
                );
                block.and(Rvalue::WrapUnsafeBinder(source, expr.ty))
            }

            ExprKind::Yield { .. }
            | ExprKind::Block { .. }
            | ExprKind::Match { .. }
            | ExprKind::If { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::Use { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::RawBorrow { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Loop { .. }
            | ExprKind::LoopMatch { .. }
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
            | ExprKind::ConstContinue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Become { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::PlaceUnwrapUnsafeBinder { .. }
            | ExprKind::ValueUnwrapUnsafeBinder { .. } => {
                // these do not have corresponding `Rvalue` variants,
                // so make an operand and then return that
                debug_assert!(!matches!(
                    Category::of(&expr.kind),
                    Some(Category::Rvalue(RvalueFunc::AsRvalue) | Category::Constant)
                ));
                let operand = unpack!(
                    block = this.as_operand(
                        block,
                        scope,
                        expr_id,
                        LocalInfo::Boring,
                        NeedsTemporary::No,
                    )
                );
                block.and(Rvalue::Use(operand))
            }

            ExprKind::ByUse { expr, span: _ } => {
                let operand = unpack!(
                    block =
                        this.as_operand(block, scope, expr, LocalInfo::Boring, NeedsTemporary::No)
                );
                block.and(Rvalue::Use(operand))
            }
        }
    }

    pub(crate) fn build_binary_op(
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
        let rvalue = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul if self.check_overflow && ty.is_integral() => {
                let result_tup = Ty::new_tup(self.tcx, &[ty, bool_ty]);
                let result_value = self.temp(result_tup, span);

                let op_with_overflow = op.wrapping_to_overflowing().unwrap();

                self.cfg.push_assign(
                    block,
                    source_info,
                    result_value,
                    Rvalue::BinaryOp(op_with_overflow, Box::new((lhs.to_copy(), rhs.to_copy()))),
                );
                let val_fld = FieldIdx::ZERO;
                let of_fld = FieldIdx::new(1);

                let tcx = self.tcx;
                let val = tcx.mk_place_field(result_value, val_fld, ty);
                let of = tcx.mk_place_field(result_value, of_fld, bool_ty);

                let err = AssertKind::Overflow(op, lhs, rhs);
                block = self.assert(block, Operand::Move(of), false, err, span);

                Rvalue::Use(Operand::Move(val))
            }
            BinOp::Shl | BinOp::Shr if self.check_overflow && ty.is_integral() => {
                // For an unsigned RHS, the shift is in-range for `rhs < bits`.
                // For a signed RHS, `IntToInt` cast to the equivalent unsigned
                // type and do that same comparison.
                // A negative value will be *at least* 128 after the cast (that's i8::MIN),
                // and 128 is an overflowing shift amount for all our currently existing types,
                // so this cast can never make us miss an overflow.
                let (lhs_size, _) = ty.int_size_and_signed(self.tcx);
                assert!(lhs_size.bits() <= 128);
                let rhs_ty = rhs.ty(&self.local_decls, self.tcx);
                let (rhs_size, _) = rhs_ty.int_size_and_signed(self.tcx);

                let (unsigned_rhs, unsigned_ty) = match rhs_ty.kind() {
                    ty::Uint(_) => (rhs.to_copy(), rhs_ty),
                    ty::Int(int_width) => {
                        let uint_ty = Ty::new_uint(self.tcx, int_width.to_unsigned());
                        let rhs_temp = self.temp(uint_ty, span);
                        self.cfg.push_assign(
                            block,
                            source_info,
                            rhs_temp,
                            Rvalue::Cast(CastKind::IntToInt, rhs.to_copy(), uint_ty),
                        );
                        (Operand::Move(rhs_temp), uint_ty)
                    }
                    _ => unreachable!("only integers are shiftable"),
                };

                // This can't overflow because the largest shiftable types are 128-bit,
                // which fits in `u8`, the smallest possible `unsigned_ty`.
                let lhs_bits = Operand::const_from_scalar(
                    self.tcx,
                    unsigned_ty,
                    Scalar::from_uint(lhs_size.bits(), rhs_size),
                    span,
                );

                let inbounds = self.temp(bool_ty, span);
                self.cfg.push_assign(
                    block,
                    source_info,
                    inbounds,
                    Rvalue::BinaryOp(BinOp::Lt, Box::new((unsigned_rhs, lhs_bits))),
                );

                let overflow_err = AssertKind::Overflow(op, lhs.to_copy(), rhs.to_copy());
                block = self.assert(block, Operand::Move(inbounds), true, overflow_err, span);
                Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
            }
            BinOp::Div | BinOp::Rem if ty.is_integral() => {
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

                Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
            }
            _ => Rvalue::BinaryOp(op, Box::new((lhs, rhs))),
        };
        block.and(rvalue)
    }

    fn build_zero_repeat(
        &mut self,
        mut block: BasicBlock,
        value: ExprId,
        scope: TempLifetime,
        outer_source_info: SourceInfo,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let this = self;
        let value_expr = &this.thir[value];
        let elem_ty = value_expr.ty;
        if let Some(Category::Constant) = Category::of(&value_expr.kind) {
            // Repeating a const does nothing
        } else {
            // For a non-const, we may need to generate an appropriate `Drop`
            let value_operand = unpack!(
                block = this.as_operand(block, scope, value, LocalInfo::Boring, NeedsTemporary::No)
            );
            if let Operand::Move(to_drop) = value_operand {
                let success = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    outer_source_info,
                    TerminatorKind::Drop {
                        place: to_drop,
                        target: success,
                        unwind: UnwindAction::Continue,
                        replace: false,
                        drop: None,
                        async_fut: None,
                    },
                );
                this.diverge_from(block);
                block = success;
            }
            this.record_operands_moved(&[Spanned { node: value_operand, span: DUMMY_SP }]);
        }
        block.and(Rvalue::Aggregate(Box::new(AggregateKind::Array(elem_ty)), IndexVec::new()))
    }

    fn limit_capture_mutability(
        &mut self,
        upvar_span: Span,
        upvar_ty: Ty<'tcx>,
        temp_lifetime: Option<region::Scope>,
        mut block: BasicBlock,
        arg: ExprId,
    ) -> BlockAnd<Operand<'tcx>> {
        let this = self;

        let source_info = this.source_info(upvar_span);
        let temp = this.local_decls.push(LocalDecl::new(upvar_ty, upvar_span));

        this.cfg.push(block, Statement::new(source_info, StatementKind::StorageLive(temp)));

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
                let enclosing_upvars_resolved = arg_place_builder.to_place(this);

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
                            "Expected local to be Local(1), found {local:?}"
                        );
                        // Not in a closure
                        debug_assert!(
                            this.upvars.len() > upvar_index.index(),
                            "Unexpected capture place, upvars={:#?}, upvar_index={:?}",
                            this.upvars,
                            upvar_index
                        );
                        this.upvars[upvar_index.index()].mutability
                    }
                    _ => bug!("Unexpected capture place"),
                }
            }
        };

        let borrow_kind = match mutability {
            Mutability::Not => BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture },
            Mutability::Mut => BorrowKind::Mut { kind: MutBorrowKind::Default },
        };

        let arg_place = arg_place_builder.to_place(this);

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
        let typing_env = ty::TypingEnv::fully_monomorphized();
        let size = self.tcx.layout_of(typing_env.as_query_input(ty)).unwrap().size;
        let literal = Const::from_bits(self.tcx, size.unsigned_int_max(), typing_env, ty);

        self.literal_operand(span, literal)
    }

    // Helper to get the minimum value of the appropriate type
    fn minval_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        assert!(ty.is_signed());
        let typing_env = ty::TypingEnv::fully_monomorphized();
        let bits = self.tcx.layout_of(typing_env.as_query_input(ty)).unwrap().size.bits();
        let n = 1 << (bits - 1);
        let literal = Const::from_bits(self.tcx, n, typing_env, ty);

        self.literal_operand(span, literal)
    }
}
