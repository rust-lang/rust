// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See docs in build/expr/mod.rs

use std;

use rustc_const_math::{ConstMathErr, Op};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use rustc_i128::i128;

use build::{BlockAnd, BlockAndExtension, Builder};
use build::expr::category::{Category, RvalueFunc};
use hair::*;
use rustc_const_math::{ConstInt, ConstIsize};
use rustc::middle::const_val::ConstVal;
use rustc::ty;
use rustc::mir::*;
use syntax::ast;
use syntax_pos::Span;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Compile `expr`, yielding an rvalue.
    pub fn as_rvalue<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Rvalue<'tcx>>
        where M: Mirror<'tcx, Output = Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_rvalue(block, expr)
    }

    fn expr_as_rvalue(&mut self,
                      mut block: BasicBlock,
                      expr: Expr<'tcx>)
                      -> BlockAnd<Rvalue<'tcx>> {
        debug!("expr_as_rvalue(block={:?}, expr={:?})", block, expr);

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        match expr.kind {
            ExprKind::Scope { extent, value } => {
                this.in_scope(extent, block, |this| this.as_rvalue(block, value))
            }
            ExprKind::InlineAsm { asm, outputs, inputs } => {
                let outputs = outputs.into_iter().map(|output| {
                    unpack!(block = this.as_lvalue(block, output))
                }).collect();

                let inputs = inputs.into_iter().map(|input| {
                    unpack!(block = this.as_operand(block, input))
                }).collect();

                block.and(Rvalue::InlineAsm {
                    asm: asm.clone(),
                    outputs: outputs,
                    inputs: inputs
                })
            }
            ExprKind::Repeat { value, count } => {
                let value_operand = unpack!(block = this.as_operand(block, value));
                block.and(Rvalue::Repeat(value_operand, count))
            }
            ExprKind::Borrow { region, borrow_kind, arg } => {
                let arg_lvalue = unpack!(block = this.as_lvalue(block, arg));
                block.and(Rvalue::Ref(region, borrow_kind, arg_lvalue))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs = unpack!(block = this.as_operand(block, lhs));
                let rhs = unpack!(block = this.as_operand(block, rhs));
                this.build_binary_op(block, op, expr_span, expr.ty,
                                     lhs, rhs)
            }
            ExprKind::Unary { op, arg } => {
                let arg = unpack!(block = this.as_operand(block, arg));
                // Check for -MIN on signed integers
                if this.hir.check_overflow() && op == UnOp::Neg && expr.ty.is_signed() {
                    let bool_ty = this.hir.bool_ty();

                    let minval = this.minval_literal(expr_span, expr.ty);
                    let is_min = this.temp(bool_ty);

                    this.cfg.push_assign(block, source_info, &is_min,
                                         Rvalue::BinaryOp(BinOp::Eq, arg.clone(), minval));

                    let err = ConstMathErr::Overflow(Op::Neg);
                    block = this.assert(block, Operand::Consume(is_min), false,
                                        AssertMessage::Math(err), expr_span);
                }
                block.and(Rvalue::UnaryOp(op, arg))
            }
            ExprKind::Box { value, value_extents } => {
                let value = this.hir.mirror(value);
                let result = this.temp(expr.ty);
                // to start, malloc some memory of suitable type (thus far, uninitialized):
                this.cfg.push_assign(block, source_info, &result, Rvalue::Box(value.ty));
                this.in_scope(value_extents, block, |this| {
                    // schedule a shallow free of that memory, lest we unwind:
                    this.schedule_box_free(expr_span, value_extents, &result, value.ty);
                    // initialize the box contents:
                    unpack!(block = this.into(&result.clone().deref(), block, value));
                    block.and(Rvalue::Use(Operand::Consume(result)))
                })
            }
            ExprKind::Cast { source } => {
                let source = this.hir.mirror(source);

                let source = unpack!(block = this.as_operand(block, source));
                block.and(Rvalue::Cast(CastKind::Misc, source, expr.ty))
            }
            ExprKind::Use { source } => {
                let source = unpack!(block = this.as_operand(block, source));
                block.and(Rvalue::Use(source))
            }
            ExprKind::ReifyFnPointer { source } => {
                let source = unpack!(block = this.as_operand(block, source));
                block.and(Rvalue::Cast(CastKind::ReifyFnPointer, source, expr.ty))
            }
            ExprKind::UnsafeFnPointer { source } => {
                let source = unpack!(block = this.as_operand(block, source));
                block.and(Rvalue::Cast(CastKind::UnsafeFnPointer, source, expr.ty))
            }
            ExprKind::Unsize { source } => {
                let source = unpack!(block = this.as_operand(block, source));
                block.and(Rvalue::Cast(CastKind::Unsize, source, expr.ty))
            }
            ExprKind::Array { fields } => {
                // (*) We would (maybe) be closer to trans if we
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
                let fields: Vec<_> =
                    fields.into_iter()
                          .map(|f| unpack!(block = this.as_operand(block, f)))
                          .collect();

                block.and(Rvalue::Aggregate(AggregateKind::Array, fields))
            }
            ExprKind::Tuple { fields } => { // see (*) above
                // first process the set of fields
                let fields: Vec<_> =
                    fields.into_iter()
                          .map(|f| unpack!(block = this.as_operand(block, f)))
                          .collect();

                block.and(Rvalue::Aggregate(AggregateKind::Tuple, fields))
            }
            ExprKind::Closure { closure_id, substs, upvars } => { // see (*) above
                let upvars =
                    upvars.into_iter()
                          .map(|upvar| unpack!(block = this.as_operand(block, upvar)))
                          .collect();
                block.and(Rvalue::Aggregate(AggregateKind::Closure(closure_id, substs), upvars))
            }
            ExprKind::Adt {
                adt_def, variant_index, substs, fields, base
            } => { // see (*) above
                let is_union = adt_def.is_union();
                let active_field_index = if is_union { Some(fields[0].name.index()) } else { None };

                // first process the set of fields that were provided
                // (evaluating them in order given by user)
                let fields_map: FxHashMap<_, _> =
                    fields.into_iter()
                          .map(|f| (f.name, unpack!(block = this.as_operand(block, f.expr))))
                          .collect();

                let field_names = this.hir.all_fields(adt_def, variant_index);

                let fields = if let Some(FruInfo { base, field_types }) = base {
                    let base = unpack!(block = this.as_lvalue(block, base));

                    // MIR does not natively support FRU, so for each
                    // base-supplied field, generate an operand that
                    // reads it from the base.
                    field_names.into_iter()
                        .zip(field_types.into_iter())
                        .map(|(n, ty)| match fields_map.get(&n) {
                            Some(v) => v.clone(),
                            None => Operand::Consume(base.clone().field(n, ty))
                        })
                        .collect()
                } else {
                    field_names.iter().filter_map(|n| fields_map.get(n).cloned()).collect()
                };

                let adt = AggregateKind::Adt(adt_def, variant_index, substs, active_field_index);
                block.and(Rvalue::Aggregate(adt, fields))
            }
            ExprKind::Assign { .. } |
            ExprKind::AssignOp { .. } => {
                block = unpack!(this.stmt_expr(block, expr));
                block.and(this.unit_rvalue())
            }
            ExprKind::Literal { .. } |
            ExprKind::Block { .. } |
            ExprKind::Match { .. } |
            ExprKind::If { .. } |
            ExprKind::NeverToAny { .. } |
            ExprKind::Loop { .. } |
            ExprKind::LogicalOp { .. } |
            ExprKind::Call { .. } |
            ExprKind::Field { .. } |
            ExprKind::Deref { .. } |
            ExprKind::Index { .. } |
            ExprKind::VarRef { .. } |
            ExprKind::SelfRef |
            ExprKind::Break { .. } |
            ExprKind::Continue { .. } |
            ExprKind::Return { .. } |
            ExprKind::StaticRef { .. } => {
                // these do not have corresponding `Rvalue` variants,
                // so make an operand and then return that
                debug_assert!(match Category::of(&expr.kind) {
                    Some(Category::Rvalue(RvalueFunc::AsRvalue)) => false,
                    _ => true,
                });
                let operand = unpack!(block = this.as_operand(block, expr));
                block.and(Rvalue::Use(operand))
            }
        }
    }

    pub fn build_binary_op(&mut self, mut block: BasicBlock,
                           op: BinOp, span: Span, ty: ty::Ty<'tcx>,
                           lhs: Operand<'tcx>, rhs: Operand<'tcx>) -> BlockAnd<Rvalue<'tcx>> {
        let source_info = self.source_info(span);
        let bool_ty = self.hir.bool_ty();
        if self.hir.check_overflow() && op.is_checkable() && ty.is_integral() {
            let result_tup = self.hir.tcx().intern_tup(&[ty, bool_ty]);
            let result_value = self.temp(result_tup);

            self.cfg.push_assign(block, source_info,
                                 &result_value, Rvalue::CheckedBinaryOp(op,
                                                                        lhs,
                                                                        rhs));
            let val_fld = Field::new(0);
            let of_fld = Field::new(1);

            let val = result_value.clone().field(val_fld, ty);
            let of = result_value.field(of_fld, bool_ty);

            let err = ConstMathErr::Overflow(match op {
                BinOp::Add => Op::Add,
                BinOp::Sub => Op::Sub,
                BinOp::Mul => Op::Mul,
                BinOp::Shl => Op::Shl,
                BinOp::Shr => Op::Shr,
                _ => {
                    bug!("MIR build_binary_op: {:?} is not checkable", op)
                }
            });

            block = self.assert(block, Operand::Consume(of), false,
                                AssertMessage::Math(err), span);

            block.and(Rvalue::Use(Operand::Consume(val)))
        } else {
            if ty.is_integral() && (op == BinOp::Div || op == BinOp::Rem) {
                // Checking division and remainder is more complex, since we 1. always check
                // and 2. there are two possible failure cases, divide-by-zero and overflow.

                let (zero_err, overflow_err) = if op == BinOp::Div {
                    (ConstMathErr::DivisionByZero,
                     ConstMathErr::Overflow(Op::Div))
                } else {
                    (ConstMathErr::RemainderByZero,
                     ConstMathErr::Overflow(Op::Rem))
                };

                // Check for / 0
                let is_zero = self.temp(bool_ty);
                let zero = self.zero_literal(span, ty);
                self.cfg.push_assign(block, source_info, &is_zero,
                                     Rvalue::BinaryOp(BinOp::Eq, rhs.clone(), zero));

                block = self.assert(block, Operand::Consume(is_zero), false,
                                    AssertMessage::Math(zero_err), span);

                // We only need to check for the overflow in one case:
                // MIN / -1, and only for signed values.
                if ty.is_signed() {
                    let neg_1 = self.neg_1_literal(span, ty);
                    let min = self.minval_literal(span, ty);

                    let is_neg_1 = self.temp(bool_ty);
                    let is_min   = self.temp(bool_ty);
                    let of       = self.temp(bool_ty);

                    // this does (rhs == -1) & (lhs == MIN). It could short-circuit instead

                    self.cfg.push_assign(block, source_info, &is_neg_1,
                                         Rvalue::BinaryOp(BinOp::Eq, rhs.clone(), neg_1));
                    self.cfg.push_assign(block, source_info, &is_min,
                                         Rvalue::BinaryOp(BinOp::Eq, lhs.clone(), min));

                    let is_neg_1 = Operand::Consume(is_neg_1);
                    let is_min = Operand::Consume(is_min);
                    self.cfg.push_assign(block, source_info, &of,
                                         Rvalue::BinaryOp(BinOp::BitAnd, is_neg_1, is_min));

                    block = self.assert(block, Operand::Consume(of), false,
                                        AssertMessage::Math(overflow_err), span);
                }
            }

            block.and(Rvalue::BinaryOp(op, lhs, rhs))
        }
    }

    // Helper to get a `-1` value of the appropriate type
    fn neg_1_literal(&mut self, span: Span, ty: ty::Ty<'tcx>) -> Operand<'tcx> {
        let literal = match ty.sty {
            ty::TyInt(ity) => {
                let val = match ity {
                    ast::IntTy::I8  => ConstInt::I8(-1),
                    ast::IntTy::I16 => ConstInt::I16(-1),
                    ast::IntTy::I32 => ConstInt::I32(-1),
                    ast::IntTy::I64 => ConstInt::I64(-1),
                    ast::IntTy::I128 => ConstInt::I128(-1),
                    ast::IntTy::Is => {
                        let int_ty = self.hir.tcx().sess.target.int_type;
                        let val = ConstIsize::new(-1, int_ty).unwrap();
                        ConstInt::Isize(val)
                    }
                };

                Literal::Value { value: ConstVal::Integral(val) }
            }
            _ => {
                span_bug!(span, "Invalid type for neg_1_literal: `{:?}`", ty)
            }
        };

        self.literal_operand(span, ty, literal)
    }

    // Helper to get the minimum value of the appropriate type
    fn minval_literal(&mut self, span: Span, ty: ty::Ty<'tcx>) -> Operand<'tcx> {
        let literal = match ty.sty {
            ty::TyInt(ity) => {
                let val = match ity {
                    ast::IntTy::I8  => ConstInt::I8(i8::min_value()),
                    ast::IntTy::I16 => ConstInt::I16(i16::min_value()),
                    ast::IntTy::I32 => ConstInt::I32(i32::min_value()),
                    ast::IntTy::I64 => ConstInt::I64(i64::min_value()),
                    ast::IntTy::I128 => ConstInt::I128(i128::min_value()),
                    ast::IntTy::Is => {
                        let int_ty = self.hir.tcx().sess.target.int_type;
                        let min = match int_ty {
                            ast::IntTy::I16 => std::i16::MIN as i64,
                            ast::IntTy::I32 => std::i32::MIN as i64,
                            ast::IntTy::I64 => std::i64::MIN,
                            _ => unreachable!()
                        };
                        let val = ConstIsize::new(min, int_ty).unwrap();
                        ConstInt::Isize(val)
                    }
                };

                Literal::Value { value: ConstVal::Integral(val) }
            }
            _ => {
                span_bug!(span, "Invalid type for minval_literal: `{:?}`", ty)
            }
        };

        self.literal_operand(span, ty, literal)
    }
}
