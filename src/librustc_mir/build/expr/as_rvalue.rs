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

use rustc_data_structures::fnv::FnvHashMap;

use build::{BlockAnd, BlockAndExtension, Builder};
use build::expr::category::{Category, RvalueFunc};
use hair::*;
use rustc::mir::repr::*;

impl<'a,'tcx> Builder<'a,'tcx> {
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
                block.and(Rvalue::BinaryOp(op, lhs, rhs))
            }
            ExprKind::Unary { op, arg } => {
                let arg = unpack!(block = this.as_operand(block, arg));
                block.and(Rvalue::UnaryOp(op, arg))
            }
            ExprKind::Box { value, value_extents } => {
                let value = this.hir.mirror(value);
                let result = this.temp(expr.ty);
                // to start, malloc some memory of suitable type (thus far, uninitialized):
                this.cfg.push_assign(block, expr_span, &result, Rvalue::Box(value.ty));
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
                if source.ty == expr.ty {
                    this.expr_as_rvalue(block, source)
                } else {
                    let source = unpack!(block = this.as_operand(block, source));
                    block.and(Rvalue::Cast(CastKind::Misc, source, expr.ty))
                }
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
            ExprKind::Vec { fields } => {
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

                block.and(Rvalue::Aggregate(AggregateKind::Vec, fields))
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
                // first process the set of fields that were provided
                // (evaluating them in order given by user)
                let fields_map: FnvHashMap<_, _> =
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
                    field_names.iter().map(|n| fields_map[n].clone()).collect()
                };

                block.and(Rvalue::Aggregate(AggregateKind::Adt(adt_def, variant_index, substs),
                                            fields))
            }
            ExprKind::Literal { .. } |
            ExprKind::Block { .. } |
            ExprKind::Match { .. } |
            ExprKind::If { .. } |
            ExprKind::Loop { .. } |
            ExprKind::LogicalOp { .. } |
            ExprKind::Call { .. } |
            ExprKind::Field { .. } |
            ExprKind::Deref { .. } |
            ExprKind::Index { .. } |
            ExprKind::VarRef { .. } |
            ExprKind::SelfRef |
            ExprKind::Assign { .. } |
            ExprKind::AssignOp { .. } |
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
}
