// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use middle::ty::Region;
use mir::repr::*;
use rustc_data_structures::tuple_slice::TupleSlice;
use syntax::codemap::Span;

pub trait Visitor<'tcx> {
    // Override these, and call `self.super_xxx` to revert back to the
    // default behavior.

    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        self.super_mir(mir);
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.super_basic_block_data(block, data);
    }

    fn visit_statement(&mut self, block: BasicBlock, statement: &Statement<'tcx>) {
        self.super_statement(block, statement);
    }

    fn visit_assign(&mut self, block: BasicBlock, lvalue: &Lvalue<'tcx>, rvalue: &Rvalue<'tcx>) {
        self.super_assign(block, lvalue, rvalue);
    }

    fn visit_terminator(&mut self, block: BasicBlock, terminator: &Terminator<'tcx>) {
        self.super_terminator(block, terminator);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        self.super_rvalue(rvalue);
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>) {
        self.super_operand(operand);
    }

    fn visit_lvalue(&mut self, lvalue: &Lvalue<'tcx>, context: LvalueContext) {
        self.super_lvalue(lvalue, context);
    }

    fn visit_branch(&mut self, source: BasicBlock, target: BasicBlock) {
        self.super_branch(source, target);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>) {
        self.super_constant(constant);
    }

    fn visit_literal(&mut self, literal: &Literal<'tcx>) {
        self.super_literal(literal);
    }

    fn visit_def_id(&mut self, def_id: DefId) {
        self.super_def_id(def_id);
    }

    fn visit_span(&mut self, span: Span) {
        self.super_span(span);
    }

    // The `super_xxx` methods comprise the default behavior and are
    // not meant to be overidden.

    fn super_mir(&mut self, mir: &Mir<'tcx>) {
        for block in mir.all_basic_blocks() {
            let data = mir.basic_block_data(block);
            self.visit_basic_block_data(block, data);
        }
    }

    fn super_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        for statement in &data.statements {
            self.visit_statement(block, statement);
        }
        self.visit_terminator(block, &data.terminator);
    }

    fn super_statement(&mut self, block: BasicBlock, statement: &Statement<'tcx>) {
        self.visit_span(statement.span);

        match statement.kind {
            StatementKind::Assign(ref lvalue, ref rvalue) => {
                self.visit_assign(block, lvalue, rvalue);
            }
            StatementKind::Drop(_, ref lvalue) => {
                self.visit_lvalue(lvalue, LvalueContext::Drop);
            }
        }
    }

    fn super_assign(&mut self, _block: BasicBlock, lvalue: &Lvalue<'tcx>, rvalue: &Rvalue<'tcx>) {
        self.visit_lvalue(lvalue, LvalueContext::Store);
        self.visit_rvalue(rvalue);
    }

    fn super_terminator(&mut self, block: BasicBlock, terminator: &Terminator<'tcx>) {
        match *terminator {
            Terminator::Goto { target } => {
                self.visit_branch(block, target);
            }

            Terminator::If { ref cond, ref targets } => {
                self.visit_operand(cond);
                for &target in targets.as_slice() {
                    self.visit_branch(block, target);
                }
            }

            Terminator::Switch { ref discr, adt_def: _, ref targets } => {
                self.visit_lvalue(discr, LvalueContext::Inspect);
                for &target in targets {
                    self.visit_branch(block, target);
                }
            }

            Terminator::SwitchInt { ref discr, switch_ty: _, values: _, ref targets } => {
                self.visit_lvalue(discr, LvalueContext::Inspect);
                for &target in targets {
                    self.visit_branch(block, target);
                }
            }

            Terminator::Diverge |
            Terminator::Resume |
            Terminator::Return => {
            }

            Terminator::Call { ref func, ref args, ref destination, ref targets } => {
                self.visit_lvalue(destination, LvalueContext::Store);
                self.visit_operand(func);
                for arg in args {
                    self.visit_operand(arg);
                }
                for &target in targets.as_slice() {
                    self.visit_branch(block, target);
                }
            }

            Terminator::DivergingCall { ref func, ref args, ref cleanup } => {
                self.visit_operand(func);
                for arg in args {
                    self.visit_operand(arg);
                }
                for &target in cleanup.as_ref() {
                    self.visit_branch(block, target);
                }
            }
        }
    }

    fn super_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(ref operand) => {
                self.visit_operand(operand);
            }

            Rvalue::Repeat(ref value, ref len) => {
                self.visit_operand(value);
                self.visit_constant(len);
            }

            Rvalue::Ref(r, bk, ref path) => {
                self.visit_lvalue(path, LvalueContext::Borrow {
                    region: r,
                    kind: bk
                });
            }

            Rvalue::Len(ref path) => {
                self.visit_lvalue(path, LvalueContext::Inspect);
            }

            Rvalue::Cast(_, ref operand, _) => {
                self.visit_operand(operand);
            }

            Rvalue::BinaryOp(_, ref lhs, ref rhs) => {
                self.visit_operand(lhs);
                self.visit_operand(rhs);
            }

            Rvalue::UnaryOp(_, ref op) => {
                self.visit_operand(op);
            }

            Rvalue::Box(_) => {
            }

            Rvalue::Aggregate(_, ref operands) => {
                for operand in operands {
                    self.visit_operand(operand);
                }
            }

            Rvalue::Slice { ref input, from_start, from_end } => {
                self.visit_lvalue(input, LvalueContext::Slice {
                    from_start: from_start,
                    from_end: from_end,
                });
            }

            Rvalue::InlineAsm(_) => {
            }
        }
    }

    fn super_operand(&mut self, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Consume(ref lvalue) => {
                self.visit_lvalue(lvalue, LvalueContext::Consume);
            }
            Operand::Constant(ref constant) => {
                self.visit_constant(constant);
            }
        }
    }

    fn super_lvalue(&mut self, lvalue: &Lvalue<'tcx>, _context: LvalueContext) {
        match *lvalue {
            Lvalue::Var(_) |
            Lvalue::Temp(_) |
            Lvalue::Arg(_) |
            Lvalue::Static(_) |
            Lvalue::ReturnPointer => {
            }
            Lvalue::Projection(ref proj) => {
                self.visit_lvalue(&proj.base, LvalueContext::Projection);
            }
        }
    }

    fn super_branch(&mut self, _source: BasicBlock, _target: BasicBlock) {
    }

    fn super_constant(&mut self, constant: &Constant<'tcx>) {
        self.visit_span(constant.span);
        self.visit_literal(&constant.literal);
    }

    fn super_literal(&mut self, literal: &Literal<'tcx>) {
        match *literal {
            Literal::Item { def_id, .. } => {
                self.visit_def_id(def_id);
            },
            Literal::Value { .. } => {
                // Nothing to do
            }
        }
    }

    fn super_def_id(&mut self, _def_id: DefId) {
    }

    fn super_span(&mut self, _span: Span) {
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LvalueContext {
    // Appears as LHS of an assignment or as dest of a call
    Store,

    // Being dropped
    Drop,

    // Being inspected in some way, like loading a len
    Inspect,

    // Being borrowed
    Borrow { region: Region, kind: BorrowKind },

    // Being sliced -- this should be same as being borrowed, probably
    Slice { from_start: usize, from_end: usize },

    // Used as base for another lvalue, e.g. `x` in `x.y`
    Projection,

    // Consumed as part of an operand
    Consume,
}

pub trait MutVisitor<'tcx> {
    // Override these, and call `self.super_xxx` to revert back to the
    // default behavior.

    fn visit_mir(&mut self, mir: &mut Mir<'tcx>) {
        self.super_mir(mir);
    }

    fn visit_basic_block_data(&mut self,
                              block: BasicBlock,
                              data: &mut BasicBlockData<'tcx>) {
        self.super_basic_block_data(block, data);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>) {
        self.super_statement(block, statement);
    }

    fn visit_assign(&mut self,
                    block: BasicBlock,
                    lvalue: &mut Lvalue<'tcx>,
                    rvalue: &mut Rvalue<'tcx>) {
        self.super_assign(block, lvalue, rvalue);
    }

    fn visit_terminator(&mut self,
                        block: BasicBlock,
                        terminator: &mut Terminator<'tcx>) {
        self.super_terminator(block, terminator);
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>) {
        self.super_rvalue(rvalue);
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>) {
        self.super_operand(operand);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    context: LvalueContext) {
        self.super_lvalue(lvalue, context);
    }

    fn visit_branch(&mut self, source: BasicBlock, target: BasicBlock) {
        self.super_branch(source, target);
    }

    fn visit_constant(&mut self, constant: &mut Constant<'tcx>) {
        self.super_constant(constant);
    }

    fn visit_literal(&mut self, literal: &mut Literal<'tcx>) {
        self.super_literal(literal);
    }

    fn visit_def_id(&mut self, def_id: &mut DefId) {
        self.super_def_id(def_id);
    }

    fn visit_span(&mut self, span: &mut Span) {
        self.super_span(span);
    }

    // The `super_xxx` methods comprise the default behavior and are
    // not meant to be overidden.

    fn super_mir(&mut self, mir: &mut Mir<'tcx>) {
        for block in mir.all_basic_blocks() {
            let data = mir.basic_block_data_mut(block);
            self.visit_basic_block_data(block, data);
        }
    }

    fn super_basic_block_data(&mut self,
                              block: BasicBlock,
                              data: &mut BasicBlockData<'tcx>) {
        for statement in &mut data.statements {
            self.visit_statement(block, statement);
        }
        self.visit_terminator(block, &mut data.terminator);
    }

    fn super_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>) {
        self.visit_span(&mut statement.span);

        match statement.kind {
            StatementKind::Assign(ref mut lvalue, ref mut rvalue) => {
                self.visit_assign(block, lvalue, rvalue);
            }
            StatementKind::Drop(_, ref mut lvalue) => {
                self.visit_lvalue(lvalue, LvalueContext::Drop);
            }
        }
    }

    fn super_assign(&mut self,
                    _block: BasicBlock,
                    lvalue: &mut Lvalue<'tcx>,
                    rvalue: &mut Rvalue<'tcx>) {
        self.visit_lvalue(lvalue, LvalueContext::Store);
        self.visit_rvalue(rvalue);
    }

    fn super_terminator(&mut self,
                        block: BasicBlock,
                        terminator: &mut Terminator<'tcx>) {
        match *terminator {
            Terminator::Goto { target } => {
                self.visit_branch(block, target);
            }

            Terminator::If { ref mut cond, ref mut targets } => {
                self.visit_operand(cond);
                for &target in targets.as_slice() {
                    self.visit_branch(block, target);
                }
            }

            Terminator::Switch { ref mut discr, adt_def: _, ref targets } => {
                self.visit_lvalue(discr, LvalueContext::Inspect);
                for &target in targets {
                    self.visit_branch(block, target);
                }
            }

            Terminator::SwitchInt { ref mut discr, switch_ty: _, values: _, ref targets } => {
                self.visit_lvalue(discr, LvalueContext::Inspect);
                for &target in targets {
                    self.visit_branch(block, target);
                }
            }

            Terminator::Diverge |
            Terminator::Resume |
            Terminator::Return => {
            }

            Terminator::Call { ref mut func,
                               ref mut args,
                               ref mut destination,
                               ref mut targets } => {
                self.visit_lvalue(destination, LvalueContext::Store);
                self.visit_operand(func);
                for arg in args {
                    self.visit_operand(arg);
                }
                for &target in targets.as_slice() {
                    self.visit_branch(block, target);
                }
            }

            Terminator::DivergingCall { ref mut func, ref mut args, ref mut cleanup } => {
                self.visit_operand(func);
                for arg in args {
                    self.visit_operand(arg);
                }
                for &target in cleanup.as_ref() {
                    self.visit_branch(block, target);
                }
            }
        }
    }

    fn super_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(ref mut operand) => {
                self.visit_operand(operand);
            }

            Rvalue::Repeat(ref mut value, ref mut len) => {
                self.visit_operand(value);
                self.visit_constant(len);
            }

            Rvalue::Ref(r, bk, ref mut path) => {
                self.visit_lvalue(path, LvalueContext::Borrow {
                    region: r,
                    kind: bk
                });
            }

            Rvalue::Len(ref mut path) => {
                self.visit_lvalue(path, LvalueContext::Inspect);
            }

            Rvalue::Cast(_, ref mut operand, _) => {
                self.visit_operand(operand);
            }

            Rvalue::BinaryOp(_, ref mut lhs, ref mut rhs) => {
                self.visit_operand(lhs);
                self.visit_operand(rhs);
            }

            Rvalue::UnaryOp(_, ref mut op) => {
                self.visit_operand(op);
            }

            Rvalue::Box(_) => {
            }

            Rvalue::Aggregate(ref mut kind, ref mut operands) => {
                match *kind {
                    AggregateKind::Closure(ref mut def_id, _) => {
                        self.visit_def_id(def_id);
                    }
                    _ => { /* nothing to do */ }
                }

                for operand in &mut operands[..] {
                    self.visit_operand(operand);
                }
            }

            Rvalue::Slice { ref mut input, from_start, from_end } => {
                self.visit_lvalue(input, LvalueContext::Slice {
                    from_start: from_start,
                    from_end: from_end,
                });
            }

            Rvalue::InlineAsm(_) => {
            }
        }
    }

    fn super_operand(&mut self, operand: &mut Operand<'tcx>) {
        match *operand {
            Operand::Consume(ref mut lvalue) => {
                self.visit_lvalue(lvalue, LvalueContext::Consume);
            }
            Operand::Constant(ref mut constant) => {
                self.visit_constant(constant);
            }
        }
    }

    fn super_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    _context: LvalueContext) {
        match *lvalue {
            Lvalue::Var(_) |
            Lvalue::Temp(_) |
            Lvalue::Arg(_) |
            Lvalue::ReturnPointer => {
            }
            Lvalue::Static(ref mut def_id) => {
                self.visit_def_id(def_id);
            }
            Lvalue::Projection(ref mut proj) => {
                self.visit_lvalue(&mut proj.base, LvalueContext::Projection);
            }
        }
    }

    fn super_branch(&mut self, _source: BasicBlock, _target: BasicBlock) {
    }

    fn super_constant(&mut self, constant: &mut Constant<'tcx>) {
        self.visit_span(&mut constant.span);
        self.visit_literal(&mut constant.literal);
    }

    fn super_literal(&mut self, literal: &mut Literal<'tcx>) {
        match *literal {
            Literal::Item { ref mut def_id, .. } => {
                self.visit_def_id(def_id);
            },
            Literal::Value { .. } => {
                // Nothing to do
            }
        }
    }

    fn super_def_id(&mut self, _def_id: &mut DefId) {
    }

    fn super_span(&mut self, _span: &mut Span) {
    }
}
