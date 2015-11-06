// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::ty::Region;
use repr::*;

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
            Terminator::Goto { target } |
            Terminator::Panic { target } => {
                self.visit_branch(block, target);
            }

            Terminator::If { ref cond, ref targets } => {
                self.visit_operand(cond);
                for &target in &targets[..] {
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
            Terminator::Return => {
            }

            Terminator::Call { ref data, ref targets } => {
                self.visit_lvalue(&data.destination, LvalueContext::Store);
                self.visit_operand(&data.func);
                for arg in &data.args {
                    self.visit_operand(arg);
                }
                for &target in &targets[..] {
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
                self.visit_operand(len);
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

    fn super_constant(&mut self, _constant: &Constant<'tcx>) {
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
