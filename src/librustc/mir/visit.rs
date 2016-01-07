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

macro_rules! make_mir_visitor {
    ($visitor_trait_name:ident, $($mutability:ident)*) => {
        pub trait $visitor_trait_name<'tcx> {
            // Override these, and call `self.super_xxx` to revert back to the
            // default behavior.

            fn visit_mir(&mut self, mir: & $($mutability)* Mir<'tcx>) {
                self.super_mir(mir);
            }

            fn visit_basic_block_data(&mut self,
                                      block: BasicBlock,
                                      data: & $($mutability)* BasicBlockData<'tcx>) {
                self.super_basic_block_data(block, data);
            }

            fn visit_statement(&mut self,
                               block: BasicBlock,
                               statement: & $($mutability)* Statement<'tcx>) {
                self.super_statement(block, statement);
            }

            fn visit_assign(&mut self,
                            block: BasicBlock,
                            lvalue: & $($mutability)* Lvalue<'tcx>,
                            rvalue: & $($mutability)* Rvalue<'tcx>) {
                self.super_assign(block, lvalue, rvalue);
            }

            fn visit_terminator(&mut self,
                                block: BasicBlock,
                                terminator: & $($mutability)* Terminator<'tcx>) {
                self.super_terminator(block, terminator);
            }

            fn visit_rvalue(&mut self,
                            rvalue: & $($mutability)* Rvalue<'tcx>) {
                self.super_rvalue(rvalue);
            }

            fn visit_operand(&mut self,
                             operand: & $($mutability)* Operand<'tcx>) {
                self.super_operand(operand);
            }

            fn visit_lvalue(&mut self,
                            lvalue: & $($mutability)* Lvalue<'tcx>,
                            context: LvalueContext) {
                self.super_lvalue(lvalue, context);
            }

            fn visit_branch(&mut self,
                            source: BasicBlock,
                            target: BasicBlock) {
                self.super_branch(source, target);
            }

            fn visit_constant(&mut self,
                              constant: & $($mutability)* Constant<'tcx>) {
                self.super_constant(constant);
            }

            fn visit_literal(&mut self,
                             literal: & $($mutability)* Literal<'tcx>) {
                self.super_literal(literal);
            }

            fn visit_def_id(&mut self,
                            def_id: & $($mutability)* DefId) {
                self.super_def_id(def_id);
            }

            fn visit_span(&mut self,
                          span: & $($mutability)* Span) {
                self.super_span(span);
            }

            // The `super_xxx` methods comprise the default behavior and are
            // not meant to be overidden.

            fn super_mir(&mut self,
                         mir: & $($mutability)* Mir<'tcx>) {
                for block in mir.all_basic_blocks() {
                    let data = & $($mutability)* mir[block];
                    self.visit_basic_block_data(block, data);
                }
            }

            fn super_basic_block_data(&mut self,
                                      block: BasicBlock,
                                      data: & $($mutability)* BasicBlockData<'tcx>) {
                for statement in & $($mutability)* data.statements {
                    self.visit_statement(block, statement);
                }

                if let Some(ref $($mutability)* terminator) = data.terminator {
                    self.visit_terminator(block, terminator);
                }
            }

            fn super_statement(&mut self,
                               block: BasicBlock,
                               statement: & $($mutability)* Statement<'tcx>) {
                self.visit_span(& $($mutability)* statement.span);

                match statement.kind {
                    StatementKind::Assign(ref $($mutability)* lvalue,
                                          ref $($mutability)* rvalue) => {
                        self.visit_assign(block, lvalue, rvalue);
                    }
                    StatementKind::Drop(_, ref $($mutability)* lvalue) => {
                        self.visit_lvalue(lvalue, LvalueContext::Drop);
                    }
                }
            }

            fn super_assign(&mut self,
                            _block: BasicBlock,
                            lvalue: &$($mutability)* Lvalue<'tcx>,
                            rvalue: &$($mutability)* Rvalue<'tcx>) {
                self.visit_lvalue(lvalue, LvalueContext::Store);
                self.visit_rvalue(rvalue);
            }

            fn super_terminator(&mut self,
                                block: BasicBlock,
                                terminator: &$($mutability)* Terminator<'tcx>) {
                match *terminator {
                    Terminator::Goto { target } => {
                        self.visit_branch(block, target);
                    }

                    Terminator::If { ref $($mutability)* cond,
                                     ref $($mutability)* targets } => {
                        self.visit_operand(cond);
                        for &target in targets.as_slice() {
                            self.visit_branch(block, target);
                        }
                    }

                    Terminator::Switch { ref $($mutability)* discr,
                                         adt_def: _,
                                         ref targets } => {
                        self.visit_lvalue(discr, LvalueContext::Inspect);
                        for &target in targets {
                            self.visit_branch(block, target);
                        }
                    }

                    Terminator::SwitchInt { ref $($mutability)* discr,
                                            switch_ty: _,
                                            values: _,
                                            ref targets } => {
                        self.visit_lvalue(discr, LvalueContext::Inspect);
                        for &target in targets {
                            self.visit_branch(block, target);
                        }
                    }

                    Terminator::Resume |
                    Terminator::Return => {
                    }

                    Terminator::Call { ref $($mutability)* func,
                                       ref $($mutability)* args,
                                       ref $($mutability)* kind } => {
                        self.visit_operand(func);
                        for arg in args {
                            self.visit_operand(arg);
                        }
                        match *kind {
                            CallKind::Converging {
                                ref $($mutability)* destination,
                                ..
                            }        |
                            CallKind::ConvergingCleanup {
                                ref $($mutability)* destination,
                                ..
                            } => {
                                self.visit_lvalue(destination, LvalueContext::Store);
                            }
                            CallKind::Diverging           |
                            CallKind::DivergingCleanup(_) => {}
                        }
                        for &target in kind.successors() {
                            self.visit_branch(block, target);
                        }
                    }
                }
            }

            fn super_rvalue(&mut self,
                            rvalue: & $($mutability)* Rvalue<'tcx>) {
                match *rvalue {
                    Rvalue::Use(ref $($mutability)* operand) => {
                        self.visit_operand(operand);
                    }

                    Rvalue::Repeat(ref $($mutability)* value,
                                   ref $($mutability)* len) => {
                        self.visit_operand(value);
                        self.visit_constant(len);
                    }

                    Rvalue::Ref(r, bk, ref $($mutability)* path) => {
                        self.visit_lvalue(path, LvalueContext::Borrow {
                            region: r,
                            kind: bk
                        });
                    }

                    Rvalue::Len(ref $($mutability)* path) => {
                        self.visit_lvalue(path, LvalueContext::Inspect);
                    }

                    Rvalue::Cast(_, ref $($mutability)* operand, _) => {
                        self.visit_operand(operand);
                    }

                    Rvalue::BinaryOp(_,
                                     ref $($mutability)* lhs,
                                     ref $($mutability)* rhs) => {
                        self.visit_operand(lhs);
                        self.visit_operand(rhs);
                    }

                    Rvalue::UnaryOp(_, ref $($mutability)* op) => {
                        self.visit_operand(op);
                    }

                    Rvalue::Box(_) => {
                    }

                    Rvalue::Aggregate(ref $($mutability)* kind,
                                      ref $($mutability)* operands) => {
                        match *kind {
                            AggregateKind::Closure(ref $($mutability)* def_id, _) => {
                                self.visit_def_id(def_id);
                            }
                            _ => { /* nothing to do */ }
                        }

                        for operand in & $($mutability)* operands[..] {
                            self.visit_operand(operand);
                        }
                    }

                    Rvalue::Slice { ref $($mutability)* input,
                                    from_start,
                                    from_end } => {
                        self.visit_lvalue(input, LvalueContext::Slice {
                            from_start: from_start,
                            from_end: from_end,
                        });
                    }

                    Rvalue::InlineAsm(_) => {
                    }
                }
            }

            fn super_operand(&mut self,
                             operand: & $($mutability)* Operand<'tcx>) {
                match *operand {
                    Operand::Consume(ref $($mutability)* lvalue) => {
                        self.visit_lvalue(lvalue, LvalueContext::Consume);
                    }
                    Operand::Constant(ref $($mutability)* constant) => {
                        self.visit_constant(constant);
                    }
                }
            }

            fn super_lvalue(&mut self,
                            lvalue: & $($mutability)* Lvalue<'tcx>,
                            _context: LvalueContext) {
                match *lvalue {
                    Lvalue::Var(_) |
                    Lvalue::Temp(_) |
                    Lvalue::Arg(_) |
                    Lvalue::ReturnPointer => {
                    }
                    Lvalue::Static(ref $($mutability)* def_id) => {
                        self.visit_def_id(def_id);
                    }
                    Lvalue::Projection(ref $($mutability)* proj) => {
                        self.visit_lvalue(& $($mutability)* proj.base,
                                          LvalueContext::Projection);
                    }
                }
            }

            fn super_branch(&mut self,
                            _source: BasicBlock,
                            _target: BasicBlock) {
            }

            fn super_constant(&mut self,
                              constant: & $($mutability)* Constant<'tcx>) {
                self.visit_span(& $($mutability)* constant.span);
                self.visit_literal(& $($mutability)* constant.literal);
            }

            fn super_literal(&mut self,
                             literal: & $($mutability)* Literal<'tcx>) {
                match *literal {
                    Literal::Item { ref $($mutability)* def_id, .. } => {
                        self.visit_def_id(def_id);
                    },
                    Literal::Value { .. } => {
                        // Nothing to do
                    }
                }
            }

            fn super_def_id(&mut self, _def_id: & $($mutability)* DefId) {
            }

            fn super_span(&mut self, _span: & $($mutability)* Span) {
            }
        }
    }
}

make_mir_visitor!(Visitor,);
make_mir_visitor!(MutVisitor,mut);

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
