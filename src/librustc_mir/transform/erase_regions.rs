// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass erases all early-bound regions from the types occuring in the MIR.
//! We want to do this once just before trans, so trans does not have to take
//! care erasing regions all over the place.

use rustc::middle::ty;
use rustc::mir::repr::*;
use transform::MirPass;
use mir_map::MirMap;

pub fn erase_regions<'tcx>(tcx: &ty::ctxt<'tcx>, mir_map: &mut MirMap<'tcx>) {
    let mut eraser = EraseRegions::new(tcx);

    for mir in mir_map.iter_mut().map(|(_, v)| v) {
        eraser.run_on_mir(mir);
    }
}

pub struct EraseRegions<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

impl<'a, 'tcx> MirPass<'tcx> for EraseRegions<'a, 'tcx> {

    fn run_on_mir(&mut self, mir: &mut Mir<'tcx>) {

        for basic_block in &mut mir.basic_blocks {
            self.erase_regions_basic_block(basic_block);
        }

        self.erase_regions_return_ty(&mut mir.return_ty);

        self.erase_regions_tys(mir.var_decls.iter_mut().map(|d| &mut d.ty));
        self.erase_regions_tys(mir.arg_decls.iter_mut().map(|d| &mut d.ty));
        self.erase_regions_tys(mir.temp_decls.iter_mut().map(|d| &mut d.ty));
    }
}

impl<'a, 'tcx> EraseRegions<'a, 'tcx> {

    pub fn new(tcx: &'a ty::ctxt<'tcx>) -> EraseRegions<'a, 'tcx> {
        EraseRegions {
            tcx: tcx
        }
    }

    fn erase_regions_basic_block(&mut self,
                                 basic_block: &mut BasicBlockData<'tcx>) {
        for statement in &mut basic_block.statements {
            self.erase_regions_statement(statement);
        }

        self.erase_regions_terminator(&mut basic_block.terminator);
    }

    fn erase_regions_statement(&mut self,
                               statement: &mut Statement<'tcx>) {
        match statement.kind {
            StatementKind::Assign(ref mut lvalue, ref mut rvalue) => {
                self.erase_regions_lvalue(lvalue);
                self.erase_regions_rvalue(rvalue);
            }
            StatementKind::Drop(_, ref mut lvalue) => {
                self.erase_regions_lvalue(lvalue);
            }
        }
    }

    fn erase_regions_terminator(&mut self,
                                terminator: &mut Terminator<'tcx>) {
        match *terminator {
            Terminator::Goto { .. } |
            Terminator::Diverge |
            Terminator::Resume |
            Terminator::Return => {
                /* nothing to do */
            }
            Terminator::If { ref mut cond, .. } => {
                self.erase_regions_operand(cond);
            }
            Terminator::Switch { ref mut discr, .. } => {
                self.erase_regions_lvalue(discr);
            }
            Terminator::SwitchInt { ref mut discr, ref mut switch_ty, .. } => {
                self.erase_regions_lvalue(discr);
                *switch_ty = self.tcx.erase_regions(switch_ty);
            },
            Terminator::Call { ref mut destination, ref mut func, ref mut args, .. } => {
                self.erase_regions_lvalue(destination);
                self.erase_regions_operand(func);
                for arg in &mut *args {
                    self.erase_regions_operand(arg);
                }
            }
            Terminator::DivergingCall { ref mut func, ref mut args, .. } => {
                self.erase_regions_operand(func);
                for arg in &mut *args {
                    self.erase_regions_operand(arg);
                }
            }
        }
    }

    fn erase_regions_operand(&mut self, operand: &mut Operand<'tcx>) {
        match *operand {
            Operand::Consume(ref mut lvalue) => {
                self.erase_regions_lvalue(lvalue);
            }
            Operand::Constant(ref mut constant) => {
                self.erase_regions_constant(constant);
            }
        }
    }

    fn erase_regions_lvalue(&mut self, lvalue: &mut Lvalue<'tcx>) {
        match *lvalue {
            Lvalue::Var(_)        |
            Lvalue::Temp(_)       |
            Lvalue::Arg(_)        |
            Lvalue::Static(_)     |
            Lvalue::ReturnPointer => {}
            Lvalue::Projection(ref mut lvalue_projection) => {
                self.erase_regions_lvalue(&mut lvalue_projection.base);
                match lvalue_projection.elem {
                    ProjectionElem::Deref              |
                    ProjectionElem::Field(_)           |
                    ProjectionElem::Downcast(..)       |
                    ProjectionElem::ConstantIndex {..} => { /* nothing to do */ }
                    ProjectionElem::Index(ref mut index) => {
                        self.erase_regions_operand(index);
                    }
                }
            }
        }
    }

    fn erase_regions_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(ref mut operand) => {
                self.erase_regions_operand(operand)
            }
            Rvalue::Repeat(ref mut operand, ref mut constant) => {
                self.erase_regions_operand(operand);
                self.erase_regions_constant(constant);
            }
            Rvalue::Ref(ref mut region, _, ref mut lvalue) => {
                *region = ty::ReStatic;
                self.erase_regions_lvalue(lvalue);
            }
            Rvalue::Len(ref mut lvalue) => self.erase_regions_lvalue(lvalue),
            Rvalue::Cast(_, ref mut operand, ref mut ty) => {
                self.erase_regions_operand(operand);
                *ty = self.tcx.erase_regions(ty);
            }
            Rvalue::BinaryOp(_, ref mut operand1, ref mut operand2) => {
                self.erase_regions_operand(operand1);
                self.erase_regions_operand(operand2);
            }
            Rvalue::UnaryOp(_, ref mut operand) => {
                self.erase_regions_operand(operand);
            }
            Rvalue::Box(ref mut ty) => *ty = self.tcx.erase_regions(ty),
            Rvalue::Aggregate(ref mut aggregate_kind, ref mut operands) => {
                match *aggregate_kind {
                    AggregateKind::Vec   |
                    AggregateKind::Tuple => {},
                    AggregateKind::Adt(_, _, ref mut substs) => {
                        let erased = self.tcx.erase_regions(*substs);
                        *substs = self.tcx.mk_substs(erased);
                    }
                    AggregateKind::Closure(def_id, ref mut closure_substs) => {
                        let cloned = Box::new(closure_substs.clone());
                        let ty = self.tcx.mk_closure_from_closure_substs(def_id,
                                                                         cloned);
                        let erased = self.tcx.erase_regions(&ty);
                        *closure_substs = match erased.sty {
                            ty::TyClosure(_, ref closure_substs) => &*closure_substs,
                            _ => unreachable!()
                        };
                    }
                }
                for operand in &mut *operands {
                    self.erase_regions_operand(operand);
                }
            }
            Rvalue::Slice { ref mut input, .. } => {
                self.erase_regions_lvalue(input);
            }
            Rvalue::InlineAsm(_) => {},
        }
    }

    fn erase_regions_constant(&mut self, constant: &mut Constant<'tcx>) {
        constant.ty = self.tcx.erase_regions(&constant.ty);
        match constant.literal {
            Literal::Item { ref mut substs, .. } => {
                *substs = self.tcx.mk_substs(self.tcx.erase_regions(substs));
            }
            Literal::Value { .. } => { /* nothing to do */ }
        }
    }

    fn erase_regions_return_ty(&mut self, fn_output: &mut ty::FnOutput<'tcx>) {
        match *fn_output {
            ty::FnConverging(ref mut ty) => {
                *ty = self.tcx.erase_regions(ty);
            },
            ty::FnDiverging => {}
        }
    }

    fn erase_regions_tys<'b, T>(&mut self, tys: T)
        where T: Iterator<Item = &'b mut ty::Ty<'tcx>>,
              'tcx: 'b
    {
        for ty in tys {
            *ty = self.tcx.erase_regions(ty);
        }
    }
}
