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

use rustc::middle::ty::{self, TyCtxt};
use rustc::mir::repr::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{MirPass, Pass};
use syntax::ast::NodeId;

struct EraseRegionsVisitor<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
}

impl<'a, 'tcx> EraseRegionsVisitor<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'tcx>) -> Self {
        EraseRegionsVisitor {
            tcx: tcx
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

impl<'a, 'tcx> MutVisitor<'tcx> for EraseRegionsVisitor<'a, 'tcx> {
    fn visit_mir(&mut self, mir: &mut Mir<'tcx>) {
        self.erase_regions_return_ty(&mut mir.return_ty);
        self.erase_regions_tys(mir.var_decls.iter_mut().map(|d| &mut d.ty));
        self.erase_regions_tys(mir.arg_decls.iter_mut().map(|d| &mut d.ty));
        self.erase_regions_tys(mir.temp_decls.iter_mut().map(|d| &mut d.ty));
        self.super_mir(mir);
    }

    fn visit_terminator(&mut self, bb: BasicBlock, terminator: &mut Terminator<'tcx>) {
        match *terminator {
            Terminator::Goto { .. } |
            Terminator::Resume |
            Terminator::Return |
            Terminator::If { .. } |
            Terminator::Switch { .. } |
            Terminator::Drop { .. } |
            Terminator::Call { .. } => {
                /* nothing to do */
            },
            Terminator::SwitchInt { ref mut switch_ty, .. } => {
                *switch_ty = self.tcx.erase_regions(switch_ty);
            },
        }
        self.super_terminator(bb, terminator);
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Use(_) |
            Rvalue::Len(_) |
            Rvalue::BinaryOp(_, _, _) |
            Rvalue::UnaryOp(_, _) |
            Rvalue::Slice { input: _, from_start: _, from_end: _ } |
            Rvalue::InlineAsm {..} => {},

            Rvalue::Repeat(_, ref mut value) => value.ty = self.tcx.erase_regions(&value.ty),
            Rvalue::Ref(ref mut region, _, _) => *region = ty::ReStatic,
            Rvalue::Cast(_, _, ref mut ty) => *ty = self.tcx.erase_regions(ty),
            Rvalue::Box(ref mut ty) => *ty = self.tcx.erase_regions(ty),


            Rvalue::Aggregate(AggregateKind::Vec, _) |
            Rvalue::Aggregate(AggregateKind::Tuple, _) => {},
            Rvalue::Aggregate(AggregateKind::Adt(_, _, ref mut substs), _) =>
                *substs = self.tcx.mk_substs(self.tcx.erase_regions(*substs)),
            Rvalue::Aggregate(AggregateKind::Closure(def_id, ref mut closure_substs), _) => {
                let cloned = Box::new(closure_substs.clone());
                let ty = self.tcx.mk_closure_from_closure_substs(def_id, cloned);
                let erased = self.tcx.erase_regions(&ty);
                *closure_substs = match erased.sty {
                    ty::TyClosure(_, ref closure_substs) => &*closure_substs,
                    _ => unreachable!()
                };
            }
        }
        self.super_rvalue(rvalue);
    }

    fn visit_constant(&mut self, constant: &mut Constant<'tcx>) {
        constant.ty = self.tcx.erase_regions(&constant.ty);
        match constant.literal {
            Literal::Item { ref mut substs, .. } => {
                *substs = self.tcx.mk_substs(self.tcx.erase_regions(substs));
            }
            Literal::Value { .. } => { /* nothing to do */ }
        }
        self.super_constant(constant);
    }
}

pub struct EraseRegions;

impl Pass for EraseRegions {}

impl<'tcx> MirPass<'tcx> for EraseRegions {
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, _: NodeId, mir: &mut Mir<'tcx>) {
        EraseRegionsVisitor::new(tcx).visit_mir(mir);
    }
}
