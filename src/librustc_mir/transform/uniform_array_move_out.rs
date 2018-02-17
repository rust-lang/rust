// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This pass converts move out from array by Subslice and
// ConstIndex{.., from_end: true} to ConstIndex move out(s) from begin
// of array. It allows detect error by mir borrowck and elaborate
// drops for array without additional work.
//
// Example:
//
// let a = [ box 1,box 2, box 3];
// if b {
//  let [_a.., _] = a;
// } else {
//  let [.., _b] = a;
// }
//
//  mir statement _10 = move _2[:-1]; replaced by:
//  StorageLive(_12);
//  _12 = move _2[0 of 3];
//  StorageLive(_13);
//  _13 = move _2[1 of 3];
//  _10 = [move _12, move _13]
//  StorageDead(_12);
//  StorageDead(_13);
//
//  and mir statement _11 = move _2[-1 of 1]; replaced by:
//  _11 = move _2[2 of 3];
//
// FIXME: convert to Subslice back for performance reason
// FIXME: integrate this transformation to the mir build

use rustc::ty;
use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::visit::Visitor;
use transform::{MirPass, MirSource};
use util::patch::MirPatch;

pub struct UniformArrayMoveOut;

impl MirPass for UniformArrayMoveOut {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _src: MirSource,
                          mir: &mut Mir<'tcx>) {
        let mut patch = MirPatch::new(mir);
        {
            let mut visitor = UniformArrayMoveOutVisitor{mir, patch: &mut patch, tcx};
            visitor.visit_mir(mir);
        }
        patch.apply(mir);
    }
}

struct UniformArrayMoveOutVisitor<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    patch: &'a mut MirPatch<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for UniformArrayMoveOutVisitor<'a, 'tcx> {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &Statement<'tcx>,
                       location: Location) {
        if let StatementKind::Assign(ref dst_place,
                                     Rvalue::Use(Operand::Move(ref src_place))) = statement.kind {
            if let Place::Projection(ref proj) = *src_place {
                if let ProjectionElem::ConstantIndex{offset: _,
                                                     min_length: _,
                                                     from_end: false} = proj.elem {
                    // no need to transformation
                } else {
                    let place_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                    if let ty::TyArray(item_ty, const_size) = place_ty.sty {
                        if let Some(size) = const_size.val.to_const_int().and_then(|v| v.to_u64()) {
                            assert!(size <= (u32::max_value() as u64),
                                    "unform array move out doesn't supported
                                     for array bigger then u32");
                            self.uniform(location, dst_place, proj, item_ty, size as u32);
                        }
                    }

                }
            }
        }
        return self.super_statement(block, statement, location);
    }
}

impl<'a, 'tcx> UniformArrayMoveOutVisitor<'a, 'tcx> {
    fn uniform(&mut self,
               location: Location,
               dst_place: &Place<'tcx>,
               proj: &PlaceProjection<'tcx>,
               item_ty: &'tcx ty::TyS<'tcx>,
               size: u32) {
        match proj.elem {
            // uniform _10 = move _2[:-1];
            ProjectionElem::Subslice{from, to} => {
                self.patch.make_nop(location);
                let temps : Vec<_> = (from..(size-to)).map(|i| {
                    let temp = self.patch.new_temp(item_ty, self.mir.source_info(location).span);
                    self.patch.add_statement(location, StatementKind::StorageLive(temp));
                    self.patch.add_assign(location,
                                          Place::Local(temp),
                                          Rvalue::Use(
                                              Operand::Move(
                                                  Place::Projection(box PlaceProjection{
                                                      base: proj.base.clone(),
                                                      elem: ProjectionElem::ConstantIndex{
                                                          offset: i,
                                                          min_length: size,
                                                          from_end: false}
                                                  }))));
                    temp
                }).collect();
                self.patch.add_assign(location,
                                      dst_place.clone(),
                                      Rvalue::Aggregate(box AggregateKind::Array(item_ty),
                                      temps.iter().map(
                                          |x| Operand::Move(Place::Local(*x))).collect()
                                      ));
                for temp in temps {
                    self.patch.add_statement(location, StatementKind::StorageDead(temp));
                }
            }
            // _11 = move _2[-1 of 1];
            ProjectionElem::ConstantIndex{offset, min_length: _, from_end: true} => {
                self.patch.make_nop(location);
                self.patch.add_assign(location,
                                      dst_place.clone(),
                                      Rvalue::Use(
                                          Operand::Move(
                                              Place::Projection(box PlaceProjection{
                                                  base: proj.base.clone(),
                                                  elem: ProjectionElem::ConstantIndex{
                                                      offset: size - offset,
                                                      min_length: size,
                                                      from_end: false }}))));
            }
            _ => {}
        }
    }
}
