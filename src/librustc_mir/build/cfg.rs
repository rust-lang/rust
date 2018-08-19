// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




//! Routines for manipulating the control-flow graph.

use build::CFG;
use rustc::middle::region;
use rustc::mir::*;
use rustc::ty::TyCtxt;

impl<'tcx> CFG<'tcx> {
    pub fn block_data(&self, blk: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks[blk]
    }

    pub fn block_data_mut(&mut self, blk: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks[blk]
    }

    // llvm.org/PR32488 makes this function use an excess of stack space. Mark
    // it as #[inline(never)] to keep rustc's stack use in check.
    #[inline(never)]
    pub fn start_new_block(&mut self) -> BasicBlock {
        self.basic_blocks.push(BasicBlockData::new(None))
    }

    pub fn start_new_cleanup_block(&mut self) -> BasicBlock {
        let bb = self.start_new_block();
        self.block_data_mut(bb).is_cleanup = true;
        bb
    }

    pub fn push(&mut self, block: BasicBlock, statement: Statement<'tcx>) {
        debug!("push({:?}, {:?})", block, statement);
        self.block_data_mut(block).statements.push(statement);
    }

    pub fn push_end_region<'a, 'gcx:'a+'tcx>(&mut self,
                                             tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                             block: BasicBlock,
                                             source_info: SourceInfo,
                                             region_scope: region::Scope) {
        if tcx.emit_end_regions() {
            if let region::ScopeData::CallSite(_) = region_scope.data() {
                // The CallSite scope (aka the root scope) is sort of weird, in that it is
                // supposed to "separate" the "interior" and "exterior" of a closure. Being
                // that, it is not really a part of the region hierarchy, but for some
                // reason it *is* considered a part of it.
                //
                // It should die a hopefully painful death with NLL, so let's leave this hack
                // for now so that nobody can complain about soundness.
                return
            }

            self.push(block, Statement {
                source_info,
                kind: StatementKind::EndRegion(region_scope),
            });
        }
    }

    pub fn push_assign(&mut self,
                       block: BasicBlock,
                       source_info: SourceInfo,
                       place: &Place<'tcx>,
                       rvalue: Rvalue<'tcx>) {
        self.push(block, Statement {
            source_info,
            kind: StatementKind::Assign(place.clone(), rvalue)
        });
    }

    pub fn push_assign_constant(&mut self,
                                block: BasicBlock,
                                source_info: SourceInfo,
                                temp: &Place<'tcx>,
                                constant: Constant<'tcx>) {
        self.push_assign(block, source_info, temp,
                         Rvalue::Use(Operand::Constant(box constant)));
    }

    pub fn push_assign_unit(&mut self,
                            block: BasicBlock,
                            source_info: SourceInfo,
                            place: &Place<'tcx>) {
        self.push_assign(block, source_info, place, Rvalue::Aggregate(
            box AggregateKind::Tuple, vec![]
        ));
    }

    pub fn terminate(&mut self,
                     block: BasicBlock,
                     source_info: SourceInfo,
                     kind: TerminatorKind<'tcx>) {
        debug!("terminating block {:?} <- {:?}", block, kind);
        debug_assert!(self.block_data(block).terminator.is_none(),
                      "terminate: block {:?}={:?} already has a terminator set",
                      block,
                      self.block_data(block));
        self.block_data_mut(block).terminator = Some(Terminator {
            source_info,
            kind,
        });
    }
}
