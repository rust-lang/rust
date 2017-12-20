// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(warnings)]

use rustc::mir::*;
use rustc::mir::visit::{MutVisitor, PlaceContext};
use rustc::ty::TyCtxt;
use rustc_data_structures::control_flow_graph::iterate::post_order_from;
use rustc_data_structures::fx::{FxHashMap};
use rustc_data_structures::indexed_set::{IdxSetBuf};
use rustc_data_structures::indexed_vec::Idx;
use transform::{MirPass, MirSource};

pub struct WeakenLastUse;

impl MirPass for WeakenLastUse {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _src: MirSource,
                          mir: &mut Mir<'tcx>) {
        // We only run when optimizing MIR (at any level).
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return
        }

        let post_order = post_order_from(mir, START_BLOCK);
        let (basic_blocks, local_decls) = mir.basic_blocks_and_local_decls_mut();
        let locals_len = local_decls.len();
        let mut simplifier = CopySimplifier {
            locals_len,
            block_needs: FxHashMap::default(),
            needs: None,
            dups_check: IdxSetBuf::new_empty(locals_len),
            bug: None,
         };
        for bb in post_order {
            simplifier.simplify(bb, &mut basic_blocks[bb]);
            if let Some((local, location)) = simplifier.bug.take() {
                bug!("Local {:?} copied twice in {:?}; last-use logic is wrong.  Blocks: {:#?}",
                    local, location, basic_blocks);
            }
        }
    }
}

struct CopySimplifier {
    locals_len: usize,
    block_needs: FxHashMap<BasicBlock, IdxSetBuf<Local>>,
    needs: Option<IdxSetBuf<Local>>,
    dups_check: IdxSetBuf<Local>,
    bug: Option<(Local, Location)>,
}

impl CopySimplifier {
    fn simplify<'tcx>(&mut self, bb: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        self.needs = Some(self.succ_needs(data.terminator()));
        debug!("Starting {:?} needing {:?}", bb, self.needs);

        let mut location = Location {
            block: bb,
            statement_index: data.statements.len(),
        };
        self.dups_check.reset_to_empty();
        self.visit_terminator(bb, data.terminator.as_mut().unwrap(), location);
        for statement in data.statements.iter_mut().rev() {
            location.statement_index -= 1;
            self.dups_check.reset_to_empty();
            self.visit_statement(bb, statement, location);
        }
        debug_assert_eq!(location.statement_index, 0);

        self.block_needs.insert(bb, self.needs.take().unwrap());
    }

    fn succ_needs<'tcx>(&mut self, terminator: &Terminator<'tcx>) -> IdxSetBuf<Local> {
        let mut needs: Option<IdxSetBuf<Local>> = None;
        for bb in terminator.successors().iter() {
            if let Some(succ_needs) = self.block_needs.get(bb) {
                if let Some(ref mut needs) = needs {
                    needs.union(succ_needs);
                } else {
                    needs = Some(succ_needs.clone());
                }
            } else {
                // Back edge, so assume it needs everything
                return IdxSetBuf::new_filled(self.locals_len);
            }
        }

        needs.unwrap_or_else(|| IdxSetBuf::new_filled(self.locals_len))
    }

    fn needs(&mut self) -> &mut IdxSetBuf<Local> {
        self.needs.as_mut().unwrap()
    }
}

impl<'tcx> MutVisitor<'tcx> for CopySimplifier {
    fn visit_local(
        &mut self,
        local: &mut Local,
        context: PlaceContext<'tcx>,
        _location: Location,
    ) {
        match context {
            PlaceContext::Store |
            PlaceContext::StorageLive |
            PlaceContext::StorageDead => {
                self.needs().remove(local);
            }

            // A call doesn't need its output populated, but also might not
            // store a value if the callee panics, so just do nothing here.
            // FIXME: Smarter handling of successors in call terminators
            // would let this be more precise, but this is sound.
            PlaceContext::Call => {}

            // While an InlineAsm output is expected to write to the output,
            // they can be read-write, so assume we need the preceeding value.
            PlaceContext::AsmOutput |
            PlaceContext::Projection(..) |
            PlaceContext::Borrow { .. } |
            PlaceContext::Inspect |
            PlaceContext::Copy |
            PlaceContext::Move |
            PlaceContext::Validate |
            PlaceContext::Drop => {
                self.needs().add(local);
            }
        }
    }

    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &mut Statement<'tcx>,
        location: Location,
    ) {
        if let StatementKind::Assign(Place::Local(local), _) = statement.kind {
            if !self.needs().contains(&local) {
                // All rvalues are side-effect-free, so if nothing needs this
                // local, we can just skip this.  The local is being referenced
                // directly, so must not be borrowed either.
                statement.make_nop();
                return;
            }
        }

        self.super_statement(block, statement, location);
    }

    fn visit_operand(
        &mut self,
        operand: &mut Operand<'tcx>,
        location: Location,
    ) {
        if let Operand::Copy(Place::Local(local)) = *operand {
            if self.dups_check.contains(&local) && self.bug.is_none() {
                self.bug = Some((local, location));
            }
            if !self.needs().contains(&local) {
                *operand = Operand::Move(Place::Local(local));
            }
            self.dups_check.add(&local);
        }

        self.super_operand(operand, location)
    }

    fn visit_terminator_kind(
        &mut self,
        block: BasicBlock,
        kind: &mut TerminatorKind<'tcx>,
        location: Location,
    ) {
        match *kind {
            TerminatorKind::Unreachable => {
                self.needs().clear();
            }
            TerminatorKind::Return => {
                self.needs().clear();
                self.needs().add(&Local::new(0));
            }
            _ => {
                self.super_terminator_kind(block, kind, location)
            }
        }
    }
}
