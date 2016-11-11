// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A number of passes which remove various redundancies in the CFG.
//!
//! The `SimplifyCfg` pass gets rid of unnecessary blocks in the CFG, whereas the `SimplifyLocals`
//! gets rid of all the unnecessary local variable declarations.
//!
//! The `SimplifyLocals` pass is kinda expensive and therefore not very suitable to be run often.
//! Most of the passes should not care or be impacted in meaningful ways due to extra locals
//! either, so running the pass once, right before translation, should suffice.
//!
//! On the other side of the spectrum, the `SimplifyCfg` pass is considerably cheap to run, thus
//! one should run it after every pass which may modify CFG in significant ways. This pass must
//! also be run before any analysis passes because it removes dead blocks, and some of these can be
//! ill-typed.
//!
//! The cause of this typing issue is typeck allowing most blocks whose end is not reachable have
//! an arbitrary return type, rather than having the usual () return type (as a note, typeck's
//! notion of reachability is in fact slightly weaker than MIR CFG reachability - see #31617). A
//! standard example of the situation is:
//!
//! ```rust
//!   fn example() {
//!       let _a: char = { return; };
//!   }
//! ```
//!
//! Here the block (`{ return; }`) has the return type `char`, rather than `()`, but the MIR we
//! naively generate still contains the `_a = ()` write in the unreachable block "after" the
//! return.

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::mir::visit::{MutVisitor, Visitor, LvalueContext};
use std::fmt;

pub struct SimplifyCfg<'a> { label: &'a str }

impl<'a> SimplifyCfg<'a> {
    pub fn new(label: &'a str) -> Self {
        SimplifyCfg { label: label }
    }
}

impl<'l, 'tcx> MirPass<'tcx> for SimplifyCfg<'l> {
    fn run_pass<'a>(&mut self, _tcx: TyCtxt<'a, 'tcx, 'tcx>, _src: MirSource, mir: &mut Mir<'tcx>) {
        debug!("SimplifyCfg({:?}) - simplifying {:?}", self.label, mir);
        CfgSimplifier::new(mir).simplify();
        remove_dead_blocks(mir);

        // FIXME: Should probably be moved into some kind of pass manager
        mir.basic_blocks_mut().raw.shrink_to_fit();
    }
}

impl<'l> Pass for SimplifyCfg<'l> {
    fn disambiguator<'a>(&'a self) -> Option<Box<fmt::Display+'a>> {
        Some(Box::new(self.label))
    }

    // avoid calling `type_name` - it contains `<'static>`
    fn name(&self) -> ::std::borrow::Cow<'static, str> { "SimplifyCfg".into() }
}

pub struct CfgSimplifier<'a, 'tcx: 'a> {
    basic_blocks: &'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    pred_count: IndexVec<BasicBlock, u32>
}

impl<'a, 'tcx: 'a> CfgSimplifier<'a, 'tcx> {
    fn new(mir: &'a mut Mir<'tcx>) -> Self {
        let mut pred_count = IndexVec::from_elem(0u32, mir.basic_blocks());

        // we can't use mir.predecessors() here because that counts
        // dead blocks, which we don't want to.
        pred_count[START_BLOCK] = 1;

        for (_, data) in traversal::preorder(mir) {
            if let Some(ref term) = data.terminator {
                for &tgt in term.successors().iter() {
                    pred_count[tgt] += 1;
                }
            }
        }

        let basic_blocks = mir.basic_blocks_mut();

        CfgSimplifier {
            basic_blocks: basic_blocks,
            pred_count: pred_count
        }
    }

    fn simplify(mut self) {
        loop {
            let mut changed = false;

            for bb in (0..self.basic_blocks.len()).map(BasicBlock::new) {
                if self.pred_count[bb] == 0 {
                    continue
                }

                debug!("simplifying {:?}", bb);

                let mut terminator = self.basic_blocks[bb].terminator.take()
                    .expect("invalid terminator state");

                for successor in terminator.successors_mut() {
                    self.collapse_goto_chain(successor, &mut changed);
                }

                let mut new_stmts = vec![];
                let mut inner_changed = true;
                while inner_changed {
                    inner_changed = false;
                    inner_changed |= self.simplify_branch(&mut terminator);
                    inner_changed |= self.merge_successor(&mut new_stmts, &mut terminator);
                    changed |= inner_changed;
                }

                self.basic_blocks[bb].statements.extend(new_stmts);
                self.basic_blocks[bb].terminator = Some(terminator);

                changed |= inner_changed;
            }

            if !changed { break }
        }
    }

    // Collapse a goto chain starting from `start`
    fn collapse_goto_chain(&mut self, start: &mut BasicBlock, changed: &mut bool) {
        let mut terminator = match self.basic_blocks[*start] {
            BasicBlockData {
                ref statements,
                terminator: ref mut terminator @ Some(Terminator {
                    kind: TerminatorKind::Goto { .. }, ..
                }), ..
            } if statements.is_empty() => terminator.take(),
            // if `terminator` is None, this means we are in a loop. In that
            // case, let all the loop collapse to its entry.
            _ => return
        };

        let target = match terminator {
            Some(Terminator { kind: TerminatorKind::Goto { ref mut target }, .. }) => {
                self.collapse_goto_chain(target, changed);
                *target
            }
            _ => unreachable!()
        };
        self.basic_blocks[*start].terminator = terminator;

        debug!("collapsing goto chain from {:?} to {:?}", *start, target);

        *changed |= *start != target;

        if self.pred_count[*start] == 1 {
            // This is the last reference to *start, so the pred-count to
            // to target is moved into the current block.
            self.pred_count[*start] = 0;
        } else {
            self.pred_count[target] += 1;
            self.pred_count[*start] -= 1;
        }

        *start = target;
    }

    // merge a block with 1 `goto` predecessor to its parent
    fn merge_successor(&mut self,
                       new_stmts: &mut Vec<Statement<'tcx>>,
                       terminator: &mut Terminator<'tcx>)
                       -> bool
    {
        let target = match terminator.kind {
            TerminatorKind::Goto { target }
                if self.pred_count[target] == 1
                => target,
            _ => return false
        };

        debug!("merging block {:?} into {:?}", target, terminator);
        *terminator = match self.basic_blocks[target].terminator.take() {
            Some(terminator) => terminator,
            None => {
                // unreachable loop - this should not be possible, as we
                // don't strand blocks, but handle it correctly.
                return false
            }
        };
        new_stmts.extend(self.basic_blocks[target].statements.drain(..));
        self.pred_count[target] = 0;

        true
    }

    // turn a branch with all successors identical to a goto
    fn simplify_branch(&mut self, terminator: &mut Terminator<'tcx>) -> bool {
        match terminator.kind {
            TerminatorKind::If { .. } |
            TerminatorKind::Switch { .. } |
            TerminatorKind::SwitchInt { .. } => {},
            _ => return false
        };

        let first_succ = {
            let successors = terminator.successors();
            if let Some(&first_succ) = terminator.successors().get(0) {
                if successors.iter().all(|s| *s == first_succ) {
                    self.pred_count[first_succ] -= (successors.len()-1) as u32;
                    first_succ
                } else {
                    return false
                }
            } else {
                return false
            }
        };

        debug!("simplifying branch {:?}", terminator);
        terminator.kind = TerminatorKind::Goto { target: first_succ };
        true
    }
}

fn remove_dead_blocks(mir: &mut Mir) {
    let mut seen = BitVector::new(mir.basic_blocks().len());
    for (bb, _) in traversal::preorder(mir) {
        seen.insert(bb.index());
    }

    let basic_blocks = mir.basic_blocks_mut();

    let num_blocks = basic_blocks.len();
    let mut replacements : Vec<_> = (0..num_blocks).map(BasicBlock::new).collect();
    let mut used_blocks = 0;
    for alive_index in seen.iter() {
        replacements[alive_index] = BasicBlock::new(used_blocks);
        if alive_index != used_blocks {
            // Swap the next alive block data with the current available slot. Since alive_index is
            // non-decreasing this is a valid operation.
            basic_blocks.raw.swap(alive_index, used_blocks);
        }
        used_blocks += 1;
    }
    basic_blocks.raw.truncate(used_blocks);

    for block in basic_blocks {
        for target in block.terminator_mut().successors_mut() {
            *target = replacements[target.index()];
        }
    }
}


pub struct SimplifyLocals;

impl Pass for SimplifyLocals {
    fn name(&self) -> ::std::borrow::Cow<'static, str> { "SimplifyLocals".into() }
}

impl<'tcx> MirPass<'tcx> for SimplifyLocals {
    fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        let mut marker = DeclMarker { locals: BitVector::new(mir.local_decls.len()) };
        marker.visit_mir(mir);
        // Return pointer and arguments are always live
        marker.locals.insert(0);
        for idx in mir.args_iter() {
            marker.locals.insert(idx.index());
        }
        let map = make_local_map(&mut mir.local_decls, marker.locals);
        // Update references to all vars and tmps now
        LocalUpdater { map: map }.visit_mir(mir);
        mir.local_decls.shrink_to_fit();
    }
}

/// Construct the mapping while swapping out unused stuff out from the `vec`.
fn make_local_map<'tcx, I: Idx, V>(vec: &mut IndexVec<I, V>, mask: BitVector) -> Vec<usize> {
    let mut map: Vec<usize> = ::std::iter::repeat(!0).take(vec.len()).collect();
    let mut used = 0;
    for alive_index in mask.iter() {
        map[alive_index] = used;
        if alive_index != used {
            vec.swap(alive_index, used);
        }
        used += 1;
    }
    vec.truncate(used);
    map
}

struct DeclMarker {
    pub locals: BitVector,
}

impl<'tcx> Visitor<'tcx> for DeclMarker {
    fn visit_lvalue(&mut self, lval: &Lvalue<'tcx>, ctx: LvalueContext<'tcx>, loc: Location) {
        if ctx == LvalueContext::StorageLive || ctx == LvalueContext::StorageDead {
            // ignore these altogether, they get removed along with their otherwise unused decls.
            return;
        }
        if let Lvalue::Local(ref v) = *lval {
            self.locals.insert(v.index());
        }
        self.super_lvalue(lval, ctx, loc);
    }
}

struct LocalUpdater {
    map: Vec<usize>,
}

impl<'tcx> MutVisitor<'tcx> for LocalUpdater {
    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        // Remove unnecessary StorageLive and StorageDead annotations.
        data.statements.retain(|stmt| {
            match stmt.kind {
                StatementKind::StorageLive(ref lval) | StatementKind::StorageDead(ref lval) => {
                    match *lval {
                        Lvalue::Local(l) => self.map[l.index()] != !0,
                        _ => true
                    }
                }
                _ => true
            }
        });
        self.super_basic_block_data(block, data);
    }
    fn visit_lvalue(&mut self, lval: &mut Lvalue<'tcx>, ctx: LvalueContext<'tcx>, loc: Location) {
        match *lval {
            Lvalue::Local(ref mut l) => *l = Local::new(self.map[l.index()]),
            _ => (),
        };
        self.super_lvalue(lval, ctx, loc);
    }
}
