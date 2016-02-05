// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mir::repr::{Mir, BasicBlockData, BasicBlock};
use mir::mir_map::MirMap;
use middle::ty::ctxt;

/// Contains various metadata about the pass.
pub trait Pass {
    /// Ordering of the pass. Lower value runs the pass earlier.
    fn priority(&self) -> usize;
    // Possibly also `fn name()` and `fn should_run(Session)` etc.
}

/// Pass which inspects the whole MirMap.
pub trait MirMapPass: Pass {
    fn run_pass<'tcx>(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>);
}

/// Pass which only inspects MIR of distinct functions.
pub trait MirPass: Pass {
    fn run_pass<'tcx>(&mut self, tcx: &ctxt<'tcx>, mir: &mut Mir<'tcx>);
}

/// Pass which only inspects basic blocks in MIR.
///
/// Invariant: The blocks are considered to be fully self-contained for the purposes of this pass –
/// the pass may not change the list of successors of the block or apply any transformations to
/// blocks based on the information collected during earlier runs of the pass.
pub trait MirBlockPass: Pass {
    fn run_pass<'tcx>(&mut self, tcx: &ctxt<'tcx>, bb: BasicBlock, data: &mut BasicBlockData<'tcx>);
}

impl<T: MirBlockPass> MirPass for T {
    fn run_pass<'tcx>(&mut self, tcx: &ctxt<'tcx>, mir: &mut Mir<'tcx>) {
        for (i, basic_block) in mir.basic_blocks.iter_mut().enumerate() {
            MirBlockPass::run_pass(self, tcx, BasicBlock::new(i), basic_block);
        }
    }
}

impl<T: MirPass> MirMapPass for T {
    fn run_pass<'tcx>(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>) {
        for (_, mir) in &mut map.map {
            MirPass::run_pass(self, tcx, mir);
        }
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<MirMapPass>>
}

impl Passes {
    pub fn new() -> Passes {
        let passes = Passes {
            passes: Vec::new()
        };
        passes
    }

    pub fn run_passes<'tcx>(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>) {
        self.passes.sort_by_key(|e| e.priority());
        for pass in &mut self.passes {
            pass.run_pass(tcx, map);
        }
    }

    pub fn push_pass(&mut self, pass: Box<MirMapPass>) {
        self.passes.push(pass);
    }
}

impl ::std::iter::Extend<Box<MirMapPass>> for Passes {
    fn extend<I: IntoIterator<Item=Box<MirMapPass>>>(&mut self, it: I) {
        self.passes.extend(it);
    }
}
