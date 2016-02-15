// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mir::repr::Mir;
use mir::mir_map::MirMap;
use middle::ty::ctxt;

/// Contains various metadata about the pass.
pub trait Pass {
    // Possibly also `fn name()` and `fn should_run(Session)` etc.
}

/// Pass which inspects the whole MirMap.
pub trait MirMapPass<'tcx>: Pass {
    fn run_pass(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>);
}

/// Pass which only inspects MIR of distinct functions.
pub trait MirPass<'tcx>: Pass {
    fn run_pass(&mut self, tcx: &ctxt<'tcx>, mir: &mut Mir<'tcx>);
}

impl<'tcx, T: MirPass<'tcx>> MirMapPass<'tcx> for T {
    fn run_pass(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>) {
        for (_, mir) in &mut map.map {
            MirPass::run_pass(self, tcx, mir);
        }
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>
}

impl Passes {
    pub fn new() -> Passes {
        let passes = Passes {
            passes: Vec::new()
        };
        passes
    }

    pub fn run_passes<'tcx>(&mut self, tcx: &ctxt<'tcx>, map: &mut MirMap<'tcx>) {
        for pass in &mut self.passes {
            pass.run_pass(tcx, map);
        }
    }

    pub fn push_pass(&mut self, pass: Box<for<'a> MirMapPass<'a>>) {
        self.passes.push(pass);
    }
}

impl ::std::iter::Extend<Box<for<'a> MirMapPass<'a>>> for Passes {
    fn extend<I: IntoIterator<Item=Box<for <'a> MirMapPass<'a>>>>(&mut self, it: I) {
        self.passes.extend(it);
    }
}
