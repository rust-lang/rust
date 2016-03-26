// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mir::mir_map::MirMap;
use mir::repr::Mir;
use ty::TyCtxt;
use syntax::ast::NodeId;

/// Various information about pass.
pub trait Pass {
    // fn name() for printouts of various sorts?
    // fn should_run(Session) to check if pass should run?
}

/// A pass which inspects the whole MirMap.
pub trait MirMapPass<'tcx>: Pass {
    fn run_pass(&mut self, cx: &TyCtxt<'tcx>, map: &mut MirMap<'tcx>);
}

/// A pass which inspects Mir of functions in isolation.
pub trait MirPass<'tcx>: Pass {
    fn run_pass(&mut self, cx: &TyCtxt<'tcx>, id: NodeId, mir: &mut Mir<'tcx>);
}

impl<'tcx, T: MirPass<'tcx>> MirMapPass<'tcx> for T {
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, map: &mut MirMap<'tcx>) {
        for (&id, mir) in &mut map.map {
            MirPass::run_pass(self, tcx, id, mir);
        }
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>,
    plugin_passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>
}

impl Passes {
    pub fn new() -> Passes {
        let passes = Passes {
            passes: Vec::new(),
            plugin_passes: Vec::new()
        };
        passes
    }

    pub fn run_passes<'tcx>(&mut self, pcx: &TyCtxt<'tcx>, map: &mut MirMap<'tcx>) {
        for pass in &mut self.plugin_passes {
            pass.run_pass(pcx, map);
        }
        for pass in &mut self.passes {
            pass.run_pass(pcx, map);
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass(&mut self, pass: Box<for<'a> MirMapPass<'a>>) {
        self.passes.push(pass);
    }
}

/// Copies the plugin passes.
impl ::std::iter::Extend<Box<for<'a> MirMapPass<'a>>> for Passes {
    fn extend<I: IntoIterator<Item=Box<for <'a> MirMapPass<'a>>>>(&mut self, it: I) {
        self.plugin_passes.extend(it);
    }
}
