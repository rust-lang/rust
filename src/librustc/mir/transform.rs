// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::DepNode;
use hir;
use hir::map::DefPathData;
use hir::def_id::DefId;
use mir::mir_map::MirMap;
use mir::repr::Mir;
use ty::TyCtxt;
use syntax::ast::NodeId;

/// Where a specific Mir comes from.
#[derive(Debug, Copy, Clone)]
pub enum MirSource {
    /// Functions and methods.
    Fn(NodeId),

    /// Constants and associated constants.
    Const(NodeId),

    /// Initializer of a `static` item.
    Static(NodeId, hir::Mutability),

    /// Promoted rvalues within a function.
    Promoted(NodeId, usize)
}

impl<'a, 'tcx> MirSource {
    pub fn from_node(tcx: TyCtxt<'a, 'tcx, 'tcx>, id: NodeId) -> MirSource {
        use hir::*;

        // Handle constants in enum discriminants, types, and repeat expressions.
        let def_id = tcx.map.local_def_id(id);
        let def_key = tcx.def_key(def_id);
        if def_key.disambiguated_data.data == DefPathData::Initializer {
            return MirSource::Const(id);
        }

        match tcx.map.get(id) {
            map::NodeItem(&Item { node: ItemConst(..), .. }) |
            map::NodeTraitItem(&TraitItem { node: ConstTraitItem(..), .. }) |
            map::NodeImplItem(&ImplItem { node: ImplItemKind::Const(..), .. }) => {
                MirSource::Const(id)
            }
            map::NodeItem(&Item { node: ItemStatic(_, m, _), .. }) => {
                MirSource::Static(id, m)
            }
            // Default to function if it's not a constant or static.
            _ => MirSource::Fn(id)
        }
    }

    pub fn item_id(&self) -> NodeId {
        match *self {
            MirSource::Fn(id) |
            MirSource::Const(id) |
            MirSource::Static(id, _) |
            MirSource::Promoted(id, _) => id
        }
    }
}

/// Various information about pass.
pub trait Pass {
    // fn name() for printouts of various sorts?
    // fn should_run(Session) to check if pass should run?
    fn dep_node(&self, def_id: DefId) -> DepNode<DefId> {
        DepNode::MirPass(def_id)
    }
}

/// A pass which inspects the whole MirMap.
pub trait MirMapPass<'tcx>: Pass {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, map: &mut MirMap<'tcx>);
}

/// A pass which inspects Mir of functions in isolation.
pub trait MirPass<'tcx>: Pass {
    fn run_pass_on_promoted<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                item_id: NodeId, index: usize,
                                mir: &mut Mir<'tcx>) {
        self.run_pass(tcx, MirSource::Promoted(item_id, index), mir);
    }
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>);
}

impl<'tcx, T: MirPass<'tcx>> MirMapPass<'tcx> for T {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, map: &mut MirMap<'tcx>) {
        for (&id, mir) in &mut map.map {
            let def_id = tcx.map.local_def_id(id);
            let _task = tcx.dep_graph.in_task(self.dep_node(def_id));

            let src = MirSource::from_node(tcx, id);
            MirPass::run_pass(self, tcx, src, mir);

            for (i, mir) in mir.promoted.iter_mut().enumerate() {
                self.run_pass_on_promoted(tcx, id, i, mir);
            }
        }
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>,
    plugin_passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>
}

impl<'a, 'tcx> Passes {
    pub fn new() -> Passes {
        let passes = Passes {
            passes: Vec::new(),
            plugin_passes: Vec::new()
        };
        passes
    }

    pub fn run_passes(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, map: &mut MirMap<'tcx>) {
        for pass in &mut self.plugin_passes {
            pass.run_pass(tcx, map);
        }
        for pass in &mut self.passes {
            pass.run_pass(tcx, map);
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass(&mut self, pass: Box<for<'b> MirMapPass<'b>>) {
        self.passes.push(pass);
    }
}

/// Copies the plugin passes.
impl ::std::iter::Extend<Box<for<'a> MirMapPass<'a>>> for Passes {
    fn extend<I: IntoIterator<Item=Box<for <'a> MirMapPass<'a>>>>(&mut self, it: I) {
        self.plugin_passes.extend(it);
    }
}
