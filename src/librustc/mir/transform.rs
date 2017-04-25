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
use hir::def_id::{DefId, LOCAL_CRATE};
use hir::map::DefPathData;
use mir::{Mir, Promoted};
use ty::TyCtxt;
use syntax::ast::NodeId;
use util::common::time;

use std::borrow::Cow;

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
    Promoted(NodeId, Promoted)
}

impl<'a, 'tcx> MirSource {
    pub fn from_node(tcx: TyCtxt<'a, 'tcx, 'tcx>, id: NodeId) -> MirSource {
        use hir::*;

        // Handle constants in enum discriminants, types, and repeat expressions.
        let def_id = tcx.hir.local_def_id(id);
        let def_key = tcx.def_key(def_id);
        if def_key.disambiguated_data.data == DefPathData::Initializer {
            return MirSource::Const(id);
        }

        match tcx.hir.get(id) {
            map::NodeItem(&Item { node: ItemConst(..), .. }) |
            map::NodeTraitItem(&TraitItem { node: TraitItemKind::Const(..), .. }) |
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
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>);
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
pub fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = unsafe { ::std::intrinsics::type_name::<T>() };
    if let Some(tail) = name.rfind(":") {
        Cow::from(&name[tail+1..])
    } else {
        Cow::from(name)
    }
}

pub trait PassHook {
    fn on_mir_pass<'a, 'tcx>(&self,
                             tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             pass: &Pass,
                             pass_num: usize,
                             is_after: bool);
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource,
                          mir: &mut Mir<'tcx>);
}

impl<T: MirPass> Pass for T {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        MirPass::name(self)
    }

    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        for &def_id in tcx.mir_keys(LOCAL_CRATE).iter() {
            run_map_pass_task(tcx, self, def_id);
        }
    }
}

fn run_map_pass_task<'a, 'tcx, T: MirPass>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           pass: &T,
                                           def_id: DefId) {
    let _task = tcx.dep_graph.in_task(DepNode::Mir(def_id));
    let mir = &mut tcx.mir(def_id).borrow_mut();
    let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
    let source = MirSource::from_node(tcx, id);
    MirPass::run_pass(pass, tcx, source, mir);

    for (i, mir) in mir.promoted.iter_enumerated_mut() {
        let source = MirSource::Promoted(id, i);
        MirPass::run_pass(pass, tcx, source, mir);
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<Pass>>,
    pass_hooks: Vec<Box<PassHook>>,
    plugin_passes: Vec<Box<Pass>>
}

impl<'a, 'tcx> Passes {
    pub fn new() -> Passes {
        let passes = Passes {
            passes: Vec::new(),
            pass_hooks: Vec::new(),
            plugin_passes: Vec::new()
        };
        passes
    }

    pub fn run_passes(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        // NB: passes are numbered from 1, since "construction" is zero.
        for (pass, pass_num) in self.plugin_passes.iter().chain(&self.passes).zip(1..) {
            for hook in &self.pass_hooks {
                hook.on_mir_pass(tcx, &**pass, pass_num, false);
            }

            time(tcx.sess.time_passes(), &*pass.name(), || pass.run_pass(tcx));

            for hook in &self.pass_hooks {
                hook.on_mir_pass(tcx, &**pass, pass_num, true);
            }
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass(&mut self, pass: Box<Pass>) {
        self.passes.push(pass);
    }

    /// Pushes a pass hook.
    pub fn push_hook(&mut self, hook: Box<PassHook>) {
        self.pass_hooks.push(hook);
    }
}

/// Copies the plugin passes.
impl ::std::iter::Extend<Box<Pass>> for Passes {
    fn extend<I: IntoIterator<Item=Box<Pass>>>(&mut self, it: I) {
        self.plugin_passes.extend(it);
    }
}
