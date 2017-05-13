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
use mir::{Mir, Promoted};
use ty::TyCtxt;
use syntax::ast::NodeId;
use util::common::time;

use std::borrow::Cow;
use std::fmt;

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
        let def_id = tcx.map.local_def_id(id);
        let def_key = tcx.def_key(def_id);
        if def_key.disambiguated_data.data == DefPathData::Initializer {
            return MirSource::Const(id);
        }

        match tcx.map.get(id) {
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
    // fn should_run(Session) to check if pass should run?
    fn name<'a>(&self) -> Cow<'static, str> {
        let name = unsafe { ::std::intrinsics::type_name::<Self>() };
        if let Some(tail) = name.rfind(":") {
            Cow::from(&name[tail+1..])
        } else {
            Cow::from(name)
        }
    }
    fn disambiguator<'a>(&'a self) -> Option<Box<fmt::Display+'a>> { None }
}

/// A pass which inspects the whole Mir map.
pub trait MirMapPass<'tcx>: Pass {
    fn run_pass<'a>(
        &mut self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        hooks: &mut [Box<for<'s> MirPassHook<'s>>]);
}

pub trait MirPassHook<'tcx>: Pass {
    fn on_mir_pass<'a>(
        &mut self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        src: MirSource,
        mir: &Mir<'tcx>,
        pass: &Pass,
        is_after: bool
    );
}

/// A pass which inspects Mir of functions in isolation.
pub trait MirPass<'tcx>: Pass {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>);
}

impl<'tcx, T: MirPass<'tcx>> MirMapPass<'tcx> for T {
    fn run_pass<'a>(&mut self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    hooks: &mut [Box<for<'s> MirPassHook<'s>>])
    {
        let def_ids = tcx.mir_map.borrow().keys();
        for def_id in def_ids {
            if !def_id.is_local() {
                continue;
            }

            let _task = tcx.dep_graph.in_task(DepNode::Mir(def_id));
            let mir = &mut tcx.mir_map.borrow()[&def_id].borrow_mut();
            tcx.dep_graph.write(DepNode::Mir(def_id));

            let id = tcx.map.as_local_node_id(def_id).unwrap();
            let src = MirSource::from_node(tcx, id);

            for hook in &mut *hooks {
                hook.on_mir_pass(tcx, src, mir, self, false);
            }
            MirPass::run_pass(self, tcx, src, mir);
            for hook in &mut *hooks {
                hook.on_mir_pass(tcx, src, mir, self, true);
            }

            for (i, mir) in mir.promoted.iter_enumerated_mut() {
                let src = MirSource::Promoted(id, i);
                for hook in &mut *hooks {
                    hook.on_mir_pass(tcx, src, mir, self, false);
                }
                MirPass::run_pass(self, tcx, src, mir);
                for hook in &mut *hooks {
                    hook.on_mir_pass(tcx, src, mir, self, true);
                }
            }
        }
    }
}

/// A manager for MIR passes.
pub struct Passes {
    passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>,
    pass_hooks: Vec<Box<for<'tcx> MirPassHook<'tcx>>>,
    plugin_passes: Vec<Box<for<'tcx> MirMapPass<'tcx>>>
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
        let Passes { ref mut passes, ref mut plugin_passes, ref mut pass_hooks } = *self;
        for pass in plugin_passes.iter_mut().chain(passes.iter_mut()) {
            time(tcx.sess.time_passes(), &*pass.name(),
                 || pass.run_pass(tcx, pass_hooks));
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass(&mut self, pass: Box<for<'b> MirMapPass<'b>>) {
        self.passes.push(pass);
    }

    /// Pushes a pass hook.
    pub fn push_hook(&mut self, hook: Box<for<'b> MirPassHook<'b>>) {
        self.pass_hooks.push(hook);
    }
}

/// Copies the plugin passes.
impl ::std::iter::Extend<Box<for<'a> MirMapPass<'a>>> for Passes {
    fn extend<I: IntoIterator<Item=Box<for <'a> MirMapPass<'a>>>>(&mut self, it: I) {
        self.plugin_passes.extend(it);
    }
}
