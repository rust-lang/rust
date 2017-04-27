// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir;
use hir::def_id::{DefId, LOCAL_CRATE};
use hir::map::DefPathData;
use mir::{Mir, Promoted};
use ty::TyCtxt;
use std::rc::Rc;
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
/// pass will be invoked to process the MIR with the given `def_id`.
/// This lets you do things before we fetch the MIR itself.  You may
/// prefer `MirPass`.
pub trait DefIdPass {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          def_id: DefId);
}

impl<T: DefIdPass> Pass for T {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        DefIdPass::name(self)
    }

    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        for &def_id in tcx.mir_keys(LOCAL_CRATE).iter() {
            DefIdPass::run_pass(self, tcx, def_id);
        }
    }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass: DepGraphSafe {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>);
}

fn for_each_assoc_mir<'a, 'tcx, OP>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    def_id: DefId,
                                    mut op: OP)
    where OP: FnMut(MirSource, &mut Mir<'tcx>)
{
    let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
    let source = MirSource::from_node(tcx, id);
    let mir = &mut tcx.mir(def_id).borrow_mut();
    op(source, mir);

    for (promoted_index, promoted_mir) in mir.promoted.iter_enumerated_mut() {
        let promoted_source = MirSource::Promoted(id, promoted_index);
        op(promoted_source, promoted_mir);
    }
}

impl<T: MirPass> DefIdPass for T {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        MirPass::name(self)
    }

    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
        for_each_assoc_mir(tcx, def_id, |src, mir| MirPass::run_pass(self, tcx, src, mir));
    }
}

/// A manager for MIR passes.
#[derive(Clone)]
pub struct Passes {
    pass_hooks: Vec<Rc<PassHook>>,
    sets: Vec<PassSet>,
}

#[derive(Clone)]
struct PassSet {
    passes: Vec<Rc<Pass>>,
}

/// The number of "pass sets" that we have:
///
/// - ready for constant evaluation
/// - unopt
/// - optimized
pub const MIR_PASS_SETS: usize = 3;

/// Run the passes we need to do constant qualification and evaluation.
pub const MIR_CONST: usize = 0;

/// Run the passes we need to consider the MIR validated and ready for borrowck etc.
pub const MIR_VALIDATED: usize = 1;

/// Run the passes we need to consider the MIR *optimized*.
pub const MIR_OPTIMIZED: usize = 2;

impl<'a, 'tcx> Passes {
    pub fn new() -> Passes {
        Passes {
            pass_hooks: Vec::new(),
            sets: (0..MIR_PASS_SETS).map(|_| PassSet { passes: Vec::new() }).collect(),
        }
    }

    pub fn run_passes(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, set_index: usize) {
        let set = &self.sets[set_index];

        let start_num: usize = self.sets[..set_index].iter().map(|s| s.passes.len()).sum();

        // NB: passes are numbered from 1, since "construction" is zero.
        for (pass, pass_num) in set.passes.iter().zip(start_num + 1..) {
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
    pub fn push_pass<T: Pass + 'static>(&mut self, set: usize, pass: T) {
        self.sets[set].passes.push(Rc::new(pass));
    }

    /// Pushes a pass hook.
    pub fn push_hook<T: PassHook + 'static>(&mut self, hook: T) {
        self.pass_hooks.push(Rc::new(hook));
    }
}
