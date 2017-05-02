// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See [the README](README.md) for details on writing your own pass.

use hir;
use hir::def_id::DefId;
use hir::map::DefPathData;
use mir::{Mir, Promoted};
use ty::TyCtxt;
use std::rc::Rc;
use syntax::ast::NodeId;

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
    pub fn from_local_def_id(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> MirSource {
        let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
        Self::from_node(tcx, id)
    }

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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MirSuite(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MirPassIndex(pub usize);

/// A pass hook is invoked both before and after each pass executes.
/// This is primarily used to dump MIR for debugging.
///
/// You can tell whether this is before or after by inspecting the
/// `mir` parameter -- before the pass executes, it will be `None` (in
/// which case you can inspect the MIR from previous pass by executing
/// `mir_cx.read_previous_mir()`); after the pass executes, it will be
/// `Some()` with the result of the pass (in which case the output
/// from the previous pass is most likely stolen, so you would not
/// want to try and access it). If the pass is interprocedural, then
/// the hook will be invoked once per output.
pub trait PassHook {
    fn on_mir_pass<'a, 'tcx: 'a>(&self,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 suite: MirSuite,
                                 pass_num: MirPassIndex,
                                 pass_name: &str,
                                 source: MirSource,
                                 mir: &Mir<'tcx>,
                                 is_after: bool);
}

/// The full suite of types that identifies a particular
/// application of a pass to a def-id.
pub type PassId = (MirSuite, MirPassIndex, DefId);

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>);
}

/// A manager for MIR passes.
///
/// FIXME(#41712) -- it is unclear whether we should have this struct.
#[derive(Clone)]
pub struct Passes {
    pass_hooks: Vec<Rc<PassHook>>,
    suites: Vec<Vec<Rc<MirPass>>>,
}

/// The number of "pass suites" that we have:
///
/// - ready for constant evaluation
/// - unopt
/// - optimized
pub const MIR_SUITES: usize = 3;

/// Run the passes we need to do constant qualification and evaluation.
pub const MIR_CONST: MirSuite = MirSuite(0);

/// Run the passes we need to consider the MIR validated and ready for borrowck etc.
pub const MIR_VALIDATED: MirSuite = MirSuite(1);

/// Run the passes we need to consider the MIR *optimized*.
pub const MIR_OPTIMIZED: MirSuite = MirSuite(2);

impl<'a, 'tcx> Passes {
    pub fn new() -> Passes {
        Passes {
            pass_hooks: Vec::new(),
            suites: (0..MIR_SUITES).map(|_| Vec::new()).collect(),
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass<T: MirPass + 'static>(&mut self, suite: MirSuite, pass: T) {
        self.suites[suite.0].push(Rc::new(pass));
    }

    /// Pushes a pass hook.
    pub fn push_hook<T: PassHook + 'static>(&mut self, hook: T) {
        self.pass_hooks.push(Rc::new(hook));
    }

    pub fn passes(&self, suite: MirSuite) -> &[Rc<MirPass>] {
        &self.suites[suite.0]
    }

    pub fn hooks(&self) -> &[Rc<PassHook>] {
        &self.pass_hooks
    }
}
