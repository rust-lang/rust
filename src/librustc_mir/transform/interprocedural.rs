// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::mir::transform::{PassId};
use rustc::ty::steal::Steal;
use rustc::ty::TyCtxt;
use rustc_data_structures::fx::FxHashMap;

/// When writing inter-procedural analyses etc, we need to read (and
/// steal) the MIR for a number of def-ids at once, not all of which
/// are local. This little cache code attempts to remember what you've
/// stolen and so forth. It is more of a placeholder meant to get
/// inlining up and going again, and is probably going to need heavy
/// revision as we scale up to more interesting optimizations.
pub struct InterproceduralCx<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    local_cache: FxHashMap<DefId, Mir<'tcx>>,
}

impl<'a, 'tcx> InterproceduralCx<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        InterproceduralCx {
            tcx: tcx,
            local_cache: FxHashMap::default(),
        }
    }

    pub fn into_local_mirs(self) -> Vec<(PassId, &'tcx Steal<Mir<'tcx>>)> {
        unimplemented!()
    }

    /// Ensures that the mir for `def_id` is available, if it can be
    /// made available.
    pub fn ensure_mir(&mut self, def_id: DefId) {
        if def_id.is_local() {
            self.ensure_mir_and_read(def_id);
        }
    }

    /// Ensures that the mir for `def_id` is available and returns it if possible;
    /// returns `None` if this is a cross-crate MIR that is not
    /// available from metadata.
    pub fn ensure_mir_and_read(&mut self, def_id: DefId) -> Option<&Mir<'tcx>> {
        if def_id.is_local() {
            Some(self.mir_mut(def_id))
        } else {
            self.tcx.maybe_item_mir(def_id)
        }
    }

    /// True if the local cache contains MIR for `def-id`.
    pub fn contains_mir(&self, def_id: DefId) -> bool {
        if def_id.is_local() {
            self.local_cache.contains_key(&def_id)
        } else {
            self.tcx.is_item_mir_available(def_id)
        }
    }

    /// Reads the MIR for `def-id`. If the MIR is local, this will
    /// panic if you have not previously invoked `ensure_mir`.
    pub fn mir(&self, def_id: DefId) -> Option<&Mir<'tcx>> {
        if def_id.is_local() {
            match self.local_cache.get(&def_id) {
                Some(p) => Some(p),
                None => {
                    panic!("MIR for local def-id `{:?}` not previously ensured", def_id)
                }
            }
        } else {
            self.tcx.maybe_item_mir(def_id)
        }
    }

    pub fn mir_mut(&mut self, def_id: DefId) -> &mut Mir<'tcx> {
        assert!(def_id.is_local(), "cannot get mutable mir of remote entry");
        unimplemented!()
    }
}
