// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For the NLL computation, we need to compute liveness, but only for those
//! local variables whose types contain regions. The others are not of interest
//! to us. This file defines a new index type (LocalWithRegion) that indexes into
//! a list of "variables whose type contain regions". It also defines a map from
//! Local to LocalWithRegion and vice versa -- this map can be given to the
//! liveness code so that it only operates over variables with regions in their
//! types, instead of all variables.

use rustc::ty::TypeFoldable;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::{Mir, Local};
use util::liveness::LiveVariableMap;

use rustc_data_structures::indexed_vec::Idx;

/// Map between Local and LocalWithRegion indices: this map is supplied to the
/// liveness code so that it will only analyze those variables whose types
/// contain regions.
crate struct NllLivenessMap {
    /// For each local variable, contains either None (if the type has no regions)
    /// or Some(i) with a suitable index.
    pub from_local: IndexVec<Local, Option<LocalWithRegion>>,
    /// For each LocalWithRegion, maps back to the original Local index.
    pub to_local: IndexVec<LocalWithRegion, Local>,

}

impl LiveVariableMap for NllLivenessMap {

    fn from_local(&self, local: Local) -> Option<Self::LiveVar> {
        self.from_local[local]
    }

    type LiveVar = LocalWithRegion;

    fn from_live_var(&self, local: Self::LiveVar) -> Local {
        self.to_local[local]
    }

    fn num_variables(&self) -> usize {
        self.to_local.len()
    }
}

impl NllLivenessMap {
    /// Iterates over the variables in Mir and assigns each Local whose type contains
    /// regions a LocalWithRegion index. Returns a map for converting back and forth.
    pub fn compute(mir: &Mir) -> Self {
        let mut to_local = IndexVec::default();
        let from_local: IndexVec<Local,Option<_>> = mir
            .local_decls
            .iter_enumerated()
            .map(|(local, local_decl)| {
                if local_decl.ty.has_free_regions() {
                    Some(to_local.push(local))
                }
                    else {
                        None
                    }
            }).collect();

        Self { from_local, to_local }
    }
}

/// Index given to each local variable whose type contains a region.
newtype_index!(LocalWithRegion);
