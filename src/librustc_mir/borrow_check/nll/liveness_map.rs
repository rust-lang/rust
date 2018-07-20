// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::{Mir, Local};
use util::liveness::LiveVariableMap;
use rustc_data_structures::indexed_vec::Idx;
use rustc::ty::TypeFoldable;

crate struct NllLivenessMap {
    pub from_local: IndexVec<Local, Option<LocalWithRegion>>,
    pub to_local: IndexVec<LocalWithRegion, Local>,

}

impl LiveVariableMap for NllLivenessMap {
    type LiveVar = LocalWithRegion;

    fn from_local(&self, local: Local) -> Option<Self::LiveVar> {
        self.from_local[local]
    }

    fn from_live_var(&self, local: Self::LiveVar) -> Local {
        self.to_local[local]
    }

    fn num_variables(&self) -> usize {
        self.to_local.len()
    }
}

impl NllLivenessMap {
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

newtype_index!(LocalWithRegion);