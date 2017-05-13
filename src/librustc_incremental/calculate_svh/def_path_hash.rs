// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::util::nodemap::DefIdMap;

pub struct DefPathHashes<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    data: DefIdMap<u64>,
}

impl<'a, 'tcx> DefPathHashes<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        DefPathHashes {
            tcx: tcx,
            data: DefIdMap()
        }
    }

    pub fn hash(&mut self, def_id: DefId) -> u64 {
        let tcx = self.tcx;
        *self.data.entry(def_id)
                  .or_insert_with(|| {
                      let def_path = tcx.def_path(def_id);
                      def_path.deterministic_hash(tcx)
                  })
    }
}
