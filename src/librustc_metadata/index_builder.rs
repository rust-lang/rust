// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use encoder::EncodeContext;
use index::IndexData;
use rbml::writer::Encoder;
use rustc::dep_graph::{DepNode, DepTask};
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc_data_structures::fnv::FnvHashMap;

pub struct IndexBuilder<'a, 'tcx: 'a> {
    ecx: &'a EncodeContext<'a, 'tcx>,
    items: IndexData,
    xrefs: FnvHashMap<XRef<'tcx>, u32>, // sequentially-assigned
}

/// "interned" entries referenced by id
#[derive(PartialEq, Eq, Hash)]
pub enum XRef<'tcx> { Predicate(ty::Predicate<'tcx>) }

impl<'a, 'tcx> IndexBuilder<'a, 'tcx> {
    pub fn new(ecx: &'a EncodeContext<'a, 'tcx>) -> Self {
        IndexBuilder {
            ecx: ecx,
            items: IndexData::new(ecx.tcx.map.num_local_def_ids()),
            xrefs: FnvHashMap()
        }
    }

    pub fn ecx(&self) -> &'a EncodeContext<'a, 'tcx> {
        self.ecx
    }

    /// Records that `id` is being emitted at the current offset.
    /// This data is later used to construct the item index in the
    /// metadata so we can quickly find the data for a given item.
    ///
    /// Returns a dep-graph task that you should keep live as long as
    /// the data for this item is being emitted.
    pub fn record(&mut self, id: DefId, rbml_w: &mut Encoder) -> DepTask<'a> {
        let position = rbml_w.mark_stable_position();
        self.items.record(id, position);
        self.ecx.tcx.dep_graph.in_task(DepNode::MetaData(id))
    }

    pub fn add_xref(&mut self, xref: XRef<'tcx>) -> u32 {
        let old_len = self.xrefs.len() as u32;
        *self.xrefs.entry(xref).or_insert(old_len)
    }

    pub fn into_fields(self) -> (IndexData, FnvHashMap<XRef<'tcx>, u32>) {
        (self.items, self.xrefs)
    }
}

