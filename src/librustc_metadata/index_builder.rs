// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::tag_items_data_item;
use encoder::EncodeContext;
use index::IndexData;
use rbml::writer::Encoder;
use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc_data_structures::fnv::FnvHashMap;
use std::ops::{Deref, DerefMut};

/// Builder that can encode new items, adding them into the index.
/// Item encoding cannot be nested.
pub struct IndexBuilder<'a, 'tcx: 'a, 'encoder: 'a> {
    items: IndexData,
    builder: ItemContentBuilder<'a, 'tcx, 'encoder>,
}

/// Builder that can encode the content of items, but can't start a
/// new item itself. Most code is attached to here.
pub struct ItemContentBuilder<'a, 'tcx: 'a, 'encoder: 'a> {
    xrefs: FnvHashMap<XRef<'tcx>, u32>, // sequentially-assigned
    pub ecx: &'a EncodeContext<'a, 'tcx>,
    pub rbml_w: &'a mut Encoder<'encoder>,
}

/// "interned" entries referenced by id
#[derive(PartialEq, Eq, Hash)]
pub enum XRef<'tcx> { Predicate(ty::Predicate<'tcx>) }

impl<'a, 'tcx, 'encoder> IndexBuilder<'a, 'tcx, 'encoder> {
    pub fn new(ecx: &'a EncodeContext<'a, 'tcx>,
               rbml_w: &'a mut Encoder<'encoder>)
               -> Self {
        IndexBuilder {
            items: IndexData::new(ecx.tcx.map.num_local_def_ids()),
            builder: ItemContentBuilder {
                ecx: ecx,
                xrefs: FnvHashMap(),
                rbml_w: rbml_w,
            },
        }
    }

    /// Records that `id` is being emitted at the current offset.
    /// This data is later used to construct the item index in the
    /// metadata so we can quickly find the data for a given item.
    ///
    /// Returns a dep-graph task that you should keep live as long as
    /// the data for this item is being emitted.
    pub fn record<DATA>(&mut self,
                        id: DefId,
                        op: fn(&mut ItemContentBuilder<'a, 'tcx, 'encoder>, DATA),
                        data: DATA)
    {
        let position = self.rbml_w.mark_stable_position();
        self.items.record(id, position);
        let _task = self.ecx.tcx.dep_graph.in_task(DepNode::MetaData(id));
        self.rbml_w.start_tag(tag_items_data_item).unwrap();
        op(self, data);
        self.rbml_w.end_tag().unwrap();
    }

    pub fn into_fields(self) -> (IndexData, FnvHashMap<XRef<'tcx>, u32>) {
        (self.items, self.builder.xrefs)
    }
}

impl<'a, 'tcx, 'encoder> Deref for IndexBuilder<'a, 'tcx, 'encoder> {
    type Target = ItemContentBuilder<'a, 'tcx, 'encoder>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<'a, 'tcx, 'encoder> DerefMut for IndexBuilder<'a, 'tcx, 'encoder> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

impl<'a, 'tcx, 'encoder> ItemContentBuilder<'a, 'tcx, 'encoder> {
    pub fn ecx(&self) -> &'a EncodeContext<'a, 'tcx> {
        self.ecx
    }

    pub fn add_xref(&mut self, xref: XRef<'tcx>) -> u32 {
        let old_len = self.xrefs.len() as u32;
        *self.xrefs.entry(xref).or_insert(old_len)
    }
}

