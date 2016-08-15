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
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::fnv::FnvHashMap;
use syntax::ast;
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

    /// Emit the data for a def-id to the metadata. The function to
    /// emit the data is `op`, and it will be given `data` as
    /// arguments. This `record` function will start/end an RBML tag
    /// and record the current offset for use in the index, calling
    /// `op` to generate the data in the RBML tag.
    ///
    /// In addition, it will setup a dep-graph task to track what data
    /// `op` accesses to generate the metadata, which is later used by
    /// incremental compilation to compute a hash for the metadata and
    /// track changes.
    ///
    /// The reason that `op` is a function pointer, and not a closure,
    /// is that we want to be able to completely track all data it has
    /// access to, so that we can be sure that `DATA: DepGraphRead`
    /// holds, and that it is therefore not gaining "secret" access to
    /// bits of HIR or other state that would not be trackd by the
    /// content system.
    pub fn record<DATA>(&mut self,
                        id: DefId,
                        op: fn(&mut ItemContentBuilder<'a, 'tcx, 'encoder>, DATA),
                        data: DATA)
        where DATA: DepGraphRead
    {
        let position = self.rbml_w.mark_stable_position();
        self.items.record(id, position);
        let _task = self.ecx.tcx.dep_graph.in_task(DepNode::MetaData(id));
        self.rbml_w.start_tag(tag_items_data_item).unwrap();
        data.read(self.ecx.tcx);
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

/// Trait that registers reads for types that are tracked in the
/// dep-graph. Mostly it is implemented for indices like DefId etc
/// which do not need to register a read.
pub trait DepGraphRead {
    fn read(&self, tcx: TyCtxt);
}

impl DepGraphRead for usize {
    fn read(&self, _tcx: TyCtxt) { }
}

impl DepGraphRead for DefId {
    fn read(&self, _tcx: TyCtxt) { }
}

impl DepGraphRead for ast::NodeId {
    fn read(&self, _tcx: TyCtxt) { }
}

impl<T> DepGraphRead for Option<T>
    where T: DepGraphRead
{
    fn read(&self, tcx: TyCtxt) {
        match *self {
            Some(ref v) => v.read(tcx),
            None => (),
        }
    }
}

impl<T> DepGraphRead for [T]
    where T: DepGraphRead
{
    fn read(&self, tcx: TyCtxt) {
        for i in self {
            i.read(tcx);
        }
    }
}

macro_rules! read_tuple {
    ($($name:ident),*) => {
        impl<$($name),*> DepGraphRead for ($($name),*)
            where $($name: DepGraphRead),*
        {
            #[allow(non_snake_case)]
            fn read(&self, tcx: TyCtxt) {
                let &($(ref $name),*) = self;
                $($name.read(tcx);)*
            }
        }
    }
}
read_tuple!(A,B);
read_tuple!(A,B,C);

macro_rules! read_hir {
    ($t:ty) => {
        impl<'tcx> DepGraphRead for &'tcx $t {
            fn read(&self, tcx: TyCtxt) {
                tcx.map.read(self.id);
            }
        }
    }
}
read_hir!(hir::Item);
read_hir!(hir::ImplItem);
read_hir!(hir::TraitItem);
read_hir!(hir::ForeignItem);

/// You can use `FromId(X, ...)` to indicate that `...` came from node
/// `X`; so we will add a read from the suitable `Hir` node.
pub struct FromId<T>(pub ast::NodeId, pub T);

impl<T> DepGraphRead for FromId<T> {
    fn read(&self, tcx: TyCtxt) {
        tcx.map.read(self.0);
    }
}
