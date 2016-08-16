// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Builder types for generating the "item data" section of the
//! metadata. This section winds up looking like this:
//!
//! ```
//! <common::data> // big list of item-like things...
//!    <common::data_item> // ...for most def-ids, there is an entry.
//!    </common::data_item>
//! </common::data>
//! ```
//!
//! As we generate this listing, we collect the offset of each
//! `data_item` entry and store it in an index. Then, when we load the
//! metadata, we can skip right to the metadata for a particular item.
//!
//! In addition to the offset, we need to track the data that was used
//! to generate the contents of each `data_item`. This is so that we
//! can figure out which HIR nodes contributed to that data for
//! incremental compilation purposes.
//!
//! The `IndexBuilder` facilitates both of these. It is created
//! with an RBML encoder isntance (`rbml_w`) along with an
//! `EncodingContext` (`ecx`), which it encapsulates. It has one main
//! method, `record()`. You invoke `record` like so to create a new
//! `data_item` element in the list:
//!
//! ```
//! index.record(some_def_id, callback_fn, data)
//! ```
//!
//! What record will do is to (a) record the current offset, (b) emit
//! the `common::data_item` tag, and then call `callback_fn` with the
//! given data as well as an `ItemContentBuilder`. Once `callback_fn`
//! returns, the `common::data_item` tag will be closed.
//!
//! The `ItemContentBuilder` is another type that just offers access
//! to the `ecx` and `rbml_w` that were given in, as well as
//! maintaining a list of `xref` instances, which are used to extract
//! common data so it is not re-serialized.
//!
//! `ItemContentBuilder` is a distinct type which does not offer the
//! `record` method, so that we can ensure that `common::data_item` elements
//! are never nested.
//!
//! In addition, while the `callback_fn` is executing, we will push a
//! task `MetaData(some_def_id)`, which can then observe the
//! reads/writes that occur in the task. For this reason, the `data`
//! argument that is given to the `callback_fn` must implement the
//! trait `DepGraphRead`, which indicates how to register reads on the
//! data in this new task (note that many types of data, such as
//! `DefId`, do not currently require any reads to be registered,
//! since they are not derived from a HIR node). This is also why we
//! give a callback fn, rather than taking a closure: it allows us to
//! easily control precisely what data is given to that fn.

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

    pub fn ecx(&self) -> &'a EncodeContext<'a, 'tcx> {
        self.builder.ecx()
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
        let position = self.builder.rbml_w.mark_stable_position();
        self.items.record(id, position);
        let _task = self.ecx().tcx.dep_graph.in_task(DepNode::MetaData(id));
        self.builder.rbml_w.start_tag(tag_items_data_item).unwrap();
        data.read(self.ecx().tcx);
        op(&mut self.builder, data);
        self.builder.rbml_w.end_tag().unwrap();
    }

    pub fn into_fields(self) -> (IndexData, FnvHashMap<XRef<'tcx>, u32>) {
        (self.items, self.builder.xrefs)
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

/// Trait used for data that can be passed from outside a dep-graph
/// task.  The data must either be of some safe type, such as a
/// `DefId` index, or implement the `read` method so that it can add
/// a read of whatever dep-graph nodes are appropriate.
pub trait DepGraphRead {
    fn read(&self, tcx: TyCtxt);
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

/// Leaks access to a value of type T without any tracking. This is
/// suitable for ambiguous types like `usize`, which *could* represent
/// tracked data (e.g., if you read it out of a HIR node) or might not
/// (e.g., if it's an index). Adding in an `Untracked` is an
/// assertion, essentially, that the data does not need to be tracked
/// (or that read edges will be added by some other way).
///
/// A good idea is to add to each use of `Untracked` an explanation of
/// why this value is ok.
pub struct Untracked<T>(pub T);

impl<T> DepGraphRead for Untracked<T> {
    fn read(&self, _tcx: TyCtxt) { }
}

/// Newtype that can be used to package up misc data extracted from a
/// HIR node that doesn't carry its own id. This will allow an
/// arbitrary `T` to be passed in, but register a read on the given
/// node-id.
pub struct FromId<T>(pub ast::NodeId, pub T);

impl<T> DepGraphRead for FromId<T> {
    fn read(&self, tcx: TyCtxt) {
        tcx.map.read(self.0);
    }
}
