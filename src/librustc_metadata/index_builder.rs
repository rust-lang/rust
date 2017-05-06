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
//! with an `EncodingContext` (`ecx`), which it encapsulates.
//! It has one main method, `record()`. You invoke `record`
//! like so to create a new `data_item` element in the list:
//!
//! ```
//! index.record(some_def_id, callback_fn, data)
//! ```
//!
//! What record will do is to (a) record the current offset, (b) emit
//! the `common::data_item` tag, and then call `callback_fn` with the
//! given data as well as the `EncodingContext`. Once `callback_fn`
//! returns, the `common::data_item` tag will be closed.
//!
//! `EncodingContext` does not offer the `record` method, so that we
//! can ensure that `common::data_item` elements are never nested.
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

use encoder::EncodeContext;
use index::Index;
use schema::*;

use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ich::{StableHashingContext, Fingerprint};
use rustc::middle::cstore::EncodedMetadataHash;
use rustc::ty::TyCtxt;
use syntax::ast;

use std::ops::{Deref, DerefMut};

use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use rustc_serialize::Encodable;

/// Builder that can encode new items, adding them into the index.
/// Item encoding cannot be nested.
pub struct IndexBuilder<'a, 'b: 'a, 'tcx: 'b> {
    items: Index,
    pub ecx: &'a mut EncodeContext<'b, 'tcx>,
}

impl<'a, 'b, 'tcx> Deref for IndexBuilder<'a, 'b, 'tcx> {
    type Target = EncodeContext<'b, 'tcx>;
    fn deref(&self) -> &Self::Target {
        self.ecx
    }
}

impl<'a, 'b, 'tcx> DerefMut for IndexBuilder<'a, 'b, 'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ecx
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    pub fn new(ecx: &'a mut EncodeContext<'b, 'tcx>) -> Self {
        IndexBuilder {
            items: Index::new(ecx.tcx.hir.definitions().def_index_counts_lo_hi()),
            ecx: ecx,
        }
    }

    /// Emit the data for a def-id to the metadata. The function to
    /// emit the data is `op`, and it will be given `data` as
    /// arguments. This `record` function will call `op` to generate
    /// the `Entry` (which may point to other encoded information)
    /// and will then record the `Lazy<Entry>` for use in the index.
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
    pub fn record<'x, DATA>(&'x mut self,
                            id: DefId,
                            op: fn(&mut EntryBuilder<'x, 'b, 'tcx>, DATA) -> Entry<'tcx>,
                            data: DATA)
        where DATA: DepGraphRead
    {
        assert!(id.is_local());
        let tcx: TyCtxt<'b, 'tcx, 'tcx> = self.ecx.tcx;

        // We don't track this since we are explicitly computing the incr. comp.
        // hashes anyway. In theory we could do some tracking here and use it to
        // avoid rehashing things (and instead cache the hashes) but it's
        // unclear whether that would be a win since hashing is cheap enough.
        let _task = tcx.dep_graph.in_ignore();

        let compute_ich = (tcx.sess.opts.debugging_opts.query_dep_graph ||
                           tcx.sess.opts.debugging_opts.incremental_cc) &&
                           tcx.sess.opts.build_dep_graph();

        let ecx: &'x mut EncodeContext<'b, 'tcx> = &mut *self.ecx;
        let mut entry_builder = EntryBuilder {
            tcx: tcx,
            ecx: ecx,
            hcx: if compute_ich {
                Some((StableHashingContext::new(tcx), StableHasher::new()))
            } else {
                None
            }
        };

        let entry = op(&mut entry_builder, data);

        if let Some((ref mut hcx, ref mut hasher)) = entry_builder.hcx {
            entry.hash_stable(hcx, hasher);
        }

        let entry = entry_builder.ecx.lazy(&entry);
        entry_builder.finish(id);
        self.items.record(id, entry);
    }

    pub fn into_items(self) -> Index {
        self.items
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
    fn read(&self, _tcx: TyCtxt) {}
}

impl DepGraphRead for ast::NodeId {
    fn read(&self, _tcx: TyCtxt) {}
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
read_tuple!(A, B);
read_tuple!(A, B, C);

macro_rules! read_hir {
    ($t:ty) => {
        impl<'tcx> DepGraphRead for &'tcx $t {
            fn read(&self, tcx: TyCtxt) {
                tcx.hir.read(self.id);
            }
        }
    }
}
read_hir!(hir::Item);
read_hir!(hir::ImplItem);
read_hir!(hir::TraitItem);
read_hir!(hir::ForeignItem);
read_hir!(hir::MacroDef);

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
    fn read(&self, _tcx: TyCtxt) {}
}

/// Newtype that can be used to package up misc data extracted from a
/// HIR node that doesn't carry its own id. This will allow an
/// arbitrary `T` to be passed in, but register a read on the given
/// node-id.
pub struct FromId<T>(pub ast::NodeId, pub T);

impl<T> DepGraphRead for FromId<T> {
    fn read(&self, tcx: TyCtxt) {
        tcx.hir.read(self.0);
    }
}

pub struct EntryBuilder<'a, 'b: 'a, 'tcx: 'b> {
    pub tcx: TyCtxt<'b, 'tcx, 'tcx>,
    ecx: &'a mut EncodeContext<'b, 'tcx>,
    hcx: Option<(StableHashingContext<'b, 'tcx>, StableHasher<Fingerprint>)>,
}

impl<'a, 'b: 'a, 'tcx: 'b> EntryBuilder<'a, 'b, 'tcx> {

    pub fn finish(self, def_id: DefId) {
        if let Some((_, hasher)) = self.hcx {
            let hash = hasher.finish();
            self.ecx.metadata_hashes.push(EncodedMetadataHash {
                def_index: def_id.index,
                hash: hash,
            });
        }
    }

    pub fn lazy<T>(&mut self, value: &T) -> Lazy<T>
        where T: Encodable + HashStable<StableHashingContext<'b, 'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            value.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy(value)
    }

    pub fn lazy_seq<I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = T>,
              T: Encodable + HashStable<StableHashingContext<'b, 'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            let iter = iter.into_iter();
            let (lower_bound, upper_bound) = iter.size_hint();

            if upper_bound == Some(lower_bound) {
                lower_bound.hash_stable(hcx, hasher);
                let mut num_items_hashed = 0;
                let ret = self.ecx.lazy_seq(iter.inspect(|item| {
                    item.hash_stable(hcx, hasher);
                    num_items_hashed += 1;
                }));

                // Sometimes items in a sequence are filtered out without being
                // hashed (e.g. for &[ast::Attribute]) and this code path cannot
                // handle that correctly, so we want to make sure we didn't hit
                // it by accident.
                if lower_bound != num_items_hashed {
                    bug!("Hashed a different number of items ({}) than expected ({})",
                         num_items_hashed,
                         lower_bound);
                }
                debug!("metadata-hash: {:?}", hasher);
                ret
            } else {
                // Collect into a vec so we know the length of the sequence
                let items: AccumulateVec<[T; 32]> = iter.collect();
                items.hash_stable(hcx, hasher);
                debug!("metadata-hash: {:?}", hasher);
                self.ecx.lazy_seq(items)
            }
        } else {
            self.ecx.lazy_seq(iter)
        }
    }

    pub fn lazy_seq_from_slice<T>(&mut self, slice: &[T]) -> LazySeq<T>
        where T: Encodable + HashStable<StableHashingContext<'b, 'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            slice.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy_seq_ref(slice.iter())
    }

    pub fn lazy_seq_ref_from_slice<T>(&mut self, slice: &[&T]) -> LazySeq<T>
        where T: Encodable + HashStable<StableHashingContext<'b, 'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            slice.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy_seq_ref(slice.iter().map(|x| *x))
    }
}
