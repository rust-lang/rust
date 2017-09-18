// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::NodeId;
use syntax::symbol::InternedString;
use ty::Instance;
use util::nodemap::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasherResult,
                                           StableHasher};
use ich::{Fingerprint, StableHashingContext, NodeIdHashingMode};

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum TransItem<'tcx> {
    Fn(Instance<'tcx>),
    Static(NodeId),
    GlobalAsm(NodeId),
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for TransItem<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                           hcx: &mut StableHashingContext<'tcx>,
                                           hasher: &mut StableHasher<W>) {
        ::std::mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            TransItem::Fn(ref instance) => {
                instance.hash_stable(hcx, hasher);
            }
            TransItem::Static(node_id)    |
            TransItem::GlobalAsm(node_id) => {
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                    node_id.hash_stable(hcx, hasher);
                })
            }
        }
    }
}

pub struct CodegenUnit<'tcx> {
    /// A name for this CGU. Incremental compilation requires that
    /// name be unique amongst **all** crates.  Therefore, it should
    /// contain something unique to this crate (e.g., a module path)
    /// as well as the crate name and disambiguator.
    name: InternedString,
    items: FxHashMap<TransItem<'tcx>, (Linkage, Visibility)>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Linkage {
    External,
    AvailableExternally,
    LinkOnceAny,
    LinkOnceODR,
    WeakAny,
    WeakODR,
    Appending,
    Internal,
    Private,
    ExternalWeak,
    Common,
}

impl_stable_hash_for!(enum self::Linkage {
    External,
    AvailableExternally,
    LinkOnceAny,
    LinkOnceODR,
    WeakAny,
    WeakODR,
    Appending,
    Internal,
    Private,
    ExternalWeak,
    Common
});

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

impl_stable_hash_for!(enum self::Visibility {
    Default,
    Hidden,
    Protected
});

impl<'tcx> CodegenUnit<'tcx> {
    pub fn new(name: InternedString) -> CodegenUnit<'tcx> {
        CodegenUnit {
            name: name,
            items: FxHashMap(),
        }
    }

    pub fn name(&self) -> &InternedString {
        &self.name
    }

    pub fn set_name(&mut self, name: InternedString) {
        self.name = name;
    }

    pub fn items(&self) -> &FxHashMap<TransItem<'tcx>, (Linkage, Visibility)> {
        &self.items
    }

    pub fn items_mut(&mut self)
        -> &mut FxHashMap<TransItem<'tcx>, (Linkage, Visibility)>
    {
        &mut self.items
    }
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for CodegenUnit<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                           hcx: &mut StableHashingContext<'tcx>,
                                           hasher: &mut StableHasher<W>) {
        let CodegenUnit {
            ref items,
            name,
        } = *self;

        name.hash_stable(hcx, hasher);

        let mut items: Vec<(Fingerprint, _)> = items.iter().map(|(trans_item, &attrs)| {
            let mut hasher = StableHasher::new();
            trans_item.hash_stable(hcx, &mut hasher);
            let trans_item_fingerprint = hasher.finish();
            (trans_item_fingerprint, attrs)
        }).collect();

        items.sort_unstable_by_key(|i| i.0);
        items.hash_stable(hcx, hasher);
    }
}

#[derive(Clone, Default)]
pub struct Stats {
    pub n_glues_created: usize,
    pub n_null_glues: usize,
    pub n_real_glues: usize,
    pub n_fns: usize,
    pub n_inlines: usize,
    pub n_closures: usize,
    pub n_llvm_insns: usize,
    pub llvm_insns: FxHashMap<String, usize>,
    // (ident, llvm-instructions)
    pub fn_stats: Vec<(String, usize)>,
}

impl_stable_hash_for!(struct self::Stats {
    n_glues_created,
    n_null_glues,
    n_real_glues,
    n_fns,
    n_inlines,
    n_closures,
    n_llvm_insns,
    llvm_insns,
    fn_stats
});

impl Stats {
    pub fn extend(&mut self, stats: Stats) {
        self.n_glues_created += stats.n_glues_created;
        self.n_null_glues += stats.n_null_glues;
        self.n_real_glues += stats.n_real_glues;
        self.n_fns += stats.n_fns;
        self.n_inlines += stats.n_inlines;
        self.n_closures += stats.n_closures;
        self.n_llvm_insns += stats.n_llvm_insns;

        for (k, v) in stats.llvm_insns {
            *self.llvm_insns.entry(k).or_insert(0) += v;
        }
        self.fn_stats.extend(stats.fn_stats);
    }
}

