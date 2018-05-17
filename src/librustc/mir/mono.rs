// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use syntax::ast::NodeId;
use syntax::symbol::InternedString;
use ty::{Instance, TyCtxt};
use util::nodemap::FxHashMap;
use rustc_data_structures::base_n;
use rustc_data_structures::stable_hasher::{HashStable, StableHasherResult,
                                           StableHasher};
use ich::{Fingerprint, StableHashingContext, NodeIdHashingMode};
use std::hash::Hash;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum MonoItem<'tcx> {
    Fn(Instance<'tcx>),
    Static(DefId),
    GlobalAsm(NodeId),
}

impl<'tcx> MonoItem<'tcx> {
    pub fn size_estimate<'a>(&self, tcx: &TyCtxt<'a, 'tcx, 'tcx>) -> usize {
        match *self {
            MonoItem::Fn(instance) => {
                // Estimate the size of a function based on how many statements
                // it contains.
                tcx.instance_def_size_estimate(instance.def)
            },
            // Conservatively estimate the size of a static declaration
            // or assembly to be 1.
            MonoItem::Static(_) | MonoItem::GlobalAsm(_) => 1,
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for MonoItem<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                           hcx: &mut StableHashingContext<'a>,
                                           hasher: &mut StableHasher<W>) {
        ::std::mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            MonoItem::Fn(ref instance) => {
                instance.hash_stable(hcx, hasher);
            }
            MonoItem::Static(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            MonoItem::GlobalAsm(node_id) => {
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
    items: FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)>,
    size_estimate: Option<usize>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
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
            size_estimate: None,
        }
    }

    pub fn name(&self) -> &InternedString {
        &self.name
    }

    pub fn set_name(&mut self, name: InternedString) {
        self.name = name;
    }

    pub fn items(&self) -> &FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)> {
        &self.items
    }

    pub fn items_mut(&mut self)
        -> &mut FxHashMap<MonoItem<'tcx>, (Linkage, Visibility)>
    {
        &mut self.items
    }

    pub fn mangle_name(human_readable_name: &str) -> String {
        // We generate a 80 bit hash from the name. This should be enough to
        // avoid collisions and is still reasonably short for filenames.
        let mut hasher = StableHasher::new();
        human_readable_name.hash(&mut hasher);
        let hash: u128 = hasher.finish();
        let hash = hash & ((1u128 << 80) - 1);
        base_n::encode(hash, base_n::CASE_INSENSITIVE)
    }

    pub fn estimate_size<'a>(&mut self, tcx: &TyCtxt<'a, 'tcx, 'tcx>) {
        // Estimate the size of a codegen unit as (approximately) the number of MIR
        // statements it corresponds to.
        self.size_estimate = Some(self.items.keys().map(|mi| mi.size_estimate(tcx)).sum());
    }

    pub fn size_estimate(&self) -> usize {
        // Should only be called if `estimate_size` has previously been called.
        self.size_estimate.expect("estimate_size must be called before getting a size_estimate")
    }

    pub fn modify_size_estimate(&mut self, delta: usize) {
        assert!(self.size_estimate.is_some());
        if let Some(size_estimate) = self.size_estimate {
            self.size_estimate = Some(size_estimate + delta);
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for CodegenUnit<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                           hcx: &mut StableHashingContext<'a>,
                                           hasher: &mut StableHasher<W>) {
        let CodegenUnit {
            ref items,
            name,
            // The size estimate is not relevant to the hash
            size_estimate: _,
        } = *self;

        name.hash_stable(hcx, hasher);

        let mut items: Vec<(Fingerprint, _)> = items.iter().map(|(mono_item, &attrs)| {
            let mut hasher = StableHasher::new();
            mono_item.hash_stable(hcx, &mut hasher);
            let mono_item_fingerprint = hasher.finish();
            (mono_item_fingerprint, attrs)
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
