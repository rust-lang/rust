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

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum TransItem<'tcx> {
    Fn(Instance<'tcx>),
    Static(NodeId),
    GlobalAsm(NodeId),
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

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
