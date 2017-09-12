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
