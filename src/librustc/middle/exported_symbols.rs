// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use util::nodemap::{FxHashMap, NodeSet};
use hir::def_id::{DefId, CrateNum};

/// The SymbolExportLevel of a symbols specifies from which kinds of crates
/// the symbol will be exported. `C` symbols will be exported from any
/// kind of crate, including cdylibs which export very few things.
/// `Rust` will only be exported if the crate produced is a Rust
/// dylib.
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SymbolExportLevel {
    C,
    Rust,
}

/// The set of symbols exported from each crate in the crate graph.
#[derive(Debug)]
pub struct ExportedSymbols {
    pub export_threshold: SymbolExportLevel,
    exports: FxHashMap<CrateNum, Vec<(String, DefId, SymbolExportLevel)>>,
    local_exports: NodeSet,
}

impl ExportedSymbols {
    pub fn new(export_threshold: SymbolExportLevel,
               exports: FxHashMap<CrateNum, Vec<(String, DefId, SymbolExportLevel)>>,
               local_exports: NodeSet) -> ExportedSymbols {
        ExportedSymbols {
            export_threshold,
            exports,
            local_exports,
        }
    }

    pub fn local_exports(&self) -> &NodeSet {
        &self.local_exports
    }

    pub fn exported_symbols(&self, cnum: CrateNum)
        -> &[(String, DefId, SymbolExportLevel)]
    {
        match self.exports.get(&cnum) {
            Some(exports) => exports,
            None => &[]
        }
    }

    pub fn for_each_exported_symbol<F>(&self, cnum: CrateNum, mut f: F)
        where F: FnMut(&str, DefId, SymbolExportLevel)
    {
        for &(ref name, def_id, export_level) in self.exported_symbols(cnum) {
            if is_below_threshold(export_level, self.export_threshold) {
                f(&name, def_id, export_level)
            }
        }
    }
}

pub fn is_below_threshold(level: SymbolExportLevel,
                          threshold: SymbolExportLevel)
                          -> bool {
    if threshold == SymbolExportLevel::Rust {
        // We export everything from Rust dylibs
        true
    } else {
        level == SymbolExportLevel::C
    }
}
