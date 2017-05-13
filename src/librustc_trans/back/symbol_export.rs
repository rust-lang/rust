// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use context::SharedCrateContext;
use monomorphize::Instance;
use symbol_map::SymbolMap;
use util::nodemap::FxHashMap;
use rustc::hir::def_id::{DefId, CrateNum, LOCAL_CRATE};
use rustc::session::config;
use syntax::attr;
use trans_item::TransItem;

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
pub struct ExportedSymbols {
    exports: FxHashMap<CrateNum, Vec<(String, SymbolExportLevel)>>,
}

impl ExportedSymbols {

    pub fn empty() -> ExportedSymbols {
        ExportedSymbols {
            exports: FxHashMap(),
        }
    }

    pub fn compute_from<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                  symbol_map: &SymbolMap<'tcx>)
                                  -> ExportedSymbols {
        let mut local_crate: Vec<_> = scx
            .exported_symbols()
            .iter()
            .map(|&node_id| {
                scx.tcx().map.local_def_id(node_id)
            })
            .map(|def_id| {
                let name = symbol_for_def_id(scx, def_id, symbol_map);
                let export_level = export_level(scx, def_id);
                debug!("EXPORTED SYMBOL (local): {} ({:?})", name, export_level);
                (name, export_level)
            })
            .collect();

        if scx.sess().entry_fn.borrow().is_some() {
            local_crate.push(("main".to_string(), SymbolExportLevel::C));
        }

        if let Some(id) = scx.sess().derive_registrar_fn.get() {
            let svh = &scx.link_meta().crate_hash;
            let def_id = scx.tcx().map.local_def_id(id);
            let idx = def_id.index;
            let registrar = scx.sess().generate_derive_registrar_symbol(svh, idx);
            local_crate.push((registrar, SymbolExportLevel::C));
        }

        if scx.sess().crate_types.borrow().contains(&config::CrateTypeDylib) {
            local_crate.push((scx.metadata_symbol_name(),
                              SymbolExportLevel::Rust));
        }

        let mut exports = FxHashMap();
        exports.insert(LOCAL_CRATE, local_crate);

        for cnum in scx.sess().cstore.crates() {
            debug_assert!(cnum != LOCAL_CRATE);

            if scx.sess().cstore.plugin_registrar_fn(cnum).is_some() ||
               scx.sess().cstore.derive_registrar_fn(cnum).is_some() {
                continue;
            }

            let crate_exports = scx
                .sess()
                .cstore
                .exported_symbols(cnum)
                .iter()
                .map(|&def_id| {
                    let name = Instance::mono(scx, def_id).symbol_name(scx);
                    let export_level = export_level(scx, def_id);
                    debug!("EXPORTED SYMBOL (re-export): {} ({:?})", name, export_level);
                    (name, export_level)
                })
                .collect();

            exports.insert(cnum, crate_exports);
        }

        return ExportedSymbols {
            exports: exports
        };

        fn export_level(scx: &SharedCrateContext,
                        sym_def_id: DefId)
                        -> SymbolExportLevel {
            let attrs = scx.tcx().get_attrs(sym_def_id);
            if attr::contains_extern_indicator(scx.sess().diagnostic(), &attrs) {
                SymbolExportLevel::C
            } else {
                SymbolExportLevel::Rust
            }
        }
    }

    pub fn exported_symbols(&self,
                            cnum: CrateNum)
                            -> &[(String, SymbolExportLevel)] {
        match self.exports.get(&cnum) {
            Some(exports) => &exports[..],
            None => &[]
        }
    }

    pub fn for_each_exported_symbol<F>(&self,
                                       cnum: CrateNum,
                                       export_threshold: SymbolExportLevel,
                                       mut f: F)
        where F: FnMut(&str, SymbolExportLevel)
    {
        for &(ref name, export_level) in self.exported_symbols(cnum) {
            if is_below_threshold(export_level, export_threshold) {
                f(&name[..], export_level)
            }
        }
    }
}

pub fn crate_export_threshold(crate_type: config::CrateType)
                                     -> SymbolExportLevel {
    match crate_type {
        config::CrateTypeExecutable |
        config::CrateTypeStaticlib  |
        config::CrateTypeProcMacro  |
        config::CrateTypeCdylib     => SymbolExportLevel::C,
        config::CrateTypeRlib       |
        config::CrateTypeDylib      => SymbolExportLevel::Rust,
    }
}

pub fn crates_export_threshold(crate_types: &[config::CrateType])
                                      -> SymbolExportLevel {
    if crate_types.iter().any(|&crate_type| {
        crate_export_threshold(crate_type) == SymbolExportLevel::Rust
    }) {
        SymbolExportLevel::Rust
    } else {
        SymbolExportLevel::C
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

fn symbol_for_def_id<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                               def_id: DefId,
                               symbol_map: &SymbolMap<'tcx>)
                               -> String {
    // Just try to look things up in the symbol map. If nothing's there, we
    // recompute.
    if let Some(node_id) = scx.tcx().map.as_local_node_id(def_id) {
        if let Some(sym) = symbol_map.get(TransItem::Static(node_id)) {
            return sym.to_owned();
        }
    }

    let instance = Instance::mono(scx, def_id);

    symbol_map.get(TransItem::Fn(instance))
              .map(str::to_owned)
              .unwrap_or_else(|| instance.symbol_name(scx))
}
