// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use monomorphize::Instance;
use rustc::util::nodemap::{FxHashMap, NodeSet};
use rustc::hir::def_id::{DefId, CrateNum, LOCAL_CRATE, INVALID_CRATE, CRATE_DEF_INDEX};
use rustc::session::config;
use rustc::ty::TyCtxt;
use rustc_allocator::ALLOCATOR_METHODS;
use syntax::attr;

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
    pub fn empty() -> ExportedSymbols {
        ExportedSymbols {
            export_threshold: SymbolExportLevel::C,
            exports: FxHashMap(),
            local_exports: NodeSet(),
        }
    }

    pub fn compute<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             local_exported_symbols: &NodeSet)
                             -> ExportedSymbols {
        let export_threshold = crates_export_threshold(&tcx.sess.crate_types.borrow());

        let mut local_crate: Vec<_> = local_exported_symbols
            .iter()
            .map(|&node_id| {
                tcx.hir.local_def_id(node_id)
            })
            .map(|def_id| {
                let name = tcx.symbol_name(Instance::mono(tcx, def_id));
                let export_level = export_level(tcx, def_id);
                debug!("EXPORTED SYMBOL (local): {} ({:?})", name, export_level);
                (str::to_owned(&name), def_id, export_level)
            })
            .collect();

        let mut local_exports = local_crate
            .iter()
            .filter_map(|&(_, def_id, level)| {
                if is_below_threshold(level, export_threshold) {
                    tcx.hir.as_local_node_id(def_id)
                } else {
                    None
                }
            })
            .collect::<NodeSet>();

        const INVALID_DEF_ID: DefId = DefId {
            krate: INVALID_CRATE,
            index: CRATE_DEF_INDEX,
        };

        if let Some(_) = *tcx.sess.entry_fn.borrow() {
            local_crate.push(("main".to_string(),
                              INVALID_DEF_ID,
                              SymbolExportLevel::C));
        }

        if tcx.sess.allocator_kind.get().is_some() {
            for method in ALLOCATOR_METHODS {
                local_crate.push((format!("__rust_{}", method.name),
                                  INVALID_DEF_ID,
                                  SymbolExportLevel::Rust));
            }
        }

        if let Some(id) = tcx.sess.derive_registrar_fn.get() {
            let def_id = tcx.hir.local_def_id(id);
            let idx = def_id.index;
            let disambiguator = tcx.sess.local_crate_disambiguator();
            let registrar = tcx.sess.generate_derive_registrar_symbol(disambiguator, idx);
            local_crate.push((registrar, def_id, SymbolExportLevel::C));
            local_exports.insert(id);
        }

        if tcx.sess.crate_types.borrow().contains(&config::CrateTypeDylib) {
            local_crate.push((metadata_symbol_name(tcx),
                              INVALID_DEF_ID,
                              SymbolExportLevel::Rust));
        }

        let mut exports = FxHashMap();
        exports.insert(LOCAL_CRATE, local_crate);

        for cnum in tcx.sess.cstore.crates() {
            debug_assert!(cnum != LOCAL_CRATE);

            // If this crate is a plugin and/or a custom derive crate, then
            // we're not even going to link those in so we skip those crates.
            if tcx.sess.cstore.plugin_registrar_fn(cnum).is_some() ||
               tcx.sess.cstore.derive_registrar_fn(cnum).is_some() {
                continue;
            }

            // Check to see if this crate is a "special runtime crate". These
            // crates, implementation details of the standard library, typically
            // have a bunch of `pub extern` and `#[no_mangle]` functions as the
            // ABI between them. We don't want their symbols to have a `C`
            // export level, however, as they're just implementation details.
            // Down below we'll hardwire all of the symbols to the `Rust` export
            // level instead.
            let special_runtime_crate =
                tcx.is_panic_runtime(cnum) || tcx.is_compiler_builtins(cnum);

            let crate_exports = tcx
                .sess
                .cstore
                .exported_symbols(cnum)
                .iter()
                .map(|&def_id| {
                    let name = tcx.symbol_name(Instance::mono(tcx, def_id));
                    let export_level = if special_runtime_crate {
                        // We can probably do better here by just ensuring that
                        // it has hidden visibility rather than public
                        // visibility, as this is primarily here to ensure it's
                        // not stripped during LTO.
                        //
                        // In general though we won't link right if these
                        // symbols are stripped, and LTO currently strips them.
                        if &*name == "rust_eh_personality" ||
                           &*name == "rust_eh_register_frames" ||
                           &*name == "rust_eh_unregister_frames" {
                            SymbolExportLevel::C
                        } else {
                            SymbolExportLevel::Rust
                        }
                    } else {
                        export_level(tcx, def_id)
                    };
                    debug!("EXPORTED SYMBOL (re-export): {} ({:?})", name, export_level);
                    (str::to_owned(&name), def_id, export_level)
                })
                .collect();

            exports.insert(cnum, crate_exports);
        }

        return ExportedSymbols {
            export_threshold,
            exports,
            local_exports,
        };

        fn export_level(tcx: TyCtxt,
                        sym_def_id: DefId)
                        -> SymbolExportLevel {
            let attrs = tcx.get_attrs(sym_def_id);
            if attr::contains_extern_indicator(tcx.sess.diagnostic(), &attrs) {
                SymbolExportLevel::C
            } else {
                SymbolExportLevel::Rust
            }
        }
    }

    pub fn local_exports(&self) -> &NodeSet {
        &self.local_exports
    }

    pub fn exported_symbols(&self,
                            cnum: CrateNum)
                            -> &[(String, DefId, SymbolExportLevel)] {
        match self.exports.get(&cnum) {
            Some(exports) => exports,
            None => &[]
        }
    }

    pub fn for_each_exported_symbol<F>(&self,
                                       cnum: CrateNum,
                                       mut f: F)
        where F: FnMut(&str, DefId, SymbolExportLevel)
    {
        for &(ref name, def_id, export_level) in self.exported_symbols(cnum) {
            if is_below_threshold(export_level, self.export_threshold) {
                f(&name, def_id, export_level)
            }
        }
    }
}

pub fn metadata_symbol_name(tcx: TyCtxt) -> String {
    format!("rust_metadata_{}_{}",
            tcx.crate_name(LOCAL_CRATE),
            tcx.crate_disambiguator(LOCAL_CRATE))
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
