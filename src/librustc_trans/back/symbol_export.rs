// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;
use std::sync::Arc;

use base;
use monomorphize::Instance;
use rustc::hir::def_id::CrateNum;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::exported_symbols::SymbolExportLevel;
use rustc::session::config;
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc::util::nodemap::FxHashMap;
use rustc_allocator::ALLOCATOR_METHODS;

pub type ExportedSymbols = FxHashMap<
    CrateNum,
    Arc<Vec<(String, Option<DefId>, SymbolExportLevel)>>,
>;

pub fn threshold(tcx: TyCtxt) -> SymbolExportLevel {
    crates_export_threshold(&tcx.sess.crate_types.borrow())
}

pub fn metadata_symbol_name(tcx: TyCtxt) -> String {
    format!("rust_metadata_{}_{}",
            tcx.crate_name(LOCAL_CRATE),
            tcx.crate_disambiguator(LOCAL_CRATE))
}

fn crate_export_threshold(crate_type: config::CrateType) -> SymbolExportLevel {
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

pub fn provide_local(providers: &mut Providers) {
    providers.exported_symbol_ids = |tcx, cnum| {
        let export_threshold = threshold(tcx);
        Rc::new(tcx.exported_symbols(cnum)
            .iter()
            .filter_map(|&(_, id, level)| {
                id.and_then(|id| {
                    if level.is_below_threshold(export_threshold) {
                        Some(id)
                    } else {
                        None
                    }
                })
            })
            .collect())
    };

    providers.is_exported_symbol = |tcx, id| {
        // FIXME(#42293) needs red/green to not break a bunch of incremental
        // tests
        tcx.dep_graph.with_ignore(|| {
            tcx.exported_symbol_ids(id.krate).contains(&id)
        })
    };

    providers.exported_symbols = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        let local_exported_symbols = base::find_exported_symbols(tcx);

        let mut local_crate: Vec<_> = local_exported_symbols
            .iter()
            .map(|&node_id| {
                tcx.hir.local_def_id(node_id)
            })
            .map(|def_id| {
                let name = tcx.symbol_name(Instance::mono(tcx, def_id));
                let export_level = export_level(tcx, def_id);
                debug!("EXPORTED SYMBOL (local): {} ({:?})", name, export_level);
                (str::to_owned(&name), Some(def_id), export_level)
            })
            .collect();

        if let Some(_) = *tcx.sess.entry_fn.borrow() {
            local_crate.push(("main".to_string(),
                              None,
                              SymbolExportLevel::C));
        }

        if tcx.sess.allocator_kind.get().is_some() {
            for method in ALLOCATOR_METHODS {
                local_crate.push((format!("__rust_{}", method.name),
                                  None,
                                  SymbolExportLevel::Rust));
            }
        }

        if let Some(id) = tcx.sess.derive_registrar_fn.get() {
            let def_id = tcx.hir.local_def_id(id);
            let idx = def_id.index;
            let disambiguator = tcx.sess.local_crate_disambiguator();
            let registrar = tcx.sess.generate_derive_registrar_symbol(disambiguator, idx);
            local_crate.push((registrar, Some(def_id), SymbolExportLevel::C));
        }

        if tcx.sess.crate_types.borrow().contains(&config::CrateTypeDylib) {
            local_crate.push((metadata_symbol_name(tcx),
                              None,
                              SymbolExportLevel::Rust));
        }
        Arc::new(local_crate)
    };
}

pub fn provide_extern(providers: &mut Providers) {
    providers.exported_symbols = |tcx, cnum| {
        // If this crate is a plugin and/or a custom derive crate, then
        // we're not even going to link those in so we skip those crates.
        if tcx.plugin_registrar_fn(cnum).is_some() ||
           tcx.derive_registrar_fn(cnum).is_some() {
            return Arc::new(Vec::new())
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
            .exported_symbol_ids(cnum)
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
                (str::to_owned(&name), Some(def_id), export_level)
            })
            .collect();

        Arc::new(crate_exports)
    };
}

fn export_level(tcx: TyCtxt, sym_def_id: DefId) -> SymbolExportLevel {
    if tcx.contains_extern_indicator(sym_def_id) {
        SymbolExportLevel::C
    } else {
        SymbolExportLevel::Rust
    }
}
