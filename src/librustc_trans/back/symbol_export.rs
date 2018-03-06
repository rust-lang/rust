// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::sync::Lrc;
use std::sync::Arc;

use monomorphize::Instance;
use rustc::hir;
use rustc::hir::def_id::CrateNum;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::exported_symbols::{SymbolExportLevel, ExportedSymbol, metadata_symbol_name};
use rustc::session::config;
use rustc::ty::{TyCtxt, SymbolName};
use rustc::ty::maps::Providers;
use rustc::util::nodemap::{FxHashMap, DefIdSet};
use rustc_allocator::ALLOCATOR_METHODS;
use syntax::attr;

pub type ExportedSymbols = FxHashMap<
    CrateNum,
    Arc<Vec<(String, SymbolExportLevel)>>,
>;

pub fn threshold(tcx: TyCtxt) -> SymbolExportLevel {
    crates_export_threshold(&tcx.sess.crate_types.borrow())
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

fn reachable_non_generics_provider<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             cnum: CrateNum)
                                             -> Lrc<DefIdSet>
{
    assert_eq!(cnum, LOCAL_CRATE);

    if !tcx.sess.opts.output_types.should_trans() {
        return Lrc::new(DefIdSet())
    }

    let export_threshold = threshold(tcx);

    // We already collect all potentially reachable non-generic items for
    // `exported_symbols`. Now we just filter them down to what is actually
    // exported for the given crate we are compiling.
    let reachable_non_generics = tcx
        .exported_symbols(LOCAL_CRATE)
        .iter()
        .filter_map(|&(exported_symbol, level)| {
            if let ExportedSymbol::NonGeneric(def_id) = exported_symbol {
                if level.is_below_threshold(export_threshold) {
                    return Some(def_id)
                }
            }

            None
        })
        .collect();

    Lrc::new(reachable_non_generics)
}

fn is_reachable_non_generic_provider<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               def_id: DefId)
                                               -> bool {
    tcx.reachable_non_generics(def_id.krate).contains(&def_id)
}

fn exported_symbols_provider_local<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             cnum: CrateNum)
                                             -> Arc<Vec<(ExportedSymbol,
                                                         SymbolExportLevel)>>
{
    assert_eq!(cnum, LOCAL_CRATE);

    if !tcx.sess.opts.output_types.should_trans() {
        return Arc::new(vec![])
    }

    // Check to see if this crate is a "special runtime crate". These
    // crates, implementation details of the standard library, typically
    // have a bunch of `pub extern` and `#[no_mangle]` functions as the
    // ABI between them. We don't want their symbols to have a `C`
    // export level, however, as they're just implementation details.
    // Down below we'll hardwire all of the symbols to the `Rust` export
    // level instead.
    let special_runtime_crate = tcx.is_panic_runtime(LOCAL_CRATE) ||
        tcx.is_compiler_builtins(LOCAL_CRATE);

    let reachable_non_generics: DefIdSet = tcx.reachable_set(LOCAL_CRATE).0
        .iter()
        .filter_map(|&node_id| {
            // We want to ignore some FFI functions that are not exposed from
            // this crate. Reachable FFI functions can be lumped into two
            // categories:
            //
            // 1. Those that are included statically via a static library
            // 2. Those included otherwise (e.g. dynamically or via a framework)
            //
            // Although our LLVM module is not literally emitting code for the
            // statically included symbols, it's an export of our library which
            // needs to be passed on to the linker and encoded in the metadata.
            //
            // As a result, if this id is an FFI item (foreign item) then we only
            // let it through if it's included statically.
            match tcx.hir.get(node_id) {
                hir::map::NodeForeignItem(..) => {
                    let def_id = tcx.hir.local_def_id(node_id);
                    if tcx.is_statically_included_foreign_item(def_id) {
                        Some(def_id)
                    } else {
                        None
                    }
                }

                // Only consider nodes that actually have exported symbols.
                hir::map::NodeItem(&hir::Item {
                    node: hir::ItemStatic(..),
                    ..
                }) |
                hir::map::NodeItem(&hir::Item {
                    node: hir::ItemFn(..), ..
                }) |
                hir::map::NodeImplItem(&hir::ImplItem {
                    node: hir::ImplItemKind::Method(..),
                    ..
                }) => {
                    let def_id = tcx.hir.local_def_id(node_id);
                    let generics = tcx.generics_of(def_id);
                    if (generics.parent_types == 0 && generics.types.is_empty()) &&
                        // Functions marked with #[inline] are only ever translated
                        // with "internal" linkage and are never exported.
                        !Instance::mono(tcx, def_id).def.requires_local(tcx) {
                        Some(def_id)
                    } else {
                        None
                    }
                }

                _ => None
            }
        })
        .collect();

    let mut symbols: Vec<_> = reachable_non_generics
        .iter()
        .map(|&def_id| {
            let export_level = if special_runtime_crate {
                let name = tcx.symbol_name(Instance::mono(tcx, def_id));
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
                tcx.symbol_export_level(def_id)
            };
            debug!("EXPORTED SYMBOL (local): {} ({:?})",
                   tcx.symbol_name(Instance::mono(tcx, def_id)),
                   export_level);
            (ExportedSymbol::NonGeneric(def_id), export_level)
        })
        .collect();

    if let Some(id) = tcx.sess.derive_registrar_fn.get() {
        let def_id = tcx.hir.local_def_id(id);
        symbols.push((ExportedSymbol::NonGeneric(def_id), SymbolExportLevel::C));
    }

    if let Some(id) = tcx.sess.plugin_registrar_fn.get() {
        let def_id = tcx.hir.local_def_id(id);
        symbols.push((ExportedSymbol::NonGeneric(def_id), SymbolExportLevel::C));
    }

    if let Some(_) = *tcx.sess.entry_fn.borrow() {
        let symbol_name = "main".to_string();
        let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(&symbol_name));

        symbols.push((exported_symbol, SymbolExportLevel::C));
    }

    if tcx.sess.allocator_kind.get().is_some() {
        for method in ALLOCATOR_METHODS {
            let symbol_name = format!("__rust_{}", method.name);
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(&symbol_name));

            symbols.push((exported_symbol, SymbolExportLevel::Rust));
        }
    }

    if tcx.sess.crate_types.borrow().contains(&config::CrateTypeDylib) {
        let symbol_name = metadata_symbol_name(tcx);
        let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(&symbol_name));

        symbols.push((exported_symbol, SymbolExportLevel::Rust));
    }

    // Sort so we get a stable incr. comp. hash.
    symbols.sort_unstable_by(|&(ref symbol1, ..), &(ref symbol2, ..)| {
        symbol1.compare_stable(tcx, symbol2)
    });

    Arc::new(symbols)
}

pub fn provide(providers: &mut Providers) {
    providers.reachable_non_generics = reachable_non_generics_provider;
    providers.is_reachable_non_generic = is_reachable_non_generic_provider;
    providers.exported_symbols = exported_symbols_provider_local;
    providers.symbol_export_level = symbol_export_level_provider;
}

pub fn provide_extern(providers: &mut Providers) {
    providers.is_reachable_non_generic = is_reachable_non_generic_provider;
    providers.symbol_export_level = symbol_export_level_provider;
}

fn symbol_export_level_provider(tcx: TyCtxt, sym_def_id: DefId) -> SymbolExportLevel {
    // We export anything that's not mangled at the "C" layer as it probably has
    // to do with ABI concerns. We do not, however, apply such treatment to
    // special symbols in the standard library for various plumbing between
    // core/std/allocators/etc. For example symbols used to hook up allocation
    // are not considered for export
    let is_extern = tcx.contains_extern_indicator(sym_def_id);
    let std_internal = attr::contains_name(&tcx.get_attrs(sym_def_id),
                                           "rustc_std_internal_symbol");
    if is_extern && !std_internal {
        SymbolExportLevel::C
    } else {
        SymbolExportLevel::Rust
    }
}
