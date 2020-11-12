use std::collections::hash_map::Entry::*;

use rustc_ast::expand::allocator::ALLOCATOR_METHODS;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::Node;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{
    metadata_symbol_name, ExportedSymbol, SymbolExportLevel,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::Instance;
use rustc_middle::ty::{SymbolName, TyCtxt};
use rustc_session::config::{CrateType, SanitizerSet};

pub fn threshold(tcx: TyCtxt<'_>) -> SymbolExportLevel {
    crates_export_threshold(&tcx.sess.crate_types())
}

fn crate_export_threshold(crate_type: CrateType) -> SymbolExportLevel {
    match crate_type {
        CrateType::Executable | CrateType::Staticlib | CrateType::ProcMacro | CrateType::Cdylib => {
            SymbolExportLevel::C
        }
        CrateType::Rlib | CrateType::Dylib => SymbolExportLevel::Rust,
    }
}

pub fn crates_export_threshold(crate_types: &[CrateType]) -> SymbolExportLevel {
    if crate_types
        .iter()
        .any(|&crate_type| crate_export_threshold(crate_type) == SymbolExportLevel::Rust)
    {
        SymbolExportLevel::Rust
    } else {
        SymbolExportLevel::C
    }
}

fn reachable_non_generics_provider(tcx: TyCtxt<'_>, cnum: CrateNum) -> DefIdMap<SymbolExportLevel> {
    assert_eq!(cnum, LOCAL_CRATE);

    if !tcx.sess.opts.output_types.should_codegen() {
        return Default::default();
    }

    // Check to see if this crate is a "special runtime crate". These
    // crates, implementation details of the standard library, typically
    // have a bunch of `pub extern` and `#[no_mangle]` functions as the
    // ABI between them. We don't want their symbols to have a `C`
    // export level, however, as they're just implementation details.
    // Down below we'll hardwire all of the symbols to the `Rust` export
    // level instead.
    let special_runtime_crate =
        tcx.is_panic_runtime(LOCAL_CRATE) || tcx.is_compiler_builtins(LOCAL_CRATE);

    let mut reachable_non_generics: DefIdMap<_> = tcx
        .reachable_set(LOCAL_CRATE)
        .iter()
        .filter_map(|&def_id| {
            // We want to ignore some FFI functions that are not exposed from
            // this crate. Reachable FFI functions can be lumped into two
            // categories:
            //
            // 1. Those that are included statically via a static library
            // 2. Those included otherwise (e.g., dynamically or via a framework)
            //
            // Although our LLVM module is not literally emitting code for the
            // statically included symbols, it's an export of our library which
            // needs to be passed on to the linker and encoded in the metadata.
            //
            // As a result, if this id is an FFI item (foreign item) then we only
            // let it through if it's included statically.
            match tcx.hir().get(tcx.hir().local_def_id_to_hir_id(def_id)) {
                Node::ForeignItem(..) => {
                    tcx.is_statically_included_foreign_item(def_id).then_some(def_id)
                }

                // Only consider nodes that actually have exported symbols.
                Node::Item(&hir::Item {
                    kind: hir::ItemKind::Static(..) | hir::ItemKind::Fn(..),
                    ..
                })
                | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }) => {
                    let generics = tcx.generics_of(def_id);
                    if !generics.requires_monomorphization(tcx)
                        // Functions marked with #[inline] are codegened with "internal"
                        // linkage and are not exported unless marked with an extern
                        // inidicator
                        && (!Instance::mono(tcx, def_id.to_def_id()).def.generates_cgu_internal_copy(tcx)
                            || tcx.codegen_fn_attrs(def_id.to_def_id()).contains_extern_indicator())
                    {
                        Some(def_id)
                    } else {
                        None
                    }
                }

                _ => None,
            }
        })
        .map(|def_id| {
            let export_level = if special_runtime_crate {
                let name = tcx.symbol_name(Instance::mono(tcx, def_id.to_def_id())).name;
                // We can probably do better here by just ensuring that
                // it has hidden visibility rather than public
                // visibility, as this is primarily here to ensure it's
                // not stripped during LTO.
                //
                // In general though we won't link right if these
                // symbols are stripped, and LTO currently strips them.
                match name {
                    "rust_eh_personality"
                    | "rust_eh_register_frames"
                    | "rust_eh_unregister_frames" =>
                        SymbolExportLevel::C,
                    _ => SymbolExportLevel::Rust,
                }
            } else {
                symbol_export_level(tcx, def_id.to_def_id())
            };
            debug!(
                "EXPORTED SYMBOL (local): {} ({:?})",
                tcx.symbol_name(Instance::mono(tcx, def_id.to_def_id())),
                export_level
            );
            (def_id.to_def_id(), export_level)
        })
        .collect();

    if let Some(id) = tcx.proc_macro_decls_static(LOCAL_CRATE) {
        reachable_non_generics.insert(id, SymbolExportLevel::C);
    }

    if let Some(id) = tcx.plugin_registrar_fn(LOCAL_CRATE) {
        reachable_non_generics.insert(id, SymbolExportLevel::C);
    }

    reachable_non_generics
}

fn is_reachable_non_generic_provider_local(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let export_threshold = threshold(tcx);

    if let Some(&level) = tcx.reachable_non_generics(def_id.krate).get(&def_id) {
        level.is_below_threshold(export_threshold)
    } else {
        false
    }
}

fn is_reachable_non_generic_provider_extern(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.reachable_non_generics(def_id.krate).contains_key(&def_id)
}

fn exported_symbols_provider_local(
    tcx: TyCtxt<'tcx>,
    cnum: CrateNum,
) -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportLevel)] {
    assert_eq!(cnum, LOCAL_CRATE);

    if !tcx.sess.opts.output_types.should_codegen() {
        return &[];
    }

    let mut symbols: Vec<_> = tcx
        .reachable_non_generics(LOCAL_CRATE)
        .iter()
        .map(|(&def_id, &level)| (ExportedSymbol::NonGeneric(def_id), level))
        .collect();

    if tcx.entry_fn(LOCAL_CRATE).is_some() {
        let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, "main"));

        symbols.push((exported_symbol, SymbolExportLevel::C));
    }

    if tcx.allocator_kind().is_some() {
        for method in ALLOCATOR_METHODS {
            let symbol_name = format!("__rust_{}", method.name);
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, &symbol_name));

            symbols.push((exported_symbol, SymbolExportLevel::Rust));
        }
    }

    if tcx.sess.opts.debugging_opts.instrument_coverage
        || tcx.sess.opts.cg.profile_generate.enabled()
    {
        // These are weak symbols that point to the profile version and the
        // profile name, which need to be treated as exported so LTO doesn't nix
        // them.
        const PROFILER_WEAK_SYMBOLS: [&str; 2] =
            ["__llvm_profile_raw_version", "__llvm_profile_filename"];

        symbols.extend(PROFILER_WEAK_SYMBOLS.iter().map(|sym| {
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, sym));
            (exported_symbol, SymbolExportLevel::C)
        }));
    }

    if tcx.sess.opts.debugging_opts.sanitizer.contains(SanitizerSet::MEMORY) {
        // Similar to profiling, preserve weak msan symbol during LTO.
        const MSAN_WEAK_SYMBOLS: [&str; 2] = ["__msan_track_origins", "__msan_keep_going"];

        symbols.extend(MSAN_WEAK_SYMBOLS.iter().map(|sym| {
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, sym));
            (exported_symbol, SymbolExportLevel::C)
        }));
    }

    if tcx.sess.crate_types().contains(&CrateType::Dylib) {
        let symbol_name = metadata_symbol_name(tcx);
        let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, &symbol_name));

        symbols.push((exported_symbol, SymbolExportLevel::Rust));
    }

    if tcx.sess.opts.share_generics() && tcx.local_crate_exports_generics() {
        use rustc_middle::mir::mono::{Linkage, MonoItem, Visibility};
        use rustc_middle::ty::InstanceDef;

        // Normally, we require that shared monomorphizations are not hidden,
        // because if we want to re-use a monomorphization from a Rust dylib, it
        // needs to be exported.
        // However, on platforms that don't allow for Rust dylibs, having
        // external linkage is enough for monomorphization to be linked to.
        let need_visibility = tcx.sess.target.dynamic_linking && !tcx.sess.target.only_cdylib;

        let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);

        for (mono_item, &(linkage, visibility)) in cgus.iter().flat_map(|cgu| cgu.items().iter()) {
            if linkage != Linkage::External {
                // We can only re-use things with external linkage, otherwise
                // we'll get a linker error
                continue;
            }

            if need_visibility && visibility == Visibility::Hidden {
                // If we potentially share things from Rust dylibs, they must
                // not be hidden
                continue;
            }

            match *mono_item {
                MonoItem::Fn(Instance { def: InstanceDef::Item(def), substs }) => {
                    if substs.non_erasable_generics().next().is_some() {
                        let symbol = ExportedSymbol::Generic(def.did, substs);
                        symbols.push((symbol, SymbolExportLevel::Rust));
                    }
                }
                MonoItem::Fn(Instance { def: InstanceDef::DropGlue(_, Some(ty)), substs }) => {
                    // A little sanity-check
                    debug_assert_eq!(
                        substs.non_erasable_generics().next(),
                        Some(GenericArgKind::Type(ty))
                    );
                    symbols.push((ExportedSymbol::DropGlue(ty), SymbolExportLevel::Rust));
                }
                _ => {
                    // Any other symbols don't qualify for sharing
                }
            }
        }
    }

    // Sort so we get a stable incr. comp. hash.
    symbols.sort_by_cached_key(|s| s.0.symbol_name_for_local_instance(tcx));

    tcx.arena.alloc_from_iter(symbols)
}

fn upstream_monomorphizations_provider(
    tcx: TyCtxt<'_>,
    cnum: CrateNum,
) -> DefIdMap<FxHashMap<SubstsRef<'_>, CrateNum>> {
    debug_assert!(cnum == LOCAL_CRATE);

    let cnums = tcx.all_crate_nums(LOCAL_CRATE);

    let mut instances: DefIdMap<FxHashMap<_, _>> = Default::default();

    let cnum_stable_ids: IndexVec<CrateNum, Fingerprint> = {
        let mut cnum_stable_ids = IndexVec::from_elem_n(Fingerprint::ZERO, cnums.len() + 1);

        for &cnum in cnums.iter() {
            cnum_stable_ids[cnum] =
                tcx.def_path_hash(DefId { krate: cnum, index: CRATE_DEF_INDEX }).0;
        }

        cnum_stable_ids
    };

    let drop_in_place_fn_def_id = tcx.lang_items().drop_in_place_fn();

    for &cnum in cnums.iter() {
        for (exported_symbol, _) in tcx.exported_symbols(cnum).iter() {
            let (def_id, substs) = match *exported_symbol {
                ExportedSymbol::Generic(def_id, substs) => (def_id, substs),
                ExportedSymbol::DropGlue(ty) => {
                    if let Some(drop_in_place_fn_def_id) = drop_in_place_fn_def_id {
                        (drop_in_place_fn_def_id, tcx.intern_substs(&[ty.into()]))
                    } else {
                        // `drop_in_place` in place does not exist, don't try
                        // to use it.
                        continue;
                    }
                }
                ExportedSymbol::NonGeneric(..) | ExportedSymbol::NoDefId(..) => {
                    // These are no monomorphizations
                    continue;
                }
            };

            let substs_map = instances.entry(def_id).or_default();

            match substs_map.entry(substs) {
                Occupied(mut e) => {
                    // If there are multiple monomorphizations available,
                    // we select one deterministically.
                    let other_cnum = *e.get();
                    if cnum_stable_ids[other_cnum] > cnum_stable_ids[cnum] {
                        e.insert(cnum);
                    }
                }
                Vacant(e) => {
                    e.insert(cnum);
                }
            }
        }
    }

    instances
}

fn upstream_monomorphizations_for_provider(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Option<&FxHashMap<SubstsRef<'_>, CrateNum>> {
    debug_assert!(!def_id.is_local());
    tcx.upstream_monomorphizations(LOCAL_CRATE).get(&def_id)
}

fn upstream_drop_glue_for_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    substs: SubstsRef<'tcx>,
) -> Option<CrateNum> {
    if let Some(def_id) = tcx.lang_items().drop_in_place_fn() {
        tcx.upstream_monomorphizations_for(def_id).and_then(|monos| monos.get(&substs).cloned())
    } else {
        None
    }
}

fn is_unreachable_local_definition_provider(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if let Some(def_id) = def_id.as_local() {
        !tcx.reachable_set(LOCAL_CRATE).contains(&def_id)
    } else {
        bug!("is_unreachable_local_definition called with non-local DefId: {:?}", def_id)
    }
}

pub fn provide(providers: &mut Providers) {
    providers.reachable_non_generics = reachable_non_generics_provider;
    providers.is_reachable_non_generic = is_reachable_non_generic_provider_local;
    providers.exported_symbols = exported_symbols_provider_local;
    providers.upstream_monomorphizations = upstream_monomorphizations_provider;
    providers.is_unreachable_local_definition = is_unreachable_local_definition_provider;
    providers.upstream_drop_glue_for = upstream_drop_glue_for_provider;
}

pub fn provide_extern(providers: &mut Providers) {
    providers.is_reachable_non_generic = is_reachable_non_generic_provider_extern;
    providers.upstream_monomorphizations_for = upstream_monomorphizations_for_provider;
}

fn symbol_export_level(tcx: TyCtxt<'_>, sym_def_id: DefId) -> SymbolExportLevel {
    // We export anything that's not mangled at the "C" layer as it probably has
    // to do with ABI concerns. We do not, however, apply such treatment to
    // special symbols in the standard library for various plumbing between
    // core/std/allocators/etc. For example symbols used to hook up allocation
    // are not considered for export
    let codegen_fn_attrs = tcx.codegen_fn_attrs(sym_def_id);
    let is_extern = codegen_fn_attrs.contains_extern_indicator();
    let std_internal =
        codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL);

    if is_extern && !std_internal {
        let target = &tcx.sess.target.llvm_target;
        // WebAssembly cannot export data symbols, so reduce their export level
        if target.contains("emscripten") {
            if let Some(Node::Item(&hir::Item { kind: hir::ItemKind::Static(..), .. })) =
                tcx.hir().get_if_local(sym_def_id)
            {
                return SymbolExportLevel::Rust;
            }
        }

        SymbolExportLevel::C
    } else {
        SymbolExportLevel::Rust
    }
}

/// This is the symbol name of the given instance instantiated in a specific crate.
pub fn symbol_name_for_instance_in_crate<'tcx>(
    tcx: TyCtxt<'tcx>,
    symbol: ExportedSymbol<'tcx>,
    instantiating_crate: CrateNum,
) -> String {
    // If this is something instantiated in the local crate then we might
    // already have cached the name as a query result.
    if instantiating_crate == LOCAL_CRATE {
        return symbol.symbol_name_for_local_instance(tcx).to_string();
    }

    // This is something instantiated in an upstream crate, so we have to use
    // the slower (because uncached) version of computing the symbol name.
    match symbol {
        ExportedSymbol::NonGeneric(def_id) => {
            rustc_symbol_mangling::symbol_name_for_instance_in_crate(
                tcx,
                Instance::mono(tcx, def_id),
                instantiating_crate,
            )
        }
        ExportedSymbol::Generic(def_id, substs) => {
            rustc_symbol_mangling::symbol_name_for_instance_in_crate(
                tcx,
                Instance::new(def_id, substs),
                instantiating_crate,
            )
        }
        ExportedSymbol::DropGlue(ty) => rustc_symbol_mangling::symbol_name_for_instance_in_crate(
            tcx,
            Instance::resolve_drop_in_place(tcx, ty),
            instantiating_crate,
        ),
        ExportedSymbol::NoDefId(symbol_name) => symbol_name.to_string(),
    }
}
