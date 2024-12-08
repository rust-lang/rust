use std::collections::hash_map::Entry::*;

use rustc_ast::expand::allocator::{ALLOCATOR_METHODS, NO_ALLOC_SHIM_IS_UNSTABLE};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LOCAL_CRATE, LocalDefId};
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{
    ExportedSymbol, SymbolExportInfo, SymbolExportKind, SymbolExportLevel, metadata_symbol_name,
};
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, Instance, SymbolName, TyCtxt};
use rustc_middle::util::Providers;
use rustc_session::config::{CrateType, OomStrategy};
use rustc_target::spec::{SanitizerSet, TlsModel};
use tracing::debug;

use crate::base::allocator_kind_for_codegen;

fn threshold(tcx: TyCtxt<'_>) -> SymbolExportLevel {
    crates_export_threshold(tcx.crate_types())
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

fn reachable_non_generics_provider(tcx: TyCtxt<'_>, _: LocalCrate) -> DefIdMap<SymbolExportInfo> {
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
        .reachable_set(())
        .items()
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
            if let Some(parent_id) = tcx.opt_local_parent(def_id)
                && let DefKind::ForeignMod = tcx.def_kind(parent_id)
            {
                let library = tcx.native_library(def_id)?;
                return library.kind.is_statically_included().then_some(def_id);
            }

            // Only consider nodes that actually have exported symbols.
            match tcx.def_kind(def_id) {
                DefKind::Fn | DefKind::Static { .. } => {}
                DefKind::AssocFn if tcx.impl_of_method(def_id.to_def_id()).is_some() => {}
                _ => return None,
            };

            let generics = tcx.generics_of(def_id);
            if generics.requires_monomorphization(tcx) {
                return None;
            }

            // Functions marked with #[inline] are codegened with "internal"
            // linkage and are not exported unless marked with an extern
            // indicator
            if !Instance::mono(tcx, def_id.to_def_id()).def.generates_cgu_internal_copy(tcx)
                || tcx.codegen_fn_attrs(def_id.to_def_id()).contains_extern_indicator()
            {
                Some(def_id)
            } else {
                None
            }
        })
        .map(|def_id| {
            // We won't link right if this symbol is stripped during LTO.
            let name = tcx.symbol_name(Instance::mono(tcx, def_id.to_def_id())).name;
            let used = name == "rust_eh_personality";

            let export_level = if special_runtime_crate {
                SymbolExportLevel::Rust
            } else {
                symbol_export_level(tcx, def_id.to_def_id())
            };
            let codegen_attrs = tcx.codegen_fn_attrs(def_id.to_def_id());
            debug!(
                "EXPORTED SYMBOL (local): {} ({:?})",
                tcx.symbol_name(Instance::mono(tcx, def_id.to_def_id())),
                export_level
            );
            let info = SymbolExportInfo {
                level: export_level,
                kind: if tcx.is_static(def_id.to_def_id()) {
                    if codegen_attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                        SymbolExportKind::Tls
                    } else {
                        SymbolExportKind::Data
                    }
                } else {
                    SymbolExportKind::Text
                },
                used: codegen_attrs.flags.contains(CodegenFnAttrFlags::USED)
                    || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
                    || used,
            };
            (def_id.to_def_id(), info)
        })
        .into();

    if let Some(id) = tcx.proc_macro_decls_static(()) {
        reachable_non_generics.insert(id.to_def_id(), SymbolExportInfo {
            level: SymbolExportLevel::C,
            kind: SymbolExportKind::Data,
            used: false,
        });
    }

    reachable_non_generics
}

fn is_reachable_non_generic_provider_local(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let export_threshold = threshold(tcx);

    if let Some(&info) = tcx.reachable_non_generics(LOCAL_CRATE).get(&def_id.to_def_id()) {
        info.level.is_below_threshold(export_threshold)
    } else {
        false
    }
}

fn is_reachable_non_generic_provider_extern(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.reachable_non_generics(def_id.krate).contains_key(&def_id)
}

fn exported_symbols_provider_local(
    tcx: TyCtxt<'_>,
    _: LocalCrate,
) -> &[(ExportedSymbol<'_>, SymbolExportInfo)] {
    if !tcx.sess.opts.output_types.should_codegen() {
        return &[];
    }

    // FIXME: Sorting this is unnecessary since we are sorting later anyway.
    //        Can we skip the later sorting?
    let sorted = tcx.with_stable_hashing_context(|hcx| {
        tcx.reachable_non_generics(LOCAL_CRATE).to_sorted(&hcx, true)
    });

    let mut symbols: Vec<_> =
        sorted.iter().map(|(&def_id, &info)| (ExportedSymbol::NonGeneric(def_id), info)).collect();

    // Export TLS shims
    if !tcx.sess.target.dll_tls_export {
        symbols.extend(sorted.iter().filter_map(|(&def_id, &info)| {
            tcx.needs_thread_local_shim(def_id).then(|| {
                (ExportedSymbol::ThreadLocalShim(def_id), SymbolExportInfo {
                    level: info.level,
                    kind: SymbolExportKind::Text,
                    used: info.used,
                })
            })
        }))
    }

    if tcx.entry_fn(()).is_some() {
        let exported_symbol =
            ExportedSymbol::NoDefId(SymbolName::new(tcx, tcx.sess.target.entry_name.as_ref()));

        symbols.push((exported_symbol, SymbolExportInfo {
            level: SymbolExportLevel::C,
            kind: SymbolExportKind::Text,
            used: false,
        }));
    }

    // Mark allocator shim symbols as exported only if they were generated.
    if allocator_kind_for_codegen(tcx).is_some() {
        for symbol_name in ALLOCATOR_METHODS
            .iter()
            .map(|method| format!("__rust_{}", method.name))
            .chain(["__rust_alloc_error_handler".to_string(), OomStrategy::SYMBOL.to_string()])
        {
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, &symbol_name));

            symbols.push((exported_symbol, SymbolExportInfo {
                level: SymbolExportLevel::Rust,
                kind: SymbolExportKind::Text,
                used: false,
            }));
        }

        let exported_symbol =
            ExportedSymbol::NoDefId(SymbolName::new(tcx, NO_ALLOC_SHIM_IS_UNSTABLE));
        symbols.push((exported_symbol, SymbolExportInfo {
            level: SymbolExportLevel::Rust,
            kind: SymbolExportKind::Data,
            used: false,
        }))
    }

    if tcx.sess.instrument_coverage() || tcx.sess.opts.cg.profile_generate.enabled() {
        // These are weak symbols that point to the profile version and the
        // profile name, which need to be treated as exported so LTO doesn't nix
        // them.
        const PROFILER_WEAK_SYMBOLS: [&str; 2] =
            ["__llvm_profile_raw_version", "__llvm_profile_filename"];

        symbols.extend(PROFILER_WEAK_SYMBOLS.iter().map(|sym| {
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, sym));
            (exported_symbol, SymbolExportInfo {
                level: SymbolExportLevel::C,
                kind: SymbolExportKind::Data,
                used: false,
            })
        }));
    }

    if tcx.sess.opts.unstable_opts.sanitizer.contains(SanitizerSet::MEMORY) {
        let mut msan_weak_symbols = Vec::new();

        // Similar to profiling, preserve weak msan symbol during LTO.
        if tcx.sess.opts.unstable_opts.sanitizer_recover.contains(SanitizerSet::MEMORY) {
            msan_weak_symbols.push("__msan_keep_going");
        }

        if tcx.sess.opts.unstable_opts.sanitizer_memory_track_origins != 0 {
            msan_weak_symbols.push("__msan_track_origins");
        }

        symbols.extend(msan_weak_symbols.into_iter().map(|sym| {
            let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, sym));
            (exported_symbol, SymbolExportInfo {
                level: SymbolExportLevel::C,
                kind: SymbolExportKind::Data,
                used: false,
            })
        }));
    }

    if tcx.crate_types().contains(&CrateType::Dylib)
        || tcx.crate_types().contains(&CrateType::ProcMacro)
    {
        let symbol_name = metadata_symbol_name(tcx);
        let exported_symbol = ExportedSymbol::NoDefId(SymbolName::new(tcx, &symbol_name));

        symbols.push((exported_symbol, SymbolExportInfo {
            level: SymbolExportLevel::C,
            kind: SymbolExportKind::Data,
            used: true,
        }));
    }

    if tcx.local_crate_exports_generics() {
        use rustc_middle::mir::mono::{Linkage, MonoItem, Visibility};
        use rustc_middle::ty::InstanceKind;

        // Normally, we require that shared monomorphizations are not hidden,
        // because if we want to re-use a monomorphization from a Rust dylib, it
        // needs to be exported.
        // However, on platforms that don't allow for Rust dylibs, having
        // external linkage is enough for monomorphization to be linked to.
        let need_visibility = tcx.sess.target.dynamic_linking && !tcx.sess.target.only_cdylib;

        let (_, cgus) = tcx.collect_and_partition_mono_items(());

        // The symbols created in this loop are sorted below it
        #[allow(rustc::potential_query_instability)]
        for (mono_item, data) in cgus.iter().flat_map(|cgu| cgu.items().iter()) {
            if data.linkage != Linkage::External {
                // We can only re-use things with external linkage, otherwise
                // we'll get a linker error
                continue;
            }

            if need_visibility && data.visibility == Visibility::Hidden {
                // If we potentially share things from Rust dylibs, they must
                // not be hidden
                continue;
            }

            if !tcx.sess.opts.share_generics() {
                if tcx.codegen_fn_attrs(mono_item.def_id()).inline == rustc_attr::InlineAttr::Never
                {
                    // this is OK, we explicitly allow sharing inline(never) across crates even
                    // without share-generics.
                } else {
                    continue;
                }
            }

            match *mono_item {
                MonoItem::Fn(Instance { def: InstanceKind::Item(def), args }) => {
                    if args.non_erasable_generics().next().is_some() {
                        let symbol = ExportedSymbol::Generic(def, args);
                        symbols.push((symbol, SymbolExportInfo {
                            level: SymbolExportLevel::Rust,
                            kind: SymbolExportKind::Text,
                            used: false,
                        }));
                    }
                }
                MonoItem::Fn(Instance { def: InstanceKind::DropGlue(_, Some(ty)), args }) => {
                    // A little sanity-check
                    assert_eq!(args.non_erasable_generics().next(), Some(GenericArgKind::Type(ty)));
                    symbols.push((ExportedSymbol::DropGlue(ty), SymbolExportInfo {
                        level: SymbolExportLevel::Rust,
                        kind: SymbolExportKind::Text,
                        used: false,
                    }));
                }
                MonoItem::Fn(Instance {
                    def: InstanceKind::AsyncDropGlueCtorShim(_, Some(ty)),
                    args,
                }) => {
                    // A little sanity-check
                    assert_eq!(args.non_erasable_generics().next(), Some(GenericArgKind::Type(ty)));
                    symbols.push((ExportedSymbol::AsyncDropGlueCtorShim(ty), SymbolExportInfo {
                        level: SymbolExportLevel::Rust,
                        kind: SymbolExportKind::Text,
                        used: false,
                    }));
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
    (): (),
) -> DefIdMap<UnordMap<GenericArgsRef<'_>, CrateNum>> {
    let cnums = tcx.crates(());

    let mut instances: DefIdMap<UnordMap<_, _>> = Default::default();

    let drop_in_place_fn_def_id = tcx.lang_items().drop_in_place_fn();
    let async_drop_in_place_fn_def_id = tcx.lang_items().async_drop_in_place_fn();

    for &cnum in cnums.iter() {
        for (exported_symbol, _) in tcx.exported_symbols(cnum).iter() {
            let (def_id, args) = match *exported_symbol {
                ExportedSymbol::Generic(def_id, args) => (def_id, args),
                ExportedSymbol::DropGlue(ty) => {
                    if let Some(drop_in_place_fn_def_id) = drop_in_place_fn_def_id {
                        (drop_in_place_fn_def_id, tcx.mk_args(&[ty.into()]))
                    } else {
                        // `drop_in_place` in place does not exist, don't try
                        // to use it.
                        continue;
                    }
                }
                ExportedSymbol::AsyncDropGlueCtorShim(ty) => {
                    if let Some(async_drop_in_place_fn_def_id) = async_drop_in_place_fn_def_id {
                        (async_drop_in_place_fn_def_id, tcx.mk_args(&[ty.into()]))
                    } else {
                        // `drop_in_place` in place does not exist, don't try
                        // to use it.
                        continue;
                    }
                }
                ExportedSymbol::NonGeneric(..)
                | ExportedSymbol::ThreadLocalShim(..)
                | ExportedSymbol::NoDefId(..) => {
                    // These are no monomorphizations
                    continue;
                }
            };

            let args_map = instances.entry(def_id).or_default();

            match args_map.entry(args) {
                Occupied(mut e) => {
                    // If there are multiple monomorphizations available,
                    // we select one deterministically.
                    let other_cnum = *e.get();
                    if tcx.stable_crate_id(other_cnum) > tcx.stable_crate_id(cnum) {
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
) -> Option<&UnordMap<GenericArgsRef<'_>, CrateNum>> {
    assert!(!def_id.is_local());
    tcx.upstream_monomorphizations(()).get(&def_id)
}

fn upstream_drop_glue_for_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: GenericArgsRef<'tcx>,
) -> Option<CrateNum> {
    let def_id = tcx.lang_items().drop_in_place_fn()?;
    tcx.upstream_monomorphizations_for(def_id)?.get(&args).cloned()
}

fn upstream_async_drop_glue_for_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: GenericArgsRef<'tcx>,
) -> Option<CrateNum> {
    let def_id = tcx.lang_items().async_drop_in_place_fn()?;
    tcx.upstream_monomorphizations_for(def_id)?.get(&args).cloned()
}

fn is_unreachable_local_definition_provider(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    !tcx.reachable_set(()).contains(&def_id)
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.reachable_non_generics = reachable_non_generics_provider;
    providers.is_reachable_non_generic = is_reachable_non_generic_provider_local;
    providers.exported_symbols = exported_symbols_provider_local;
    providers.upstream_monomorphizations = upstream_monomorphizations_provider;
    providers.is_unreachable_local_definition = is_unreachable_local_definition_provider;
    providers.upstream_drop_glue_for = upstream_drop_glue_for_provider;
    providers.upstream_async_drop_glue_for = upstream_async_drop_glue_for_provider;
    providers.wasm_import_module_map = wasm_import_module_map;
    providers.extern_queries.is_reachable_non_generic = is_reachable_non_generic_provider_extern;
    providers.extern_queries.upstream_monomorphizations_for =
        upstream_monomorphizations_for_provider;
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
            if let DefKind::Static { .. } = tcx.def_kind(sym_def_id) {
                return SymbolExportLevel::Rust;
            }
        }

        SymbolExportLevel::C
    } else {
        SymbolExportLevel::Rust
    }
}

/// This is the symbol name of the given instance instantiated in a specific crate.
pub(crate) fn symbol_name_for_instance_in_crate<'tcx>(
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
        ExportedSymbol::Generic(def_id, args) => {
            rustc_symbol_mangling::symbol_name_for_instance_in_crate(
                tcx,
                Instance::new(def_id, args),
                instantiating_crate,
            )
        }
        ExportedSymbol::ThreadLocalShim(def_id) => {
            rustc_symbol_mangling::symbol_name_for_instance_in_crate(
                tcx,
                ty::Instance {
                    def: ty::InstanceKind::ThreadLocalShim(def_id),
                    args: ty::GenericArgs::empty(),
                },
                instantiating_crate,
            )
        }
        ExportedSymbol::DropGlue(ty) => rustc_symbol_mangling::symbol_name_for_instance_in_crate(
            tcx,
            Instance::resolve_drop_in_place(tcx, ty),
            instantiating_crate,
        ),
        ExportedSymbol::AsyncDropGlueCtorShim(ty) => {
            rustc_symbol_mangling::symbol_name_for_instance_in_crate(
                tcx,
                Instance::resolve_async_drop_in_place(tcx, ty),
                instantiating_crate,
            )
        }
        ExportedSymbol::NoDefId(symbol_name) => symbol_name.to_string(),
    }
}

/// This is the symbol name of the given instance as seen by the linker.
///
/// On 32-bit Windows symbols are decorated according to their calling conventions.
pub(crate) fn linking_symbol_name_for_instance_in_crate<'tcx>(
    tcx: TyCtxt<'tcx>,
    symbol: ExportedSymbol<'tcx>,
    instantiating_crate: CrateNum,
) -> String {
    use rustc_target::callconv::Conv;

    let mut undecorated = symbol_name_for_instance_in_crate(tcx, symbol, instantiating_crate);

    // thread local will not be a function call,
    // so it is safe to return before windows symbol decoration check.
    if let Some(name) = maybe_emutls_symbol_name(tcx, symbol, &undecorated) {
        return name;
    }

    let target = &tcx.sess.target;
    if !target.is_like_windows {
        // Mach-O has a global "_" suffix and `object` crate will handle it.
        // ELF does not have any symbol decorations.
        return undecorated;
    }

    let prefix = match &target.arch[..] {
        "x86" => Some('_'),
        "x86_64" => None,
        "arm64ec" => Some('#'),
        // Only x86/64 use symbol decorations.
        _ => return undecorated,
    };

    let instance = match symbol {
        ExportedSymbol::NonGeneric(def_id) | ExportedSymbol::Generic(def_id, _)
            if tcx.is_static(def_id) =>
        {
            None
        }
        ExportedSymbol::NonGeneric(def_id) => Some(Instance::mono(tcx, def_id)),
        ExportedSymbol::Generic(def_id, args) => Some(Instance::new(def_id, args)),
        // DropGlue always use the Rust calling convention and thus follow the target's default
        // symbol decoration scheme.
        ExportedSymbol::DropGlue(..) => None,
        // AsyncDropGlueCtorShim always use the Rust calling convention and thus follow the
        // target's default symbol decoration scheme.
        ExportedSymbol::AsyncDropGlueCtorShim(..) => None,
        // NoDefId always follow the target's default symbol decoration scheme.
        ExportedSymbol::NoDefId(..) => None,
        // ThreadLocalShim always follow the target's default symbol decoration scheme.
        ExportedSymbol::ThreadLocalShim(..) => None,
    };

    let (conv, args) = instance
        .map(|i| {
            tcx.fn_abi_of_instance(
                ty::TypingEnv::fully_monomorphized().as_query_input((i, ty::List::empty())),
            )
            .unwrap_or_else(|_| bug!("fn_abi_of_instance({i:?}) failed"))
        })
        .map(|fnabi| (fnabi.conv, &fnabi.args[..]))
        .unwrap_or((Conv::Rust, &[]));

    // Decorate symbols with prefixes, suffixes and total number of bytes of arguments.
    // Reference: https://docs.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170
    let (prefix, suffix) = match conv {
        Conv::X86Fastcall => ("@", "@"),
        Conv::X86Stdcall => ("_", "@"),
        Conv::X86VectorCall => ("", "@@"),
        _ => {
            if let Some(prefix) = prefix {
                undecorated.insert(0, prefix);
            }
            return undecorated;
        }
    };

    let args_in_bytes: u64 = args
        .iter()
        .map(|abi| abi.layout.size.bytes().next_multiple_of(target.pointer_width as u64 / 8))
        .sum();
    format!("{prefix}{undecorated}{suffix}{args_in_bytes}")
}

pub(crate) fn exporting_symbol_name_for_instance_in_crate<'tcx>(
    tcx: TyCtxt<'tcx>,
    symbol: ExportedSymbol<'tcx>,
    cnum: CrateNum,
) -> String {
    let undecorated = symbol_name_for_instance_in_crate(tcx, symbol, cnum);
    maybe_emutls_symbol_name(tcx, symbol, &undecorated).unwrap_or(undecorated)
}

fn maybe_emutls_symbol_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    symbol: ExportedSymbol<'tcx>,
    undecorated: &str,
) -> Option<String> {
    if matches!(tcx.sess.tls_model(), TlsModel::Emulated)
        && let ExportedSymbol::NonGeneric(def_id) = symbol
        && tcx.is_thread_local_static(def_id)
    {
        // When using emutls, LLVM will add the `__emutls_v.` prefix to thread local symbols,
        // and exported symbol name need to match this.
        Some(format!("__emutls_v.{undecorated}"))
    } else {
        None
    }
}

fn wasm_import_module_map(tcx: TyCtxt<'_>, cnum: CrateNum) -> DefIdMap<String> {
    // Build up a map from DefId to a `NativeLib` structure, where
    // `NativeLib` internally contains information about
    // `#[link(wasm_import_module = "...")]` for example.
    let native_libs = tcx.native_libraries(cnum);

    let def_id_to_native_lib = native_libs
        .iter()
        .filter_map(|lib| lib.foreign_module.map(|id| (id, lib)))
        .collect::<DefIdMap<_>>();

    let mut ret = DefIdMap::default();
    for (def_id, lib) in tcx.foreign_modules(cnum).iter() {
        let module = def_id_to_native_lib.get(def_id).and_then(|s| s.wasm_import_module());
        let Some(module) = module else { continue };
        ret.extend(lib.foreign_items.iter().map(|id| {
            assert_eq!(id.krate, cnum);
            (*id, module.to_string())
        }));
    }

    ret
}
