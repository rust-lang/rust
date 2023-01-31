use rustc_ast::{NestedMetaItem, CRATE_NODE_ID};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::ty::{List, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::config::CrateType;
use rustc_session::cstore::{DllCallingConvention, DllImport, NativeLib, PeImportNameType};
use rustc_session::parse::feature_err;
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;

use crate::errors::{
    AsNeededCompatibility, BundleNeedsStatic, EmptyLinkName, EmptyRenamingTarget,
    FrameworkOnlyWindows, ImportNameTypeForm, ImportNameTypeRaw, ImportNameTypeX86,
    IncompatibleWasmLink, InvalidLinkModifier, LibFrameworkApple, LinkCfgForm,
    LinkCfgSinglePredicate, LinkFrameworkApple, LinkKindForm, LinkModifiersForm, LinkNameForm,
    LinkOrdinalRawDylib, LinkRequiresName, MissingNativeLibrary, MultipleCfgs,
    MultipleImportNameType, MultipleKindsInLink, MultipleLinkModifiers, MultipleModifiers,
    MultipleNamesInLink, MultipleRenamings, MultipleWasmImport, NoLinkModOverride, RawDylibNoNul,
    RenamingNoLink, UnexpectedLinkArg, UnknownImportNameType, UnknownLinkKind, UnknownLinkModifier,
    UnsupportedAbi, UnsupportedAbiI686, WasmImportForm, WholeArchiveNeedsStatic,
};

use std::path::PathBuf;

pub fn find_native_static_library(
    name: &str,
    verbatim: bool,
    search_paths: &[PathBuf],
    sess: &Session,
) -> PathBuf {
    let formats = if verbatim {
        vec![("".into(), "".into())]
    } else {
        let os = (sess.target.staticlib_prefix.clone(), sess.target.staticlib_suffix.clone());
        // On Windows, static libraries sometimes show up as libfoo.a and other
        // times show up as foo.lib
        let unix = ("lib".into(), ".a".into());
        if os == unix { vec![os] } else { vec![os, unix] }
    };

    for path in search_paths {
        for (prefix, suffix) in &formats {
            let test = path.join(format!("{prefix}{name}{suffix}"));
            if test.exists() {
                return test;
            }
        }
    }

    sess.emit_fatal(MissingNativeLibrary::new(name, verbatim));
}

fn find_bundled_library(
    name: Option<Symbol>,
    verbatim: Option<bool>,
    kind: NativeLibKind,
    sess: &Session,
) -> Option<Symbol> {
    if sess.opts.unstable_opts.packed_bundled_libs &&
            sess.crate_types().iter().any(|ct| ct == &CrateType::Rlib || ct == &CrateType::Staticlib) &&
            let NativeLibKind::Static { bundle: Some(true) | None, .. } = kind {
        find_native_static_library(
            name.unwrap().as_str(),
            verbatim.unwrap_or(false),
            &sess.target_filesearch(PathKind::Native).search_path_dirs(),
            sess,
        ).file_name().and_then(|s| s.to_str()).map(Symbol::intern)
    } else {
        None
    }
}

pub(crate) fn collect(tcx: TyCtxt<'_>) -> Vec<NativeLib> {
    let mut collector = Collector { tcx, libs: Vec::new() };
    for id in tcx.hir().items() {
        collector.process_item(id);
    }
    collector.process_command_line();
    collector.libs
}

pub(crate) fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, CRATE_NODE_ID, None),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLib>,
}

impl<'tcx> Collector<'tcx> {
    fn process_item(&mut self, id: rustc_hir::ItemId) {
        if !matches!(self.tcx.def_kind(id.owner_id), DefKind::ForeignMod) {
            return;
        }

        let it = self.tcx.hir().item(id);
        let hir::ItemKind::ForeignMod { abi, items: foreign_mod_items } = it.kind else {
            return;
        };

        if matches!(abi, Abi::Rust | Abi::RustIntrinsic | Abi::PlatformIntrinsic) {
            return;
        }

        // Process all of the #[link(..)]-style arguments
        let sess = &self.tcx.sess;
        let features = self.tcx.features();
        for m in self.tcx.hir().attrs(it.hir_id()).iter().filter(|a| a.has_name(sym::link)) {
            let Some(items) = m.meta_item_list() else {
                continue;
            };

            let mut name = None;
            let mut kind = None;
            let mut modifiers = None;
            let mut cfg = None;
            let mut wasm_import_module = None;
            let mut import_name_type = None;
            for item in items.iter() {
                match item.name_or_empty() {
                    sym::name => {
                        if name.is_some() {
                            sess.emit_err(MultipleNamesInLink { span: item.span() });
                            continue;
                        }
                        let Some(link_name) = item.value_str() else {
                            sess.emit_err(LinkNameForm { span: item.span() });
                            continue;
                        };
                        let span = item.name_value_literal_span().unwrap();
                        if link_name.is_empty() {
                            sess.emit_err(EmptyLinkName { span });
                        }
                        name = Some((link_name, span));
                    }
                    sym::kind => {
                        if kind.is_some() {
                            sess.emit_err(MultipleKindsInLink { span: item.span() });
                            continue;
                        }
                        let Some(link_kind) = item.value_str() else {
                            sess.emit_err(LinkKindForm { span: item.span() });
                            continue;
                        };

                        let span = item.name_value_literal_span().unwrap();
                        let link_kind = match link_kind.as_str() {
                            "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
                            "dylib" => NativeLibKind::Dylib { as_needed: None },
                            "framework" => {
                                if !sess.target.is_like_osx {
                                    sess.emit_err(LinkFrameworkApple { span });
                                }
                                NativeLibKind::Framework { as_needed: None }
                            }
                            "raw-dylib" => {
                                if !sess.target.is_like_windows {
                                    sess.emit_err(FrameworkOnlyWindows { span });
                                } else if !features.raw_dylib && sess.target.arch == "x86" {
                                    feature_err(
                                        &sess.parse_sess,
                                        sym::raw_dylib,
                                        span,
                                        "link kind `raw-dylib` is unstable on x86",
                                    )
                                    .emit();
                                }
                                NativeLibKind::RawDylib
                            }
                            kind => {
                                sess.emit_err(UnknownLinkKind { span, kind });
                                continue;
                            }
                        };
                        kind = Some(link_kind);
                    }
                    sym::modifiers => {
                        if modifiers.is_some() {
                            sess.emit_err(MultipleLinkModifiers { span: item.span() });
                            continue;
                        }
                        let Some(link_modifiers) = item.value_str() else {
                            sess.emit_err(LinkModifiersForm { span: item.span() });
                            continue;
                        };
                        modifiers = Some((link_modifiers, item.name_value_literal_span().unwrap()));
                    }
                    sym::cfg => {
                        if cfg.is_some() {
                            sess.emit_err(MultipleCfgs { span: item.span() });
                            continue;
                        }
                        let Some(link_cfg) = item.meta_item_list() else {
                            sess.emit_err(LinkCfgForm { span: item.span() });
                            continue;
                        };
                        let [NestedMetaItem::MetaItem(link_cfg)] = link_cfg else {
                            sess.emit_err(LinkCfgSinglePredicate { span: item.span() });
                            continue;
                        };
                        if !features.link_cfg {
                            feature_err(
                                &sess.parse_sess,
                                sym::link_cfg,
                                item.span(),
                                "link cfg is unstable",
                            )
                            .emit();
                        }
                        cfg = Some(link_cfg.clone());
                    }
                    sym::wasm_import_module => {
                        if wasm_import_module.is_some() {
                            sess.emit_err(MultipleWasmImport { span: item.span() });
                            continue;
                        }
                        let Some(link_wasm_import_module) = item.value_str() else {
                            sess.emit_err(WasmImportForm { span: item.span() });
                            continue;
                        };
                        wasm_import_module = Some((link_wasm_import_module, item.span()));
                    }
                    sym::import_name_type => {
                        if import_name_type.is_some() {
                            sess.emit_err(MultipleImportNameType { span: item.span() });
                            continue;
                        }
                        let Some(link_import_name_type) = item.value_str() else {
                            sess.emit_err(ImportNameTypeForm { span: item.span() });
                            continue;
                        };
                        if self.tcx.sess.target.arch != "x86" {
                            sess.emit_err(ImportNameTypeX86 { span: item.span() });
                            continue;
                        }

                        let link_import_name_type = match link_import_name_type.as_str() {
                            "decorated" => PeImportNameType::Decorated,
                            "noprefix" => PeImportNameType::NoPrefix,
                            "undecorated" => PeImportNameType::Undecorated,
                            import_name_type => {
                                sess.emit_err(UnknownImportNameType {
                                    span: item.span(),
                                    import_name_type,
                                });
                                continue;
                            }
                        };
                        if !features.raw_dylib {
                            let span = item.name_value_literal_span().unwrap();
                            feature_err(
                                &sess.parse_sess,
                                sym::raw_dylib,
                                span,
                                "import name type is unstable",
                            )
                            .emit();
                        }
                        import_name_type = Some((link_import_name_type, item.span()));
                    }
                    _ => {
                        sess.emit_err(UnexpectedLinkArg { span: item.span() });
                    }
                }
            }

            // Do this outside the above loop so we don't depend on modifiers coming after kinds
            let mut verbatim = None;
            if let Some((modifiers, span)) = modifiers {
                for modifier in modifiers.as_str().split(',') {
                    let (modifier, value) = match modifier.strip_prefix(&['+', '-']) {
                        Some(m) => (m, modifier.starts_with('+')),
                        None => {
                            sess.emit_err(InvalidLinkModifier { span });
                            continue;
                        }
                    };

                    macro report_unstable_modifier($feature: ident) {
                        if !features.$feature {
                            feature_err(
                                &sess.parse_sess,
                                sym::$feature,
                                span,
                                &format!("linking modifier `{modifier}` is unstable"),
                            )
                            .emit();
                        }
                    }
                    let assign_modifier = |dst: &mut Option<bool>| {
                        if dst.is_some() {
                            sess.emit_err(MultipleModifiers { span, modifier });
                        } else {
                            *dst = Some(value);
                        }
                    };
                    match (modifier, &mut kind) {
                        ("bundle", Some(NativeLibKind::Static { bundle, .. })) => {
                            assign_modifier(bundle)
                        }
                        ("bundle", _) => {
                            sess.emit_err(BundleNeedsStatic { span });
                        }

                        ("verbatim", _) => assign_modifier(&mut verbatim),

                        ("whole-archive", Some(NativeLibKind::Static { whole_archive, .. })) => {
                            assign_modifier(whole_archive)
                        }
                        ("whole-archive", _) => {
                            sess.emit_err(WholeArchiveNeedsStatic { span });
                        }

                        ("as-needed", Some(NativeLibKind::Dylib { as_needed }))
                        | ("as-needed", Some(NativeLibKind::Framework { as_needed })) => {
                            report_unstable_modifier!(native_link_modifiers_as_needed);
                            assign_modifier(as_needed)
                        }
                        ("as-needed", _) => {
                            sess.emit_err(AsNeededCompatibility { span });
                        }

                        _ => {
                            sess.emit_err(UnknownLinkModifier { span, modifier });
                        }
                    }
                }
            }

            if let Some((_, span)) = wasm_import_module {
                if name.is_some() || kind.is_some() || modifiers.is_some() || cfg.is_some() {
                    sess.emit_err(IncompatibleWasmLink { span });
                }
            } else if name.is_none() {
                sess.emit_err(LinkRequiresName { span: m.span });
            }

            // Do this outside of the loop so that `import_name_type` can be specified before `kind`.
            if let Some((_, span)) = import_name_type {
                if kind != Some(NativeLibKind::RawDylib) {
                    sess.emit_err(ImportNameTypeRaw { span });
                }
            }

            let dll_imports = match kind {
                Some(NativeLibKind::RawDylib) => {
                    if let Some((name, span)) = name && name.as_str().contains('\0') {
                        sess.emit_err(RawDylibNoNul { span });
                    }
                    foreign_mod_items
                        .iter()
                        .map(|child_item| {
                            self.build_dll_import(
                                abi,
                                import_name_type.map(|(import_name_type, _)| import_name_type),
                                child_item,
                            )
                        })
                        .collect()
                }
                _ => {
                    for child_item in foreign_mod_items {
                        if self.tcx.def_kind(child_item.id.owner_id).has_codegen_attrs()
                            && self
                                .tcx
                                .codegen_fn_attrs(child_item.id.owner_id)
                                .link_ordinal
                                .is_some()
                        {
                            let link_ordinal_attr = self
                                .tcx
                                .hir()
                                .attrs(child_item.id.owner_id.into())
                                .iter()
                                .find(|a| a.has_name(sym::link_ordinal))
                                .unwrap();
                            sess.emit_err(LinkOrdinalRawDylib { span: link_ordinal_attr.span });
                        }
                    }

                    Vec::new()
                }
            };

            let name = name.map(|(name, _)| name);
            let kind = kind.unwrap_or(NativeLibKind::Unspecified);
            let filename = find_bundled_library(name, verbatim, kind, sess);
            self.libs.push(NativeLib {
                name,
                filename,
                kind,
                cfg,
                foreign_module: Some(it.owner_id.to_def_id()),
                wasm_import_module: wasm_import_module.map(|(name, _)| name),
                verbatim,
                dll_imports,
            });
        }
    }

    // Process libs passed on the command line
    fn process_command_line(&mut self) {
        // First, check for errors
        let mut renames = FxHashSet::default();
        for lib in &self.tcx.sess.opts.libs {
            if let NativeLibKind::Framework { .. } = lib.kind && !self.tcx.sess.target.is_like_osx {
                // Cannot check this when parsing options because the target is not yet available.
                self.tcx.sess.emit_err(LibFrameworkApple);
            }
            if let Some(ref new_name) = lib.new_name {
                let any_duplicate = self
                    .libs
                    .iter()
                    .filter_map(|lib| lib.name.as_ref())
                    .any(|n| n.as_str() == lib.name);
                if new_name.is_empty() {
                    self.tcx.sess.emit_err(EmptyRenamingTarget { lib_name: &lib.name });
                } else if !any_duplicate {
                    self.tcx.sess.emit_err(RenamingNoLink { lib_name: &lib.name });
                } else if !renames.insert(&lib.name) {
                    self.tcx.sess.emit_err(MultipleRenamings { lib_name: &lib.name });
                }
            }
        }

        // Update kind and, optionally, the name of all native libraries
        // (there may be more than one) with the specified name. If any
        // library is mentioned more than once, keep the latest mention
        // of it, so that any possible dependent libraries appear before
        // it. (This ensures that the linker is able to see symbols from
        // all possible dependent libraries before linking in the library
        // in question.)
        for passed_lib in &self.tcx.sess.opts.libs {
            // If we've already added any native libraries with the same
            // name, they will be pulled out into `existing`, so that we
            // can move them to the end of the list below.
            let mut existing = self
                .libs
                .drain_filter(|lib| {
                    if let Some(lib_name) = lib.name {
                        if lib_name.as_str() == passed_lib.name {
                            // FIXME: This whole logic is questionable, whether modifiers are
                            // involved or not, library reordering and kind overriding without
                            // explicit `:rename` in particular.
                            if lib.has_modifiers() || passed_lib.has_modifiers() {
                                match lib.foreign_module {
                                    Some(def_id) => self.tcx.sess.emit_err(NoLinkModOverride {
                                        span: Some(self.tcx.def_span(def_id)),
                                    }),
                                    None => {
                                        self.tcx.sess.emit_err(NoLinkModOverride { span: None })
                                    }
                                };
                            }
                            if passed_lib.kind != NativeLibKind::Unspecified {
                                lib.kind = passed_lib.kind;
                            }
                            if let Some(new_name) = &passed_lib.new_name {
                                lib.name = Some(Symbol::intern(new_name));
                            }
                            lib.verbatim = passed_lib.verbatim;
                            return true;
                        }
                    }
                    false
                })
                .collect::<Vec<_>>();
            if existing.is_empty() {
                // Add if not found
                let new_name: Option<&str> = passed_lib.new_name.as_deref();
                let name = Some(Symbol::intern(new_name.unwrap_or(&passed_lib.name)));
                let sess = self.tcx.sess;
                let filename =
                    find_bundled_library(name, passed_lib.verbatim, passed_lib.kind, sess);
                self.libs.push(NativeLib {
                    name,
                    filename,
                    kind: passed_lib.kind,
                    cfg: None,
                    foreign_module: None,
                    wasm_import_module: None,
                    verbatim: passed_lib.verbatim,
                    dll_imports: Vec::new(),
                });
            } else {
                // Move all existing libraries with the same name to the
                // end of the command line.
                self.libs.append(&mut existing);
            }
        }
    }

    fn i686_arg_list_size(&self, item: &hir::ForeignItemRef) -> usize {
        let argument_types: &List<Ty<'_>> = self.tcx.erase_late_bound_regions(
            self.tcx
                .type_of(item.id.owner_id)
                .fn_sig(self.tcx)
                .inputs()
                .map_bound(|slice| self.tcx.mk_type_list(slice.iter())),
        );

        argument_types
            .iter()
            .map(|ty| {
                let layout = self
                    .tcx
                    .layout_of(ParamEnvAnd { param_env: ParamEnv::empty(), value: ty })
                    .expect("layout")
                    .layout;
                // In both stdcall and fastcall, we always round up the argument size to the
                // nearest multiple of 4 bytes.
                (layout.size().bytes_usize() + 3) & !3
            })
            .sum()
    }

    fn build_dll_import(
        &self,
        abi: Abi,
        import_name_type: Option<PeImportNameType>,
        item: &hir::ForeignItemRef,
    ) -> DllImport {
        let calling_convention = if self.tcx.sess.target.arch == "x86" {
            match abi {
                Abi::C { .. } | Abi::Cdecl { .. } => DllCallingConvention::C,
                Abi::Stdcall { .. } | Abi::System { .. } => {
                    DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                }
                Abi::Fastcall { .. } => {
                    DllCallingConvention::Fastcall(self.i686_arg_list_size(item))
                }
                Abi::Vectorcall { .. } => {
                    DllCallingConvention::Vectorcall(self.i686_arg_list_size(item))
                }
                _ => {
                    self.tcx.sess.emit_fatal(UnsupportedAbiI686 { span: item.span });
                }
            }
        } else {
            match abi {
                Abi::C { .. } | Abi::Win64 { .. } | Abi::System { .. } => DllCallingConvention::C,
                _ => {
                    self.tcx.sess.emit_fatal(UnsupportedAbi { span: item.span });
                }
            }
        };

        let codegen_fn_attrs = self.tcx.codegen_fn_attrs(item.id.owner_id);
        let import_name_type = codegen_fn_attrs
            .link_ordinal
            .map_or(import_name_type, |ord| Some(PeImportNameType::Ordinal(ord)));

        DllImport {
            name: codegen_fn_attrs.link_name.unwrap_or(item.ident.name),
            import_name_type,
            calling_convention,
            span: item.span,
            is_fn: self.tcx.def_kind(item.id.owner_id).is_fn_like(),
        }
    }
}
