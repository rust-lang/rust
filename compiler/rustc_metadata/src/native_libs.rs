use rustc_ast::{NestedMetaItem, CRATE_NODE_ID};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::{List, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::config::CrateType;
use rustc_session::cstore::{
    DllCallingConvention, DllImport, ForeignModule, NativeLib, PeImportNameType,
};
use rustc_session::parse::feature_err;
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;

use crate::errors;

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

    sess.dcx().emit_fatal(errors::MissingNativeLibrary::new(name, verbatim));
}

fn find_bundled_library(
    name: Symbol,
    verbatim: Option<bool>,
    kind: NativeLibKind,
    has_cfg: bool,
    tcx: TyCtxt<'_>,
) -> Option<Symbol> {
    let sess = tcx.sess;
    if let NativeLibKind::Static { bundle: Some(true) | None, whole_archive } = kind
        && tcx.crate_types().iter().any(|t| matches!(t, &CrateType::Rlib | CrateType::Staticlib))
        && (sess.opts.unstable_opts.packed_bundled_libs || has_cfg || whole_archive == Some(true))
    {
        let verbatim = verbatim.unwrap_or(false);
        let search_paths = &sess.target_filesearch(PathKind::Native).search_path_dirs();
        return find_native_static_library(name.as_str(), verbatim, search_paths, sess)
            .file_name()
            .and_then(|s| s.to_str())
            .map(Symbol::intern);
    }
    None
}

pub(crate) fn collect(tcx: TyCtxt<'_>, LocalCrate: LocalCrate) -> Vec<NativeLib> {
    let mut collector = Collector { tcx, libs: Vec::new() };
    if tcx.sess.opts.unstable_opts.link_directives {
        for module in tcx.foreign_modules(LOCAL_CRATE).values() {
            collector.process_module(module);
        }
    }
    collector.process_command_line();
    collector.libs
}

pub(crate) fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, sess, CRATE_NODE_ID, None),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLib>,
}

impl<'tcx> Collector<'tcx> {
    fn process_module(&mut self, module: &ForeignModule) {
        let ForeignModule { def_id, abi, ref foreign_items } = *module;
        let def_id = def_id.expect_local();

        let sess = self.tcx.sess;

        if matches!(abi, Abi::Rust | Abi::RustIntrinsic) {
            return;
        }

        // Process all of the #[link(..)]-style arguments
        let features = self.tcx.features();

        for m in self.tcx.get_attrs(def_id, sym::link) {
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
                            sess.dcx().emit_err(errors::MultipleNamesInLink { span: item.span() });
                            continue;
                        }
                        let Some(link_name) = item.value_str() else {
                            sess.dcx().emit_err(errors::LinkNameForm { span: item.span() });
                            continue;
                        };
                        let span = item.name_value_literal_span().unwrap();
                        if link_name.is_empty() {
                            sess.dcx().emit_err(errors::EmptyLinkName { span });
                        }
                        name = Some((link_name, span));
                    }
                    sym::kind => {
                        if kind.is_some() {
                            sess.dcx().emit_err(errors::MultipleKindsInLink { span: item.span() });
                            continue;
                        }
                        let Some(link_kind) = item.value_str() else {
                            sess.dcx().emit_err(errors::LinkKindForm { span: item.span() });
                            continue;
                        };

                        let span = item.name_value_literal_span().unwrap();
                        let link_kind = match link_kind.as_str() {
                            "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
                            "dylib" => NativeLibKind::Dylib { as_needed: None },
                            "framework" => {
                                if !sess.target.is_like_osx {
                                    sess.dcx().emit_err(errors::LinkFrameworkApple { span });
                                }
                                NativeLibKind::Framework { as_needed: None }
                            }
                            "raw-dylib" => {
                                if !sess.target.is_like_windows {
                                    sess.dcx().emit_err(errors::FrameworkOnlyWindows { span });
                                }
                                NativeLibKind::RawDylib
                            }
                            "link-arg" => {
                                if !features.link_arg_attribute {
                                    feature_err(
                                        sess,
                                        sym::link_arg_attribute,
                                        span,
                                        "link kind `link-arg` is unstable",
                                    )
                                    .emit();
                                }
                                NativeLibKind::LinkArg
                            }
                            kind => {
                                sess.dcx().emit_err(errors::UnknownLinkKind { span, kind });
                                continue;
                            }
                        };
                        kind = Some(link_kind);
                    }
                    sym::modifiers => {
                        if modifiers.is_some() {
                            sess.dcx()
                                .emit_err(errors::MultipleLinkModifiers { span: item.span() });
                            continue;
                        }
                        let Some(link_modifiers) = item.value_str() else {
                            sess.dcx().emit_err(errors::LinkModifiersForm { span: item.span() });
                            continue;
                        };
                        modifiers = Some((link_modifiers, item.name_value_literal_span().unwrap()));
                    }
                    sym::cfg => {
                        if cfg.is_some() {
                            sess.dcx().emit_err(errors::MultipleCfgs { span: item.span() });
                            continue;
                        }
                        let Some(link_cfg) = item.meta_item_list() else {
                            sess.dcx().emit_err(errors::LinkCfgForm { span: item.span() });
                            continue;
                        };
                        let [NestedMetaItem::MetaItem(link_cfg)] = link_cfg else {
                            sess.dcx()
                                .emit_err(errors::LinkCfgSinglePredicate { span: item.span() });
                            continue;
                        };
                        if !features.link_cfg {
                            feature_err(sess, sym::link_cfg, item.span(), "link cfg is unstable")
                                .emit();
                        }
                        cfg = Some(link_cfg.clone());
                    }
                    sym::wasm_import_module => {
                        if wasm_import_module.is_some() {
                            sess.dcx().emit_err(errors::MultipleWasmImport { span: item.span() });
                            continue;
                        }
                        let Some(link_wasm_import_module) = item.value_str() else {
                            sess.dcx().emit_err(errors::WasmImportForm { span: item.span() });
                            continue;
                        };
                        wasm_import_module = Some((link_wasm_import_module, item.span()));
                    }
                    sym::import_name_type => {
                        if import_name_type.is_some() {
                            sess.dcx()
                                .emit_err(errors::MultipleImportNameType { span: item.span() });
                            continue;
                        }
                        let Some(link_import_name_type) = item.value_str() else {
                            sess.dcx().emit_err(errors::ImportNameTypeForm { span: item.span() });
                            continue;
                        };
                        if self.tcx.sess.target.arch != "x86" {
                            sess.dcx().emit_err(errors::ImportNameTypeX86 { span: item.span() });
                            continue;
                        }

                        let link_import_name_type = match link_import_name_type.as_str() {
                            "decorated" => PeImportNameType::Decorated,
                            "noprefix" => PeImportNameType::NoPrefix,
                            "undecorated" => PeImportNameType::Undecorated,
                            import_name_type => {
                                sess.dcx().emit_err(errors::UnknownImportNameType {
                                    span: item.span(),
                                    import_name_type,
                                });
                                continue;
                            }
                        };
                        import_name_type = Some((link_import_name_type, item.span()));
                    }
                    _ => {
                        sess.dcx().emit_err(errors::UnexpectedLinkArg { span: item.span() });
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
                            sess.dcx().emit_err(errors::InvalidLinkModifier { span });
                            continue;
                        }
                    };

                    macro report_unstable_modifier($feature: ident) {
                        if !features.$feature {
                            feature_err(
                                sess,
                                sym::$feature,
                                span,
                                format!("linking modifier `{modifier}` is unstable"),
                            )
                            .emit();
                        }
                    }
                    let assign_modifier = |dst: &mut Option<bool>| {
                        if dst.is_some() {
                            sess.dcx().emit_err(errors::MultipleModifiers { span, modifier });
                        } else {
                            *dst = Some(value);
                        }
                    };
                    match (modifier, &mut kind) {
                        ("bundle", Some(NativeLibKind::Static { bundle, .. })) => {
                            assign_modifier(bundle)
                        }
                        ("bundle", _) => {
                            sess.dcx().emit_err(errors::BundleNeedsStatic { span });
                        }

                        ("verbatim", _) => assign_modifier(&mut verbatim),

                        ("whole-archive", Some(NativeLibKind::Static { whole_archive, .. })) => {
                            assign_modifier(whole_archive)
                        }
                        ("whole-archive", _) => {
                            sess.dcx().emit_err(errors::WholeArchiveNeedsStatic { span });
                        }

                        ("as-needed", Some(NativeLibKind::Dylib { as_needed }))
                        | ("as-needed", Some(NativeLibKind::Framework { as_needed })) => {
                            report_unstable_modifier!(native_link_modifiers_as_needed);
                            assign_modifier(as_needed)
                        }
                        ("as-needed", _) => {
                            sess.dcx().emit_err(errors::AsNeededCompatibility { span });
                        }

                        _ => {
                            sess.dcx().emit_err(errors::UnknownLinkModifier { span, modifier });
                        }
                    }
                }
            }

            if let Some((_, span)) = wasm_import_module {
                if name.is_some() || kind.is_some() || modifiers.is_some() || cfg.is_some() {
                    sess.dcx().emit_err(errors::IncompatibleWasmLink { span });
                }
            }

            if wasm_import_module.is_some() {
                (name, kind) = (wasm_import_module, Some(NativeLibKind::WasmImportModule));
            }
            let Some((name, name_span)) = name else {
                sess.dcx().emit_err(errors::LinkRequiresName { span: m.span });
                continue;
            };

            // Do this outside of the loop so that `import_name_type` can be specified before `kind`.
            if let Some((_, span)) = import_name_type {
                if kind != Some(NativeLibKind::RawDylib) {
                    sess.dcx().emit_err(errors::ImportNameTypeRaw { span });
                }
            }

            let dll_imports = match kind {
                Some(NativeLibKind::RawDylib) => {
                    if name.as_str().contains('\0') {
                        sess.dcx().emit_err(errors::RawDylibNoNul { span: name_span });
                    }
                    foreign_items
                        .iter()
                        .map(|&child_item| {
                            self.build_dll_import(
                                abi,
                                import_name_type.map(|(import_name_type, _)| import_name_type),
                                child_item,
                            )
                        })
                        .collect()
                }
                _ => {
                    for &child_item in foreign_items {
                        if self.tcx.def_kind(child_item).has_codegen_attrs()
                            && self.tcx.codegen_fn_attrs(child_item).link_ordinal.is_some()
                        {
                            let link_ordinal_attr =
                                self.tcx.get_attr(child_item, sym::link_ordinal).unwrap();
                            sess.dcx().emit_err(errors::LinkOrdinalRawDylib {
                                span: link_ordinal_attr.span,
                            });
                        }
                    }

                    Vec::new()
                }
            };

            let kind = kind.unwrap_or(NativeLibKind::Unspecified);
            let filename = find_bundled_library(name, verbatim, kind, cfg.is_some(), self.tcx);
            self.libs.push(NativeLib {
                name,
                filename,
                kind,
                cfg,
                foreign_module: Some(def_id.to_def_id()),
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
            if let NativeLibKind::Framework { .. } = lib.kind
                && !self.tcx.sess.target.is_like_osx
            {
                // Cannot check this when parsing options because the target is not yet available.
                self.tcx.dcx().emit_err(errors::LibFrameworkApple);
            }
            if let Some(ref new_name) = lib.new_name {
                let any_duplicate = self.libs.iter().any(|n| n.name.as_str() == lib.name);
                if new_name.is_empty() {
                    self.tcx.dcx().emit_err(errors::EmptyRenamingTarget { lib_name: &lib.name });
                } else if !any_duplicate {
                    self.tcx.dcx().emit_err(errors::RenamingNoLink { lib_name: &lib.name });
                } else if !renames.insert(&lib.name) {
                    self.tcx.dcx().emit_err(errors::MultipleRenamings { lib_name: &lib.name });
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
                .extract_if(|lib| {
                    if lib.name.as_str() == passed_lib.name {
                        // FIXME: This whole logic is questionable, whether modifiers are
                        // involved or not, library reordering and kind overriding without
                        // explicit `:rename` in particular.
                        if lib.has_modifiers() || passed_lib.has_modifiers() {
                            match lib.foreign_module {
                                Some(def_id) => {
                                    self.tcx.dcx().emit_err(errors::NoLinkModOverride {
                                        span: Some(self.tcx.def_span(def_id)),
                                    })
                                }
                                None => self
                                    .tcx
                                    .dcx()
                                    .emit_err(errors::NoLinkModOverride { span: None }),
                            };
                        }
                        if passed_lib.kind != NativeLibKind::Unspecified {
                            lib.kind = passed_lib.kind;
                        }
                        if let Some(new_name) = &passed_lib.new_name {
                            lib.name = Symbol::intern(new_name);
                        }
                        lib.verbatim = passed_lib.verbatim;
                        return true;
                    }
                    false
                })
                .collect::<Vec<_>>();
            if existing.is_empty() {
                // Add if not found
                let new_name: Option<&str> = passed_lib.new_name.as_deref();
                let name = Symbol::intern(new_name.unwrap_or(&passed_lib.name));
                let filename = find_bundled_library(
                    name,
                    passed_lib.verbatim,
                    passed_lib.kind,
                    false,
                    self.tcx,
                );
                self.libs.push(NativeLib {
                    name,
                    filename,
                    kind: passed_lib.kind,
                    cfg: None,
                    foreign_module: None,
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

    fn i686_arg_list_size(&self, item: DefId) -> usize {
        let argument_types: &List<Ty<'_>> = self.tcx.instantiate_bound_regions_with_erased(
            self.tcx
                .type_of(item)
                .instantiate_identity()
                .fn_sig(self.tcx)
                .inputs()
                .map_bound(|slice| self.tcx.mk_type_list(slice)),
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
        item: DefId,
    ) -> DllImport {
        let span = self.tcx.def_span(item);

        // this logic is similar to `Target::adjust_abi` (in rustc_target/src/spec/mod.rs) but errors on unsupported inputs
        let calling_convention = if self.tcx.sess.target.arch == "x86" {
            match abi {
                Abi::C { .. } | Abi::Cdecl { .. } => DllCallingConvention::C,
                Abi::Stdcall { .. } => DllCallingConvention::Stdcall(self.i686_arg_list_size(item)),
                // On Windows, `extern "system"` behaves like msvc's `__stdcall`.
                // `__stdcall` only applies on x86 and on non-variadic functions:
                // https://learn.microsoft.com/en-us/cpp/cpp/stdcall?view=msvc-170
                Abi::System { .. } => {
                    let c_variadic =
                        self.tcx.type_of(item).instantiate_identity().fn_sig(self.tcx).c_variadic();

                    if c_variadic {
                        DllCallingConvention::C
                    } else {
                        DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                    }
                }
                Abi::Fastcall { .. } => {
                    DllCallingConvention::Fastcall(self.i686_arg_list_size(item))
                }
                Abi::Vectorcall { .. } => {
                    DllCallingConvention::Vectorcall(self.i686_arg_list_size(item))
                }
                _ => {
                    self.tcx.dcx().emit_fatal(errors::UnsupportedAbiI686 { span });
                }
            }
        } else {
            match abi {
                Abi::C { .. } | Abi::Win64 { .. } | Abi::System { .. } => DllCallingConvention::C,
                _ => {
                    self.tcx.dcx().emit_fatal(errors::UnsupportedAbi { span });
                }
            }
        };

        let codegen_fn_attrs = self.tcx.codegen_fn_attrs(item);
        let import_name_type = codegen_fn_attrs
            .link_ordinal
            .map_or(import_name_type, |ord| Some(PeImportNameType::Ordinal(ord)));

        DllImport {
            name: codegen_fn_attrs.link_name.unwrap_or(self.tcx.item_name(item)),
            import_name_type,
            calling_convention,
            span,
            is_fn: self.tcx.def_kind(item).is_fn_like(),
        }
    }
}
