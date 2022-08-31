use rustc_ast::{NestedMetaItem, CRATE_NODE_ID};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::ty::{List, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::cstore::{DllCallingConvention, DllImport, NativeLib, PeImportNameType};
use rustc_session::parse::feature_err;
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;

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
        if !matches!(self.tcx.def_kind(id.def_id), DefKind::ForeignMod) {
            return;
        }

        let it = self.tcx.hir().item(id);
        let hir::ItemKind::ForeignMod { abi, items: foreign_mod_items } = it.kind else {
            return;
        };

        if abi == Abi::Rust || abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
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
                            let msg = "multiple `name` arguments in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_name) = item.value_str() else {
                            let msg = "link name must be of the form `name = \"string\"`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };
                        let span = item.name_value_literal_span().unwrap();
                        if link_name.is_empty() {
                            struct_span_err!(sess, span, E0454, "link name must not be empty")
                                .span_label(span, "empty link name")
                                .emit();
                        }
                        name = Some((link_name, span));
                    }
                    sym::kind => {
                        if kind.is_some() {
                            let msg = "multiple `kind` arguments in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_kind) = item.value_str() else {
                            let msg = "link kind must be of the form `kind = \"string\"`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };

                        let span = item.name_value_literal_span().unwrap();
                        let link_kind = match link_kind.as_str() {
                            "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
                            "dylib" => NativeLibKind::Dylib { as_needed: None },
                            "framework" => {
                                if !sess.target.is_like_osx {
                                    struct_span_err!(
                                        sess,
                                        span,
                                        E0455,
                                        "link kind `framework` is only supported on Apple targets"
                                    )
                                    .emit();
                                }
                                NativeLibKind::Framework { as_needed: None }
                            }
                            "raw-dylib" => {
                                if !sess.target.is_like_windows {
                                    struct_span_err!(
                                        sess,
                                        span,
                                        E0455,
                                        "link kind `raw-dylib` is only supported on Windows targets"
                                    )
                                    .emit();
                                } else if !features.raw_dylib {
                                    feature_err(
                                        &sess.parse_sess,
                                        sym::raw_dylib,
                                        span,
                                        "link kind `raw-dylib` is unstable",
                                    )
                                    .emit();
                                }
                                NativeLibKind::RawDylib
                            }
                            kind => {
                                let msg = format!(
                                    "unknown link kind `{kind}`, expected one of: \
                                     static, dylib, framework, raw-dylib"
                                );
                                struct_span_err!(sess, span, E0458, "{}", msg)
                                    .span_label(span, "unknown link kind")
                                    .emit();
                                continue;
                            }
                        };
                        kind = Some(link_kind);
                    }
                    sym::modifiers => {
                        if modifiers.is_some() {
                            let msg =
                                "multiple `modifiers` arguments in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_modifiers) = item.value_str() else {
                            let msg = "link modifiers must be of the form `modifiers = \"string\"`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };
                        modifiers = Some((link_modifiers, item.name_value_literal_span().unwrap()));
                    }
                    sym::cfg => {
                        if cfg.is_some() {
                            let msg = "multiple `cfg` arguments in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_cfg) = item.meta_item_list() else {
                            let msg = "link cfg must be of the form `cfg(/* predicate */)`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };
                        let [NestedMetaItem::MetaItem(link_cfg)] = link_cfg else {
                            let msg = "link cfg must have a single predicate argument";
                            sess.span_err(item.span(), msg);
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
                            let msg = "multiple `wasm_import_module` arguments \
                                       in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_wasm_import_module) = item.value_str() else {
                            let msg = "wasm import module must be of the form \
                                       `wasm_import_module = \"string\"`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };
                        wasm_import_module = Some((link_wasm_import_module, item.span()));
                    }
                    sym::import_name_type => {
                        if import_name_type.is_some() {
                            let msg = "multiple `import_name_type` arguments in a single `#[link]` attribute";
                            sess.span_err(item.span(), msg);
                            continue;
                        }
                        let Some(link_import_name_type) = item.value_str() else {
                            let msg = "import name type must be of the form `import_name_type = \"string\"`";
                            sess.span_err(item.span(), msg);
                            continue;
                        };
                        if self.tcx.sess.target.arch != "x86" {
                            let msg = "import name type is only supported on x86";
                            sess.span_err(item.span(), msg);
                            continue;
                        }

                        let link_import_name_type = match link_import_name_type.as_str() {
                            "decorated" => PeImportNameType::Decorated,
                            "noprefix" => PeImportNameType::NoPrefix,
                            "undecorated" => PeImportNameType::Undecorated,
                            import_name_type => {
                                let msg = format!(
                                    "unknown import name type `{import_name_type}`, expected one of: \
                                     decorated, noprefix, undecorated"
                                );
                                sess.span_err(item.span(), msg);
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
                        let msg = "unexpected `#[link]` argument, expected one of: \
                                   name, kind, modifiers, cfg, wasm_import_module, import_name_type";
                        sess.span_err(item.span(), msg);
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
                            sess.span_err(
                                span,
                                "invalid linking modifier syntax, expected '+' or '-' prefix \
                                before one of: bundle, verbatim, whole-archive, as-needed",
                            );
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
                            let msg = format!(
                                "multiple `{modifier}` modifiers in a single `modifiers` argument"
                            );
                            sess.span_err(span, &msg);
                        } else {
                            *dst = Some(value);
                        }
                    };
                    match (modifier, &mut kind) {
                        ("bundle", Some(NativeLibKind::Static { bundle, .. })) => {
                            assign_modifier(bundle)
                        }
                        ("bundle", _) => {
                            sess.span_err(
                                span,
                                "linking modifier `bundle` is only compatible with \
                                 `static` linking kind",
                            );
                        }

                        ("verbatim", _) => {
                            report_unstable_modifier!(native_link_modifiers_verbatim);
                            assign_modifier(&mut verbatim)
                        }

                        ("whole-archive", Some(NativeLibKind::Static { whole_archive, .. })) => {
                            assign_modifier(whole_archive)
                        }
                        ("whole-archive", _) => {
                            sess.span_err(
                                span,
                                "linking modifier `whole-archive` is only compatible with \
                                 `static` linking kind",
                            );
                        }

                        ("as-needed", Some(NativeLibKind::Dylib { as_needed }))
                        | ("as-needed", Some(NativeLibKind::Framework { as_needed })) => {
                            report_unstable_modifier!(native_link_modifiers_as_needed);
                            assign_modifier(as_needed)
                        }
                        ("as-needed", _) => {
                            sess.span_err(
                                span,
                                "linking modifier `as-needed` is only compatible with \
                                 `dylib` and `framework` linking kinds",
                            );
                        }

                        _ => {
                            sess.span_err(
                                span,
                                format!(
                                    "unknown linking modifier `{modifier}`, expected one of: \
                                     bundle, verbatim, whole-archive, as-needed"
                                ),
                            );
                        }
                    }
                }
            }

            if let Some((_, span)) = wasm_import_module {
                if name.is_some() || kind.is_some() || modifiers.is_some() || cfg.is_some() {
                    let msg = "`wasm_import_module` is incompatible with \
                               other arguments in `#[link]` attributes";
                    sess.span_err(span, msg);
                }
            } else if name.is_none() {
                struct_span_err!(
                    sess,
                    m.span,
                    E0459,
                    "`#[link]` attribute requires a `name = \"string\"` argument"
                )
                .span_label(m.span, "missing `name` argument")
                .emit();
            }

            // Do this outside of the loop so that `import_name_type` can be specified before `kind`.
            if let Some((_, span)) = import_name_type {
                if kind != Some(NativeLibKind::RawDylib) {
                    let msg = "import name type can only be used with link kind `raw-dylib`";
                    sess.span_err(span, msg);
                }
            }

            let dll_imports = match kind {
                Some(NativeLibKind::RawDylib) => {
                    if let Some((name, span)) = name && name.as_str().contains('\0') {
                        sess.span_err(
                            span,
                            "link name must not contain NUL characters if link kind is `raw-dylib`",
                        );
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
                        if self.tcx.def_kind(child_item.id.def_id).has_codegen_attrs()
                            && self
                                .tcx
                                .codegen_fn_attrs(child_item.id.def_id)
                                .link_ordinal
                                .is_some()
                        {
                            let link_ordinal_attr = self
                                .tcx
                                .hir()
                                .attrs(self.tcx.hir().local_def_id_to_hir_id(child_item.id.def_id))
                                .iter()
                                .find(|a| a.has_name(sym::link_ordinal))
                                .unwrap();
                            sess.span_err(
                                link_ordinal_attr.span,
                                "`#[link_ordinal]` is only supported if link kind is `raw-dylib`",
                            );
                        }
                    }

                    Vec::new()
                }
            };
            self.libs.push(NativeLib {
                name: name.map(|(name, _)| name),
                kind: kind.unwrap_or(NativeLibKind::Unspecified),
                cfg,
                foreign_module: Some(it.def_id.to_def_id()),
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
                self.tcx.sess.err("library kind `framework` is only supported on Apple targets");
            }
            if let Some(ref new_name) = lib.new_name {
                let any_duplicate = self
                    .libs
                    .iter()
                    .filter_map(|lib| lib.name.as_ref())
                    .any(|n| n.as_str() == lib.name);
                if new_name.is_empty() {
                    self.tcx.sess.err(format!(
                        "an empty renaming target was specified for library `{}`",
                        lib.name
                    ));
                } else if !any_duplicate {
                    self.tcx.sess.err(format!(
                        "renaming of the library `{}` was specified, \
                                                however this crate contains no `#[link(...)]` \
                                                attributes referencing this library",
                        lib.name
                    ));
                } else if !renames.insert(&lib.name) {
                    self.tcx.sess.err(format!(
                        "multiple renamings were \
                                                specified for library `{}`",
                        lib.name
                    ));
                }
            }
        }

        // Update kind and, optionally, the name of all native libraries
        // (there may be more than one) with the specified name.  If any
        // library is mentioned more than once, keep the latest mention
        // of it, so that any possible dependent libraries appear before
        // it.  (This ensures that the linker is able to see symbols from
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
                                let msg = "overriding linking modifiers from command line is not supported";
                                match lib.foreign_module {
                                    Some(def_id) => self.tcx.sess.span_err(self.tcx.def_span(def_id), msg),
                                    None => self.tcx.sess.err(msg),
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
                self.libs.push(NativeLib {
                    name: Some(Symbol::intern(new_name.unwrap_or(&passed_lib.name))),
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
                .type_of(item.id.def_id)
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
                    self.tcx.sess.span_fatal(
                        item.span,
                        r#"ABI not supported by `#[link(kind = "raw-dylib")]` on i686"#,
                    );
                }
            }
        } else {
            match abi {
                Abi::C { .. } | Abi::Win64 { .. } | Abi::System { .. } => DllCallingConvention::C,
                _ => {
                    self.tcx.sess.span_fatal(
                        item.span,
                        r#"ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture"#,
                    );
                }
            }
        };

        let import_name_type = self
            .tcx
            .codegen_fn_attrs(item.id.def_id)
            .link_ordinal
            .map_or(import_name_type, |ord| Some(PeImportNameType::Ordinal(ord)));

        DllImport {
            name: item.ident.name,
            import_name_type,
            calling_convention,
            span: item.span,
            is_fn: self.tcx.def_kind(item.id.def_id).is_fn_like(),
        }
    }
}
