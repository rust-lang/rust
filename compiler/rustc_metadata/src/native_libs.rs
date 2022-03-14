use rustc_ast::{NestedMetaItem, CRATE_NODE_ID};
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::ty::{List, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::cstore::{DllCallingConvention, DllImport, NativeLib};
use rustc_session::parse::feature_err;
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;

use std::{iter, mem};

crate fn collect(tcx: TyCtxt<'_>) -> Vec<NativeLib> {
    let mut collector = Collector { tcx, libs: Vec::new(), attr_libs: 0 };
    for id in tcx.hir().items() {
        collector.process_item(id);
    }
    collector.attr_libs = collector.libs.len();
    collector.process_command_line();
    collector.unify_kinds_and_modifiers();
    collector.compat_reorder();
    collector.libs
}

crate fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, CRATE_NODE_ID, None),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLib>,
    attr_libs: usize,
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
                            "static-nobundle" => {
                                sess.struct_span_warn(
                                    span,
                                    "link kind `static-nobundle` has been superseded by specifying \
                                     modifier `-bundle` with link kind `static`",
                                )
                                .emit();
                                if !features.static_nobundle {
                                    feature_err(
                                        &sess.parse_sess,
                                        sym::static_nobundle,
                                        span,
                                        "link kind `static-nobundle` is unstable",
                                    )
                                    .emit();
                                }
                                NativeLibKind::Static { bundle: Some(false), whole_archive: None }
                            }
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
                    _ => {
                        let msg = "unexpected `#[link]` argument, expected one of: \
                                   name, kind, modifiers, cfg, wasm_import_module";
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
                            report_unstable_modifier!(native_link_modifiers_bundle);
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
                        .map(|child_item| self.build_dll_import(abi, child_item))
                        .collect()
                }
                _ => Vec::new(),
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
        // Collect overrides and check them for errors
        let mut overrides = FxHashMap::default();
        for cmd_lib in &self.tcx.sess.opts.libs {
            if let NativeLibKind::Framework { .. } = cmd_lib.kind && !self.tcx.sess.target.is_like_osx {
                // Cannot check this when parsing options because the target is not yet available.
                self.tcx.sess.err("library kind `framework` is only supported on Apple targets");
            }
            if let Some(override_name) = &cmd_lib.new_name {
                if override_name.is_empty() {
                    self.tcx.sess.err(&format!(
                        "empty override name was specified for library `{}`",
                        cmd_lib.name
                    ));
                } else if self
                    .libs
                    .iter()
                    .filter_map(|attr_lib| attr_lib.name)
                    .all(|attr_lib_name| attr_lib_name.as_str() != cmd_lib.name)
                {
                    self.tcx.sess.err(&format!(
                        "override of the library `{}` was specified, however this crate \
                         contains no `#[link]` attributes referencing this library",
                        cmd_lib.name
                    ));
                } else if overrides.insert(&cmd_lib.name, cmd_lib).is_some() {
                    self.tcx.sess.err(&format!(
                        "multiple overrides were specified for library `{}`",
                        cmd_lib.name
                    ));
                }
            }
        }

        // Apply overrides
        if !overrides.is_empty() {
            let orig_attr_lib_names = Vec::from_iter(self.libs.iter().map(|lib| lib.name));
            for (name, override_lib) in overrides {
                for (orig_attr_lib_name, attr_lib) in
                    iter::zip(&orig_attr_lib_names, &mut self.libs)
                {
                    if let Some(orig_attr_lib_name) = orig_attr_lib_name
                        && orig_attr_lib_name.as_str() == name {
                        // The name is overridden unconditionally
                        attr_lib.name =
                            Some(Symbol::intern(&override_lib.new_name.as_ref().unwrap()));
                        // The kind and modifiers are overridden only if the override specifies
                        // them explicitly
                        if override_lib.kind != NativeLibKind::Unspecified {
                            if attr_lib.has_modifiers() && !override_lib.has_modifiers() {
                                // Not clear what behavior is desirable here
                                self.tcx.sess.err(&format!(
                                    "override for library `{name}` must specify modifiers because \
                                     the overridden `#[link]` attribute specified modifiers",
                                ));
                            }
                            attr_lib.kind = override_lib.kind;
                            attr_lib.verbatim = override_lib.verbatim;
                        }
                    }
                }
            }
        }

        // Add regular (non-override) libraries from the command line
        for cmd_lib in &self.tcx.sess.opts.libs {
            if cmd_lib.new_name.is_none() {
                self.libs.push(NativeLib {
                    name: Some(Symbol::intern(&cmd_lib.name)),
                    kind: cmd_lib.kind,
                    cfg: None,
                    foreign_module: None,
                    wasm_import_module: None,
                    verbatim: cmd_lib.verbatim,
                    dll_imports: Vec::new(),
                });
            }
        }
    }

    fn unify_kinds_and_modifiers(&mut self) {
        let mut kinds_and_modifiers =
            FxHashMap::<Symbol, FxHashSet<(NativeLibKind, Option<bool>)>>::default();
        for NativeLib { name, kind, verbatim, cfg, .. } in &self.libs {
            if let Some(name) = *name && *kind != NativeLibKind::Unspecified && cfg.is_none() {
                kinds_and_modifiers.entry(name).or_default().insert((*kind, *verbatim));
            }
        }

        for NativeLib { name, kind, verbatim, .. } in &mut self.libs {
            if let Some(name) = name
                && *kind == NativeLibKind::Unspecified
                && let Some(kinds_and_modifiers) = kinds_and_modifiers.get(name) {
                    if kinds_and_modifiers.len() == 1 {
                        (*kind, *verbatim) = *kinds_and_modifiers.iter().next().unwrap();
                    } else {
                        self.tcx.sess.err(&format!(
                            "cannot infer kind for library `{name}`, it is linked more than once \
                             with different kinds or modifiers",
                        ));
                    }
            }
        }
    }

    fn compat_reorder(&mut self) {
        let mut tmp = Vec::with_capacity(self.libs.len());

        let mut cmd_libs = Vec::from_iter(self.libs.drain(self.attr_libs..));
        cmd_libs.reverse();
        let mut attr_libs = mem::take(&mut self.libs);
        attr_libs.reverse();

        while !cmd_libs.is_empty() {
            let cmd_lib = cmd_libs.remove(0);
            let name = cmd_lib.name;
            tmp.push(cmd_lib);
            tmp.extend(cmd_libs.drain_filter(|cmd_lib| cmd_lib.name == name));
            tmp.extend(attr_libs.drain_filter(|attr_lib| attr_lib.name == name));
        }

        tmp.append(&mut attr_libs);
        tmp.reverse();

        self.libs = tmp;
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

    fn build_dll_import(&self, abi: Abi, item: &hir::ForeignItemRef) -> DllImport {
        let calling_convention = if self.tcx.sess.target.arch == "x86" {
            match abi {
                Abi::C { .. } | Abi::Cdecl { .. } => DllCallingConvention::C,
                Abi::Stdcall { .. } | Abi::System { .. } => {
                    DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                }
                Abi::Fastcall { .. } => {
                    DllCallingConvention::Fastcall(self.i686_arg_list_size(item))
                }
                // Vectorcall is intentionally not supported at this time.
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

        DllImport {
            name: item.ident.name,
            ordinal: self.tcx.codegen_fn_attrs(item.id.def_id).link_ordinal,
            calling_convention,
            span: item.span,
        }
    }
}
