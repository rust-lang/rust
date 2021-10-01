use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::middle::cstore::{DllCallingConvention, DllImport, NativeLib};
use rustc_middle::ty::{List, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::parse::feature_err;
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<NativeLib> {
    let mut collector = Collector { tcx, libs: Vec::new() };
    tcx.hir().visit_all_item_likes(&mut collector);
    collector.process_command_line();
    collector.libs
}

crate fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLib>,
}

impl ItemLikeVisitor<'tcx> for Collector<'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let (abi, foreign_mod_items) = match it.kind {
            hir::ItemKind::ForeignMod { abi, items } => (abi, items),
            _ => return,
        };

        if abi == Abi::Rust || abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            return;
        }

        // Process all of the #[link(..)]-style arguments
        let sess = &self.tcx.sess;
        for m in self.tcx.hir().attrs(it.hir_id()).iter().filter(|a| a.has_name(sym::link)) {
            let items = match m.meta_item_list() {
                Some(item) => item,
                None => continue,
            };
            let mut lib = NativeLib {
                name: None,
                kind: NativeLibKind::Unspecified,
                cfg: None,
                foreign_module: Some(it.def_id.to_def_id()),
                wasm_import_module: None,
                verbatim: None,
                dll_imports: Vec::new(),
            };
            let mut kind_specified = false;

            for item in items.iter() {
                if item.has_name(sym::kind) {
                    kind_specified = true;
                    let kind = match item.value_str() {
                        Some(name) => name,
                        None => continue, // skip like historical compilers
                    };
                    lib.kind = match &*kind.as_str() {
                        "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
                        "static-nobundle" => {
                            sess.struct_span_warn(
                                item.span(),
                                "library kind `static-nobundle` has been superseded by specifying \
                                modifier `-bundle` with library kind `static`",
                            )
                            .emit();
                            if !self.tcx.features().static_nobundle {
                                feature_err(
                                    &self.tcx.sess.parse_sess,
                                    sym::static_nobundle,
                                    item.span(),
                                    "kind=\"static-nobundle\" is unstable",
                                )
                                .emit();
                            }
                            NativeLibKind::Static { bundle: Some(false), whole_archive: None }
                        }
                        "dylib" => NativeLibKind::Dylib { as_needed: None },
                        "framework" => NativeLibKind::Framework { as_needed: None },
                        "raw-dylib" => NativeLibKind::RawDylib,
                        k => {
                            struct_span_err!(sess, item.span(), E0458, "unknown kind: `{}`", k)
                                .span_label(item.span(), "unknown kind")
                                .span_label(m.span, "")
                                .emit();
                            NativeLibKind::Unspecified
                        }
                    };
                } else if item.has_name(sym::name) {
                    lib.name = item.value_str();
                } else if item.has_name(sym::cfg) {
                    let cfg = match item.meta_item_list() {
                        Some(list) => list,
                        None => continue, // skip like historical compilers
                    };
                    if cfg.is_empty() {
                        sess.span_err(item.span(), "`cfg()` must have an argument");
                    } else if let cfg @ Some(..) = cfg[0].meta_item() {
                        lib.cfg = cfg.cloned();
                    } else {
                        sess.span_err(cfg[0].span(), "invalid argument for `cfg(..)`");
                    }
                } else if item.has_name(sym::wasm_import_module) {
                    match item.value_str() {
                        Some(s) => lib.wasm_import_module = Some(s),
                        None => {
                            let msg = "must be of the form `#[link(wasm_import_module = \"...\")]`";
                            sess.span_err(item.span(), msg);
                        }
                    }
                } else {
                    // currently, like past compilers, ignore unknown
                    // directives here.
                }
            }

            // Do this outside the above loop so we don't depend on modifiers coming
            // after kinds
            if let Some(item) = items.iter().find(|item| item.has_name(sym::modifiers)) {
                if let Some(modifiers) = item.value_str() {
                    let span = item.name_value_literal_span().unwrap();
                    for modifier in modifiers.as_str().split(',') {
                        let (modifier, value) = match modifier.strip_prefix(&['+', '-'][..]) {
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

                        match (modifier, &mut lib.kind) {
                            ("bundle", NativeLibKind::Static { bundle, .. }) => {
                                *bundle = Some(value);
                            }
                            ("bundle", _) => sess.span_err(
                                span,
                                "bundle linking modifier is only compatible with \
                                `static` linking kind",
                            ),

                            ("verbatim", _) => lib.verbatim = Some(value),

                            ("whole-archive", NativeLibKind::Static { whole_archive, .. }) => {
                                *whole_archive = Some(value);
                            }
                            ("whole-archive", _) => sess.span_err(
                                span,
                                "whole-archive linking modifier is only compatible with \
                                `static` linking kind",
                            ),

                            ("as-needed", NativeLibKind::Dylib { as_needed })
                            | ("as-needed", NativeLibKind::Framework { as_needed }) => {
                                *as_needed = Some(value);
                            }
                            ("as-needed", _) => sess.span_err(
                                span,
                                "as-needed linking modifier is only compatible with \
                                `dylib` and `framework` linking kinds",
                            ),

                            _ => sess.span_err(
                                span,
                                &format!(
                                    "unrecognized linking modifier `{}`, expected one \
                                    of: bundle, verbatim, whole-archive, as-needed",
                                    modifier
                                ),
                            ),
                        }
                    }
                } else {
                    let msg = "must be of the form `#[link(modifiers = \"...\")]`";
                    sess.span_err(item.span(), msg);
                }
            }

            // In general we require #[link(name = "...")] but we allow
            // #[link(wasm_import_module = "...")] without the `name`.
            let requires_name = kind_specified || lib.wasm_import_module.is_none();
            if lib.name.is_none() && requires_name {
                struct_span_err!(
                    sess,
                    m.span,
                    E0459,
                    "`#[link(...)]` specified without \
                                  `name = \"foo\"`"
                )
                .span_label(m.span, "missing `name` argument")
                .emit();
            }

            if lib.kind == NativeLibKind::RawDylib {
                lib.dll_imports.extend(
                    foreign_mod_items
                        .iter()
                        .map(|child_item| self.build_dll_import(abi, child_item)),
                );
            }

            self.register_native_lib(Some(m.span), lib);
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _it: &'tcx hir::ForeignItem<'tcx>) {}
}

impl Collector<'tcx> {
    fn register_native_lib(&mut self, span: Option<Span>, lib: NativeLib) {
        if lib.name.as_ref().map_or(false, |&s| s == kw::Empty) {
            match span {
                Some(span) => {
                    struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0454,
                        "`#[link(name = \"\")]` given with empty name"
                    )
                    .span_label(span, "empty name given")
                    .emit();
                }
                None => {
                    self.tcx.sess.err("empty library name given via `-l`");
                }
            }
            return;
        }
        let is_osx = self.tcx.sess.target.is_like_osx;
        if matches!(lib.kind, NativeLibKind::Framework { .. }) && !is_osx {
            let msg = "native frameworks are only available on macOS targets";
            match span {
                Some(span) => struct_span_err!(self.tcx.sess, span, E0455, "{}", msg).emit(),
                None => self.tcx.sess.err(msg),
            }
        }
        if lib.cfg.is_some() && !self.tcx.features().link_cfg {
            feature_err(
                &self.tcx.sess.parse_sess,
                sym::link_cfg,
                span.unwrap(),
                "kind=\"link_cfg\" is unstable",
            )
            .emit();
        }
        // this just unwraps lib.name; we already established that it isn't empty above.
        if let (NativeLibKind::RawDylib, Some(lib_name)) = (lib.kind, lib.name) {
            let span = match span {
                Some(s) => s,
                None => {
                    bug!("raw-dylib libraries are not supported on the command line");
                }
            };

            if !self.tcx.sess.target.options.is_like_windows {
                self.tcx.sess.span_fatal(
                    span,
                    "`#[link(...)]` with `kind = \"raw-dylib\"` only supported on Windows",
                );
            } else if !self.tcx.sess.target.options.is_like_msvc {
                self.tcx.sess.span_warn(
                    span,
                    "`#[link(...)]` with `kind = \"raw-dylib\"` not supported on windows-gnu",
                );
            }

            if lib_name.as_str().contains('\0') {
                self.tcx.sess.span_err(span, "library name may not contain NUL characters");
            }

            if !self.tcx.features().raw_dylib {
                feature_err(
                    &self.tcx.sess.parse_sess,
                    sym::raw_dylib,
                    span,
                    "kind=\"raw-dylib\" is unstable",
                )
                .emit();
            }
        }

        self.libs.push(lib);
    }

    // Process libs passed on the command line
    fn process_command_line(&mut self) {
        // First, check for errors
        let mut renames = FxHashSet::default();
        for lib in &self.tcx.sess.opts.libs {
            if let Some(ref new_name) = lib.new_name {
                let any_duplicate = self
                    .libs
                    .iter()
                    .filter_map(|lib| lib.name.as_ref())
                    .any(|n| &n.as_str() == &lib.name);
                if new_name.is_empty() {
                    self.tcx.sess.err(&format!(
                        "an empty renaming target was specified for library `{}`",
                        lib.name
                    ));
                } else if !any_duplicate {
                    self.tcx.sess.err(&format!(
                        "renaming of the library `{}` was specified, \
                                                however this crate contains no `#[link(...)]` \
                                                attributes referencing this library.",
                        lib.name
                    ));
                } else if !renames.insert(&lib.name) {
                    self.tcx.sess.err(&format!(
                        "multiple renamings were \
                                                specified for library `{}` .",
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
                let new_name = passed_lib.new_name.as_ref().map(|s| &**s); // &Option<String> -> Option<&str>
                let lib = NativeLib {
                    name: Some(Symbol::intern(new_name.unwrap_or(&passed_lib.name))),
                    kind: passed_lib.kind,
                    cfg: None,
                    foreign_module: None,
                    wasm_import_module: None,
                    verbatim: passed_lib.verbatim,
                    dll_imports: Vec::new(),
                };
                self.register_native_lib(None, lib);
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
                (layout.size.bytes_usize() + 3) & !3
            })
            .sum()
    }

    fn build_dll_import(&self, abi: Abi, item: &hir::ForeignItemRef) -> DllImport {
        let calling_convention = if self.tcx.sess.target.arch == "x86" {
            match abi {
                Abi::C { .. } | Abi::Cdecl => DllCallingConvention::C,
                Abi::Stdcall { .. } | Abi::System { .. } => {
                    DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                }
                Abi::Fastcall => DllCallingConvention::Fastcall(self.i686_arg_list_size(item)),
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
                Abi::C { .. } | Abi::Win64 | Abi::System { .. } => DllCallingConvention::C,
                _ => {
                    self.tcx.sess.span_fatal(
                        item.span,
                        r#"ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture"#,
                    );
                }
            }
        };
        DllImport { name: item.ident.name, ordinal: None, calling_convention, span: item.span }
    }
}
