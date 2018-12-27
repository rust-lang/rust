use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::middle::cstore::{self, NativeLibrary};
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc::util::nodemap::FxHashSet;
use rustc_target::spec::abi::Abi;
use syntax::attr;
use syntax::source_map::Span;
use syntax::feature_gate::{self, GateIssue};
use syntax::symbol::Symbol;
use errors::{Applicability, DiagnosticId};

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<NativeLibrary> {
    let mut collector = Collector {
        tcx,
        libs: Vec::new(),
    };
    tcx.hir().krate().visit_all_item_likes(&mut collector);
    collector.process_command_line();
    return collector.libs
}

pub fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

struct Collector<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    libs: Vec<NativeLibrary>,
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for Collector<'a, 'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item) {
        let fm = match it.node {
            hir::ItemKind::ForeignMod(ref fm) => fm,
            _ => return,
        };

        if fm.abi == Abi::Rust ||
            fm.abi == Abi::RustIntrinsic ||
            fm.abi == Abi::PlatformIntrinsic {
            return
        }

        // Process all of the #[link(..)]-style arguments
        for m in it.attrs.iter().filter(|a| a.check_name("link")) {
            let items = match m.meta_item_list() {
                Some(item) => item,
                None => {
                    let mut err = self.tcx.sess.struct_span_err_with_code(
                        m.span,
                        "#[link(...)] specified without arguments",
                        DiagnosticId::Error("E0459".into()),
                    );
                    if let Some(value) = m.value_str() {
                        err.span_suggestion_with_applicability(
                            m.span,
                            "specify a `name` argument instead",
                            format!("#[link(name = \"{}\")]", value),
                            Applicability::MachineApplicable,
                        );
                    }
                    err.emit();
                    continue;
                }
            };
            let mut lib = NativeLibrary {
                name: None,
                kind: cstore::NativeUnknown,
                cfg: None,
                foreign_module: Some(self.tcx.hir().local_def_id(it.id)),
                wasm_import_module: None,
            };
            let mut kind_specified = false;

            for item in items.iter() {
                let handle_duplicate_arg = |name, report| {
                    if report {
                        struct_span_err!(self.tcx.sess, m.span, E0494,
                            "#[link(...)] contains repeated `{}` arguments", name)
                        .span_label(item.span, format!("repeated `{}` argument", name))
                        .emit();
                    }
                };

                if item.check_name("kind") {
                    handle_duplicate_arg("kind", kind_specified);
                    kind_specified = true;
                    let kind = match item.value_str() {
                        Some(name) => name,
                        None => continue, // skip like historical compilers
                    };
                    lib.kind = match &kind.as_str()[..] {
                        "static" => cstore::NativeStatic,
                        "static-nobundle" => cstore::NativeStaticNobundle,
                        "dylib" => cstore::NativeUnknown,
                        "framework" => cstore::NativeFramework,
                        k => {
                            struct_span_err!(self.tcx.sess, m.span, E0458,
                                      "unknown kind: `{}`", k)
                                .span_label(item.span, "unknown kind").emit();
                            cstore::NativeUnknown
                        }
                    };
                } else if item.check_name("name") {
                    handle_duplicate_arg("name", lib.name.is_some());
                    lib.name = item.value_str();
                } else if item.check_name("cfg") {
                    handle_duplicate_arg("cfg", lib.cfg.is_some());
                    let cfg = match item.meta_item_list() {
                        Some(list) => list,
                        None => continue, // skip like historical compilers
                    };
                    if cfg.is_empty() {
                        self.tcx.sess.span_err(
                            item.span(),
                            "`cfg()` must have an argument",
                        );
                    } else if let cfg @ Some(..) = cfg[0].meta_item() {
                        lib.cfg = cfg.cloned();
                    } else {
                        self.tcx.sess.span_err(cfg[0].span(), "invalid argument for `cfg(..)`");
                    }
                } else if item.check_name("wasm_import_module") {
                    handle_duplicate_arg("wasm_import_module", lib.wasm_import_module.is_some());
                    match item.value_str() {
                        Some(s) => lib.wasm_import_module = Some(s),
                        None => {
                            let msg = "must be of the form #[link(wasm_import_module = \"...\")]";
                            self.tcx.sess.span_err(item.span(), msg);
                        }
                    }
                } else {
                    // currently, like past compilers, ignore unknown
                    // directives here.
                }
            }

            // In general we require #[link(name = "...")] but we allow
            // #[link(wasm_import_module = "...")] without the `name`.
            let requires_name = kind_specified || lib.wasm_import_module.is_none();
            if lib.name.is_none() && requires_name {
                struct_span_err!(self.tcx.sess, m.span, E0459,
                                 "#[link(...)] specified without \
                                  `name = \"foo\"`")
                    .span_label(m.span, "missing `name` argument")
                    .emit();
            }
            self.register_native_lib(Some(m.span), lib);
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem) {}
}

impl<'a, 'tcx> Collector<'a, 'tcx> {
    fn register_native_lib(&mut self, span: Option<Span>, lib: NativeLibrary) {
        if lib.name.as_ref().map(|s| s.as_str().is_empty()).unwrap_or(false) {
            match span {
                Some(span) => {
                    struct_span_err!(self.tcx.sess, span, E0454,
                                     "#[link(name = \"\")] given with empty name")
                        .span_label(span, "empty name given")
                        .emit();
                }
                None => {
                    self.tcx.sess.err("empty library name given via `-l`");
                }
            }
            return
        }
        let is_osx = self.tcx.sess.target.target.options.is_like_osx;
        if lib.kind == cstore::NativeFramework && !is_osx {
            let msg = "native frameworks are only available on macOS targets";
            match span {
                Some(span) => span_err!(self.tcx.sess, span, E0455, "{}", msg),
                None => self.tcx.sess.err(msg),
            }
        }
        if lib.cfg.is_some() && !self.tcx.features().link_cfg {
            feature_gate::emit_feature_err(&self.tcx.sess.parse_sess,
                                           "link_cfg",
                                           span.unwrap(),
                                           GateIssue::Language,
                                           "is feature gated");
        }
        if lib.kind == cstore::NativeStaticNobundle &&
           !self.tcx.features().static_nobundle {
            feature_gate::emit_feature_err(&self.tcx.sess.parse_sess,
                                           "static_nobundle",
                                           span.unwrap(),
                                           GateIssue::Language,
                                           "kind=\"static-nobundle\" is feature gated");
        }
        self.libs.push(lib);
    }

    // Process libs passed on the command line
    fn process_command_line(&mut self) {
        // First, check for errors
        let mut renames = FxHashSet::default();
        for &(ref name, ref new_name, _) in &self.tcx.sess.opts.libs {
            if let &Some(ref new_name) = new_name {
                let any_duplicate = self.libs
                    .iter()
                    .filter_map(|lib| lib.name.as_ref())
                    .any(|n| n == name);
                if new_name.is_empty() {
                    self.tcx.sess.err(
                        &format!("an empty renaming target was specified for library `{}`",name));
                } else if !any_duplicate {
                    self.tcx.sess.err(&format!("renaming of the library `{}` was specified, \
                                                however this crate contains no #[link(...)] \
                                                attributes referencing this library.", name));
                } else if renames.contains(name) {
                    self.tcx.sess.err(&format!("multiple renamings were \
                                                specified for library `{}` .",
                                               name));
                } else {
                    renames.insert(name);
                }
            }
        }

        // Update kind and, optionally, the name of all native libraries
        // (there may be more than one) with the specified name.
        for &(ref name, ref new_name, kind) in &self.tcx.sess.opts.libs {
            let mut found = false;
            for lib in self.libs.iter_mut() {
                let lib_name = match lib.name {
                    Some(n) => n,
                    None => continue,
                };
                if lib_name == name as &str {
                    let mut changed = false;
                    if let Some(k) = kind {
                        lib.kind = k;
                        changed = true;
                    }
                    if let &Some(ref new_name) = new_name {
                        lib.name = Some(Symbol::intern(new_name));
                        changed = true;
                    }
                    if !changed {
                        let msg = format!("redundant linker flag specified for \
                                           library `{}`", name);
                        self.tcx.sess.warn(&msg);
                    }

                    found = true;
                }
            }
            if !found {
                // Add if not found
                let new_name = new_name.as_ref().map(|s| &**s); // &Option<String> -> Option<&str>
                let lib = NativeLibrary {
                    name: Some(Symbol::intern(new_name.unwrap_or(name))),
                    kind: if let Some(k) = kind { k } else { cstore::NativeUnknown },
                    cfg: None,
                    foreign_module: None,
                    wasm_import_module: None,
                };
                self.register_native_lib(None, lib);
            }
        }
    }
}
