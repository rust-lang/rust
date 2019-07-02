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
use syntax::symbol::{Symbol, sym};
use syntax::{span_err, struct_span_err};

pub fn collect(tcx: TyCtxt<'_>) -> Vec<NativeLibrary> {
    let mut collector = Collector {
        tcx,
        libs: Vec::new(),
    };
    tcx.hir().krate().visit_all_item_likes(&mut collector);
    collector.process_command_line();
    return collector.libs;
}

pub fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLibrary>,
}

impl ItemLikeVisitor<'tcx> for Collector<'tcx> {
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
        for m in it.attrs.iter().filter(|a| a.check_name(sym::link)) {
            let items = match m.meta_item_list() {
                Some(item) => item,
                None => continue,
            };
            let mut lib = NativeLibrary {
                name: None,
                kind: cstore::NativeUnknown,
                cfg: None,
                foreign_module: Some(self.tcx.hir().local_def_id_from_hir_id(it.hir_id)),
                wasm_import_module: None,
            };
            let mut kind_specified = false;

            for item in items.iter() {
                if item.check_name(sym::kind) {
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
                            struct_span_err!(self.tcx.sess, item.span(), E0458,
                                      "unknown kind: `{}`", k)
                                .span_label(item.span(), "unknown kind")
                                .span_label(m.span, "").emit();
                            cstore::NativeUnknown
                        }
                    };
                } else if item.check_name(sym::name) {
                    lib.name = item.value_str();
                } else if item.check_name(sym::cfg) {
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
                } else if item.check_name(sym::wasm_import_module) {
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

impl Collector<'tcx> {
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
                                           sym::link_cfg,
                                           span.unwrap(),
                                           GateIssue::Language,
                                           "is feature gated");
        }
        if lib.kind == cstore::NativeStaticNobundle &&
           !self.tcx.features().static_nobundle {
            feature_gate::emit_feature_err(&self.tcx.sess.parse_sess,
                                           sym::static_nobundle,
                                           span.unwrap_or_else(|| syntax_pos::DUMMY_SP),
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
                    .any(|n| n.as_str() == *name);
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
        // (there may be more than one) with the specified name.  If any
        // library is mentioned more than once, keep the latest mention
        // of it, so that any possible dependent libraries appear before
        // it.  (This ensures that the linker is able to see symbols from
        // all possible dependent libraries before linking in the library
        // in question.)
        for &(ref name, ref new_name, kind) in &self.tcx.sess.opts.libs {
            // If we've already added any native libraries with the same
            // name, they will be pulled out into `existing`, so that we
            // can move them to the end of the list below.
            let mut existing = self.libs.drain_filter(|lib| {
                if let Some(lib_name) = lib.name {
                    if lib_name.as_str() == *name {
                        if let Some(k) = kind {
                            lib.kind = k;
                        }
                        if let &Some(ref new_name) = new_name {
                            lib.name = Some(Symbol::intern(new_name));
                        }
                        return true;
                    }
                }
                false
            }).collect::<Vec<_>>();
            if existing.is_empty() {
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
            } else {
                // Move all existing libraries with the same name to the
                // end of the command line.
                self.libs.append(&mut existing);
            }
        }
    }
}
