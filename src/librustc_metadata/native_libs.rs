// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::middle::cstore::{self, NativeLibrary};
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc::util::nodemap::FxHashSet;
use syntax::abi::Abi;
use syntax::attr;
use syntax::codemap::Span;
use syntax::feature_gate::{self, GateIssue};
use syntax::symbol::Symbol;

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<NativeLibrary> {
    let mut collector = Collector {
        tcx,
        libs: Vec::new(),
    };
    tcx.hir.krate().visit_all_item_likes(&mut collector);
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
            hir::ItemForeignMod(ref fm) => fm,
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
                None => continue,
            };
            let kind = items.iter().find(|k| {
                k.check_name("kind")
            }).and_then(|a| a.value_str()).map(Symbol::as_str);
            let kind = match kind.as_ref().map(|s| &s[..]) {
                Some("static") => cstore::NativeStatic,
                Some("static-nobundle") => cstore::NativeStaticNobundle,
                Some("dylib") => cstore::NativeUnknown,
                Some("framework") => cstore::NativeFramework,
                Some(k) => {
                    struct_span_err!(self.tcx.sess, m.span, E0458,
                              "unknown kind: `{}`", k)
                        .span_label(m.span, "unknown kind").emit();
                    cstore::NativeUnknown
                }
                None => cstore::NativeUnknown
            };
            let n = items.iter().find(|n| {
                n.check_name("name")
            }).and_then(|a| a.value_str());
            let n = match n {
                Some(n) => n,
                None => {
                    struct_span_err!(self.tcx.sess, m.span, E0459,
                                     "#[link(...)] specified without `name = \"foo\"`")
                        .span_label(m.span, "missing `name` argument").emit();
                    Symbol::intern("foo")
                }
            };
            let cfg = items.iter().find(|k| {
                k.check_name("cfg")
            }).and_then(|a| a.meta_item_list());
            let cfg = cfg.map(|list| {
                list[0].meta_item().unwrap().clone()
            });
            let foreign_items = fm.items.iter()
                .map(|it| self.tcx.hir.local_def_id(it.id))
                .collect();
            let lib = NativeLibrary {
                name: n,
                kind,
                cfg,
                foreign_items,
            };
            self.register_native_lib(Some(m.span), lib);
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem) {}
}

impl<'a, 'tcx> Collector<'a, 'tcx> {
    fn register_native_lib(&mut self, span: Option<Span>, lib: NativeLibrary) {
        if lib.name.as_str().is_empty() {
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
        if lib.cfg.is_some() && !self.tcx.sess.features.borrow().link_cfg {
            feature_gate::emit_feature_err(&self.tcx.sess.parse_sess,
                                           "link_cfg",
                                           span.unwrap(),
                                           GateIssue::Language,
                                           "is feature gated");
        }
        if lib.kind == cstore::NativeStaticNobundle &&
           !self.tcx.sess.features.borrow().static_nobundle {
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
        let mut renames = FxHashSet();
        for &(ref name, ref new_name, _) in &self.tcx.sess.opts.libs {
            if let &Some(ref new_name) = new_name {
                if new_name.is_empty() {
                    self.tcx.sess.err(
                        &format!("an empty renaming target was specified for library `{}`",name));
                } else if !self.libs.iter().any(|lib| lib.name == name as &str) {
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

        // Update kind and, optionally, the name of all native libaries
        // (there may be more than one) with the specified name.
        for &(ref name, ref new_name, kind) in &self.tcx.sess.opts.libs {
            let mut found = false;
            for lib in self.libs.iter_mut() {
                if lib.name == name as &str {
                    let mut changed = false;
                    if let Some(k) = kind {
                        lib.kind = k;
                        changed = true;
                    }
                    if let &Some(ref new_name) = new_name {
                        lib.name = Symbol::intern(new_name);
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
                    name: Symbol::intern(new_name.unwrap_or(name)),
                    kind: if let Some(k) = kind { k } else { cstore::NativeUnknown },
                    cfg: None,
                    foreign_items: Vec::new(),
                };
                self.register_native_lib(None, lib);
            }
        }
    }
}
