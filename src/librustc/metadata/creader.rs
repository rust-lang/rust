// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

//! Validates all used crates and extern libraries and loads their metadata

use back::link;
use back::svh::Svh;
use driver::session::Session;
use driver::{driver, config};
use metadata::cstore;
use metadata::cstore::{CStore, CrateSource};
use metadata::decoder;
use metadata::loader;
use metadata::loader::CratePaths;

use std::rc::Rc;
use collections::HashMap;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{Span};
use syntax::diagnostic::SpanHandler;
use syntax::ext::base::{CrateLoader, MacroCrate};
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::crateid::CrateId;
use syntax::visit;

struct Env<'a> {
    sess: &'a Session,
    next_crate_num: ast::CrateNum,
}

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(sess: &Session,
                   krate: &ast::Crate) {
    let mut e = Env {
        sess: sess,
        next_crate_num: sess.cstore.next_crate_num(),
    };
    visit_crate(&e, krate);
    visit::walk_crate(&mut e, krate, ());
    dump_crates(&sess.cstore);
    warn_if_multiple_versions(sess.diagnostic(), &sess.cstore)
}

impl<'a> visit::Visitor<()> for Env<'a> {
    fn visit_view_item(&mut self, a: &ast::ViewItem, _: ()) {
        visit_view_item(self, a);
        visit::walk_view_item(self, a, ());
    }
    fn visit_item(&mut self, a: &ast::Item, _: ()) {
        visit_item(self, a);
        visit::walk_item(self, a, ());
    }
}

fn dump_crates(cstore: &CStore) {
    debug!("resolved crates:");
    cstore.iter_crate_data_origins(|_, data, opt_source| {
        debug!("crate_id: {}", data.crate_id());
        debug!("  cnum: {}", data.cnum);
        debug!("  hash: {}", data.hash());
        opt_source.map(|cs| {
            let CrateSource { dylib, rlib, cnum: _ } = cs;
            dylib.map(|dl| debug!("  dylib: {}", dl.display()));
            rlib.map(|rl|  debug!("   rlib: {}", rl.display()));
        });
    })
}

fn warn_if_multiple_versions(diag: &SpanHandler, cstore: &CStore) {
    let mut map = HashMap::new();
    cstore.iter_crate_data(|cnum, data| {
        let crateid = data.crate_id();
        let key = (crateid.name.clone(), crateid.path.clone());
        map.find_or_insert_with(key, |_| Vec::new()).push(cnum);
    });

    for ((name, _), dupes) in map.move_iter() {
        if dupes.len() == 1 { continue }
        diag.handler().warn(
            format!("using multiple versions of crate `{}`",
                    name).as_slice());
        for dupe in dupes.move_iter() {
            let data = cstore.get_crate_data(dupe);
            diag.span_note(data.span, "used here");
            loader::note_crateid_attr(diag, &data.crate_id());
        }
    }
}

fn visit_crate(e: &Env, c: &ast::Crate) {
    for a in c.attrs.iter().filter(|m| m.name().equiv(&("link_args"))) {
        match a.value_str() {
            Some(ref linkarg) => e.sess.cstore.add_used_link_args(linkarg.get()),
            None => { /* fallthrough */ }
        }
    }
}

fn should_link(i: &ast::ViewItem) -> bool {
    i.attrs.iter().all(|attr| {
        attr.name().get() != "phase" ||
            attr.meta_item_list().map_or(false, |phases| {
                attr::contains_name(phases.as_slice(), "link")
            })
    })
}

fn visit_view_item(e: &mut Env, i: &ast::ViewItem) {
    if !should_link(i) {
        return;
    }

    match extract_crate_info(e, i) {
        Some(info) => {
            let (cnum, _, _) = resolve_crate(e,
                                             &None,
                                             info.ident.as_slice(),
                                             &info.crate_id,
                                             None,
                                             i.span);
            e.sess.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
        }
        None => ()
    }
}

struct CrateInfo {
    ident: StrBuf,
    crate_id: CrateId,
    id: ast::NodeId,
    should_link: bool,
}

fn extract_crate_info(e: &Env, i: &ast::ViewItem) -> Option<CrateInfo> {
    match i.node {
        ast::ViewItemExternCrate(ident, ref path_opt, id) => {
            let ident = token::get_ident(ident);
            debug!("resolving extern crate stmt. ident: {:?} path_opt: {:?}",
                   ident, path_opt);
            let crate_id = match *path_opt {
                Some((ref path_str, _)) => {
                    let crateid: Option<CrateId> = from_str(path_str.get());
                    match crateid {
                        None => {
                            e.sess.span_err(i.span, "malformed crate id");
                            return None
                        }
                        Some(id) => id
                    }
                }
                None => from_str(ident.get().to_str().as_slice()).unwrap()
            };
            Some(CrateInfo {
                ident: ident.get().to_strbuf(),
                crate_id: crate_id,
                id: id,
                should_link: should_link(i),
            })
        }
        _ => None
    }
}

fn visit_item(e: &Env, i: &ast::Item) {
    match i.node {
        ast::ItemForeignMod(ref fm) => {
            if fm.abi == abi::Rust || fm.abi == abi::RustIntrinsic {
                return;
            }

            // First, add all of the custom link_args attributes
            let link_args = i.attrs.iter()
                .filter_map(|at| if at.name().equiv(&("link_args")) {
                    Some(at)
                } else {
                    None
                })
                .collect::<Vec<&ast::Attribute>>();
            for m in link_args.iter() {
                match m.value_str() {
                    Some(linkarg) => e.sess.cstore.add_used_link_args(linkarg.get()),
                    None => { /* fallthrough */ }
                }
            }

            // Next, process all of the #[link(..)]-style arguments
            let link_args = i.attrs.iter()
                .filter_map(|at| if at.name().equiv(&("link")) {
                    Some(at)
                } else {
                    None
                })
                .collect::<Vec<&ast::Attribute>>();
            for m in link_args.iter() {
                match m.meta_item_list() {
                    Some(items) => {
                        let kind = items.iter().find(|k| {
                            k.name().equiv(&("kind"))
                        }).and_then(|a| a.value_str());
                        let kind = match kind {
                            Some(k) => {
                                if k.equiv(&("static")) {
                                    cstore::NativeStatic
                                } else if e.sess.targ_cfg.os == abi::OsMacos &&
                                          k.equiv(&("framework")) {
                                    cstore::NativeFramework
                                } else if k.equiv(&("framework")) {
                                    e.sess.span_err(m.span,
                                        "native frameworks are only available \
                                         on OSX targets");
                                    cstore::NativeUnknown
                                } else {
                                    e.sess.span_err(m.span,
                                        format!("unknown kind: `{}`",
                                                k).as_slice());
                                    cstore::NativeUnknown
                                }
                            }
                            None => cstore::NativeUnknown
                        };
                        let n = items.iter().find(|n| {
                            n.name().equiv(&("name"))
                        }).and_then(|a| a.value_str());
                        let n = match n {
                            Some(n) => n,
                            None => {
                                e.sess.span_err(m.span,
                                    "#[link(...)] specified without \
                                     `name = \"foo\"`");
                                InternedString::new("foo")
                            }
                        };
                        if n.get().is_empty() {
                            e.sess.span_err(m.span,
                                            "#[link(name = \"\")] given with \
                                             empty name");
                        } else {
                            e.sess
                             .cstore
                             .add_used_library(n.get().to_strbuf(), kind);
                        }
                    }
                    None => {}
                }
            }
        }
        _ => { }
    }
}

fn existing_match(e: &Env, crate_id: &CrateId,
                  hash: Option<&Svh>) -> Option<ast::CrateNum> {
    let mut ret = None;
    e.sess.cstore.iter_crate_data(|cnum, data| {
        let other_id = data.crate_id();
        if crate_id.matches(&other_id) {
            let other_hash = data.hash();
            match hash {
                Some(hash) if *hash != other_hash => {}
                Some(..) | None => { ret = Some(cnum); }
            }
        }
    });
    return ret;
}

fn register_crate<'a>(e: &mut Env,
                  root: &Option<CratePaths>,
                  ident: &str,
                  crate_id: &CrateId,
                  span: Span,
                  lib: loader::Library)
                        -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                            cstore::CrateSource) {
    // Claim this crate number and cache it
    let cnum = e.next_crate_num;
    e.next_crate_num += 1;

    // Stash paths for top-most crate locally if necessary.
    let crate_paths = if root.is_none() {
        Some(CratePaths {
            ident: ident.to_strbuf(),
            dylib: lib.dylib.clone(),
            rlib:  lib.rlib.clone(),
        })
    } else {
        None
    };
    // Maintain a reference to the top most crate.
    let root = if root.is_some() { root } else { &crate_paths };

    let cnum_map = resolve_crate_deps(e, root, lib.metadata.as_slice(), span);

    let loader::Library{ dylib, rlib, metadata } = lib;

    let cmeta = Rc::new( cstore::crate_metadata {
        name: crate_id.name.to_strbuf(),
        data: metadata,
        cnum_map: cnum_map,
        cnum: cnum,
        span: span,
    });

    let source = cstore::CrateSource {
        dylib: dylib,
        rlib: rlib,
        cnum: cnum,
    };

    e.sess.cstore.set_crate_data(cnum, cmeta.clone());
    e.sess.cstore.add_used_crate_source(source.clone());
    (cnum, cmeta, source)
}

fn resolve_crate<'a>(e: &mut Env,
                 root: &Option<CratePaths>,
                 ident: &str,
                 crate_id: &CrateId,
                 hash: Option<&Svh>,
                 span: Span)
                     -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                         cstore::CrateSource) {
    match existing_match(e, crate_id, hash) {
        None => {
            let id_hash = link::crate_id_hash(crate_id);
            let mut load_ctxt = loader::Context {
                sess: e.sess,
                span: span,
                ident: ident,
                crate_id: crate_id,
                id_hash: id_hash.as_slice(),
                hash: hash.map(|a| &*a),
                filesearch: e.sess.target_filesearch(),
                os: config::cfg_os_to_meta_os(e.sess.targ_cfg.os),
                triple: e.sess.targ_cfg.target_strs.target_triple.as_slice(),
                root: root,
                rejected_via_hash: vec!(),
                rejected_via_triple: vec!(),
            };
            let library = load_ctxt.load_library_crate();
            register_crate(e, root, ident, crate_id, span, library)
        }
        Some(cnum) => (cnum,
                       e.sess.cstore.get_crate_data(cnum),
                       e.sess.cstore.get_used_crate_source(cnum).unwrap())
    }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: &mut Env,
                      root: &Option<CratePaths>,
                      cdata: &[u8], span : Span)
                   -> cstore::cnum_map {
    debug!("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    decoder::get_crate_deps(cdata).iter().map(|dep| {
        debug!("resolving dep crate {} hash: `{}`", dep.crate_id, dep.hash);
        let (local_cnum, _, _) = resolve_crate(e, root,
                                               dep.crate_id.name.as_slice(),
                                               &dep.crate_id,
                                               Some(&dep.hash),
                                               span);
        (dep.cnum, local_cnum)
    }).collect()
}

pub struct Loader<'a> {
    env: Env<'a>,
}

impl<'a> Loader<'a> {
    pub fn new(sess: &'a Session) -> Loader<'a> {
        Loader {
            env: Env {
                sess: sess,
                next_crate_num: sess.cstore.next_crate_num(),
            }
        }
    }
}

impl<'a> CrateLoader for Loader<'a> {
    fn load_crate(&mut self, krate: &ast::ViewItem) -> MacroCrate {
        let info = extract_crate_info(&self.env, krate).unwrap();
        let target_triple = self.env.sess.targ_cfg.target_strs.target_triple.as_slice();
        let is_cross = target_triple != driver::host_triple();
        let mut should_link = info.should_link && !is_cross;
        let id_hash = link::crate_id_hash(&info.crate_id);
        let os = config::get_os(driver::host_triple()).unwrap();
        let mut load_ctxt = loader::Context {
            sess: self.env.sess,
            span: krate.span,
            ident: info.ident.as_slice(),
            crate_id: &info.crate_id,
            id_hash: id_hash.as_slice(),
            hash: None,
            filesearch: self.env.sess.host_filesearch(),
            triple: driver::host_triple(),
            os: config::cfg_os_to_meta_os(os),
            root: &None,
            rejected_via_hash: vec!(),
            rejected_via_triple: vec!(),
        };
        let library = match load_ctxt.maybe_load_library_crate() {
            Some (l) => l,
            None if is_cross => {
                // try loading from target crates (only valid if there are
                // no syntax extensions)
                load_ctxt.triple = target_triple;
                load_ctxt.os = config::cfg_os_to_meta_os(self.env.sess.targ_cfg.os);
                load_ctxt.filesearch = self.env.sess.target_filesearch();
                let lib = load_ctxt.load_library_crate();
                if decoder::get_macro_registrar_fn(lib.metadata.as_slice()).is_some() {
                    let message = format!("crate `{}` contains a macro_registrar fn but \
                                  only a version for triple `{}` could be found (need {})",
                                  info.ident, target_triple, driver::host_triple());
                    self.env.sess.span_err(krate.span, message.as_slice());
                    // need to abort now because the syntax expansion
                    // code will shortly attempt to load and execute
                    // code from the found library.
                    self.env.sess.abort_if_errors();
                }
                should_link = info.should_link;
                lib
            }
            None => { load_ctxt.report_load_errs(); unreachable!() },
        };
        let macros = decoder::get_exported_macros(library.metadata.as_slice());
        let registrar = decoder::get_macro_registrar_fn(library.metadata.as_slice()).map(|id| {
            decoder::get_symbol(library.metadata.as_slice(), id).to_strbuf()
        });
        let mc = MacroCrate {
            lib: library.dylib.clone(),
            macros: macros.move_iter().map(|x| x.to_strbuf()).collect(),
            registrar_symbol: registrar,
        };
        if should_link {
            // register crate now to avoid double-reading metadata
            register_crate(&mut self.env, &None, info.ident.as_slice(),
                           &info.crate_id, krate.span, library);
        }
        mc
    }
}
