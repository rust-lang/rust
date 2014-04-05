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
use driver::{driver, session};
use driver::session::Session;
use metadata::csearch;
use metadata::cstore;
use metadata::decoder;
use metadata::loader;
use metadata::loader::Os;
use metadata::loader::CratePaths;

use std::cell::RefCell;
use std::rc::Rc;
use collections::HashMap;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{Span};
use syntax::diagnostic::SpanHandler;
use syntax::ext::base::{CrateLoader, MacroCrate};
use syntax::parse::token::{IdentInterner, InternedString};
use syntax::parse::token;
use syntax::crateid::CrateId;
use syntax::visit;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(sess: &Session,
                   krate: &ast::Crate,
                   os: loader::Os,
                   intr: Rc<IdentInterner>) {
    let mut e = Env {
        sess: sess,
        os: os,
        crate_cache: @RefCell::new(Vec::new()),
        next_crate_num: 1,
        intr: intr
    };
    visit_crate(&e, krate);
    visit::walk_crate(&mut e, krate, ());
    dump_crates(e.crate_cache.borrow().as_slice());
    warn_if_multiple_versions(&mut e,
                              sess.diagnostic(),
                              e.crate_cache.borrow().as_slice());
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

#[deriving(Clone)]
struct cache_entry {
    cnum: ast::CrateNum,
    span: Span,
    hash: Svh,
    crate_id: CrateId,
}

fn dump_crates(crate_cache: &[cache_entry]) {
    debug!("resolved crates:");
    for entry in crate_cache.iter() {
        debug!("cnum: {:?}", entry.cnum);
        debug!("span: {:?}", entry.span);
        debug!("hash: {:?}", entry.hash);
    }
}

fn warn_if_multiple_versions(e: &mut Env,
                             diag: &SpanHandler,
                             crate_cache: &[cache_entry]) {
    if crate_cache.len() != 0u {
        let name = crate_cache[crate_cache.len() - 1].crate_id.name.clone();

        let (matches, non_matches) = crate_cache.partitioned(|entry|
            name == entry.crate_id.name);

        assert!(!matches.is_empty());

        if matches.len() != 1u {
            diag.handler().warn(
                format!("using multiple versions of crate `{}`", name));
            for match_ in matches.iter() {
                diag.span_note(match_.span, "used here");
                loader::note_crateid_attr(diag, &match_.crate_id);
            }
        }

        warn_if_multiple_versions(e, diag, non_matches);
    }
}

struct Env<'a> {
    sess: &'a Session,
    os: loader::Os,
    crate_cache: @RefCell<Vec<cache_entry>>,
    next_crate_num: ast::CrateNum,
    intr: Rc<IdentInterner>
}

fn visit_crate(e: &Env, c: &ast::Crate) {
    for a in c.attrs.iter().filter(|m| m.name().equiv(&("link_args"))) {
        match a.value_str() {
            Some(ref linkarg) => e.sess.cstore.add_used_link_args(linkarg.get()),
            None => { /* fallthrough */ }
        }
    }
}

fn visit_view_item(e: &mut Env, i: &ast::ViewItem) {
    let should_load = i.attrs.iter().all(|attr| {
        attr.name().get() != "phase" ||
            attr.meta_item_list().map_or(false, |phases| {
                attr::contains_name(phases.as_slice(), "link")
            })
    });

    if !should_load {
        return;
    }

    match extract_crate_info(e, i) {
        Some(info) => {
            let cnum = resolve_crate(e, &None, info.ident, &info.crate_id, None,
                                     i.span);
            e.sess.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
        }
        None => ()
    }
}

struct CrateInfo {
    ident: ~str,
    crate_id: CrateId,
    id: ast::NodeId,
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
                None => from_str(ident.get().to_str()).unwrap()
            };
            Some(CrateInfo {
                ident: ident.get().to_str(),
                crate_id: crate_id,
                id: id,
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
                .collect::<~[&ast::Attribute]>();
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
                .collect::<~[&ast::Attribute]>();
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
                                        format!("unknown kind: `{}`", k));
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
                            e.sess.span_err(m.span, "#[link(name = \"\")] given with empty name");
                        } else {
                            e.sess.cstore.add_used_library(n.get().to_owned(), kind);
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
    for c in e.crate_cache.borrow().iter() {
        if !crate_id.matches(&c.crate_id) { continue }
        match hash {
            Some(hash) if *hash != c.hash => {}
            Some(..) | None => return Some(c.cnum)
        }
    }
    None
}

fn resolve_crate<'a>(e: &mut Env,
                     root: &Option<CratePaths>,
                     ident: &str,
                     crate_id: &CrateId,
                     hash: Option<&Svh>,
                     span: Span)
                     -> ast::CrateNum {
    match existing_match(e, crate_id, hash) {
        None => {
            let id_hash = link::crate_id_hash(crate_id);
            let mut load_ctxt = loader::Context {
                sess: e.sess,
                span: span,
                ident: ident,
                crate_id: crate_id,
                id_hash: id_hash,
                hash: hash.map(|a| &*a),
                os: e.os,
                intr: e.intr.clone(),
                rejected_via_hash: vec!(),
            };
            let loader::Library {
                dylib, rlib, metadata
            } = load_ctxt.load_library_crate(root);

            let crate_id = decoder::get_crate_id(metadata.as_slice());
            let hash = decoder::get_crate_hash(metadata.as_slice());

            // Claim this crate number and cache it
            let cnum = e.next_crate_num;
            e.crate_cache.borrow_mut().push(cache_entry {
                cnum: cnum,
                span: span,
                hash: hash,
                crate_id: crate_id,
            });
            e.next_crate_num += 1;

            // Stash paths for top-most crate locally if necessary.
            let crate_paths = if root.is_none() {
                Some(CratePaths {
                    ident: load_ctxt.ident.to_owned(),
                    dylib: dylib.clone(),
                    rlib:  rlib.clone(),
                })
            } else {
                None
            };
            // Maintain a reference to the top most crate.
            let root = if root.is_some() { root } else { &crate_paths };

            // Now resolve the crates referenced by this crate
            let cnum_map = resolve_crate_deps(e,
            root,
            metadata.as_slice(),
            span);

            let cmeta = @cstore::crate_metadata {
                name: load_ctxt.crate_id.name.to_owned(),
                data: metadata,
                cnum_map: cnum_map,
                cnum: cnum
            };

            e.sess.cstore.set_crate_data(cnum, cmeta);
            e.sess.cstore.add_used_crate_source(cstore::CrateSource {
                dylib: dylib,
                rlib: rlib,
                cnum: cnum,
            });
            cnum
        }
        Some(cnum) => cnum
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
    let mut cnum_map = HashMap::new();
    let r = decoder::get_crate_deps(cdata);
    for dep in r.iter() {
        let extrn_cnum = dep.cnum;
        debug!("resolving dep crate {} hash: `{}`", dep.crate_id, dep.hash);
        let local_cnum = resolve_crate(e, root,
                                       dep.crate_id.name.as_slice(),
                                       &dep.crate_id,
                                       Some(&dep.hash),
                                       span);
        cnum_map.insert(extrn_cnum, local_cnum);
    }
    return @RefCell::new(cnum_map);
}

pub struct Loader<'a> {
    env: Env<'a>,
}

impl<'a> Loader<'a> {
    pub fn new(sess: &'a Session) -> Loader<'a> {
        let os = driver::get_os(driver::host_triple()).unwrap();
        let os = session::sess_os_to_meta_os(os);
        Loader {
            env: Env {
                sess: sess,
                os: os,
                crate_cache: @RefCell::new(Vec::new()),
                next_crate_num: 1,
                intr: token::get_ident_interner(),
            }
        }
    }
}

impl<'a> CrateLoader for Loader<'a> {
    fn load_crate(&mut self, krate: &ast::ViewItem) -> MacroCrate {
        let info = extract_crate_info(&self.env, krate).unwrap();
        let cnum = resolve_crate(&mut self.env, &None, info.ident,
                                 &info.crate_id, None, krate.span);
        let library = self.env.sess.cstore.get_used_crate_source(cnum).unwrap();
        MacroCrate {
            lib: library.dylib,
            cnum: cnum
        }
    }

    fn get_exported_macros(&mut self, cnum: ast::CrateNum) -> Vec<~str> {
        csearch::get_exported_macros(&self.env.sess.cstore, cnum).move_iter()
                                                                 .collect()
    }

    fn get_registrar_symbol(&mut self, cnum: ast::CrateNum) -> Option<~str> {
        let cstore = &self.env.sess.cstore;
        csearch::get_macro_registrar_fn(cstore, cnum)
            .map(|did| csearch::get_symbol(cstore, did))
    }
}
