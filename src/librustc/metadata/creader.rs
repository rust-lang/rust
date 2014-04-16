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
use metadata::cstore;
use metadata::cstore::CStore;
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

struct Env<'a> {
    sess: &'a Session,
    os: loader::Os,
    next_crate_num: ast::CrateNum,
    intr: Rc<IdentInterner>
}

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(sess: &Session,
                   krate: &ast::Crate,
                   os: loader::Os,
                   intr: Rc<IdentInterner>) {
    let mut e = Env {
        sess: sess,
        os: os,
        next_crate_num: sess.cstore.next_crate_num(),
        intr: intr
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
    cstore.iter_crate_data(|_, data| {
        debug!("crate_id: {}", data.crate_id());
        debug!("  cnum: {}", data.cnum);
        debug!("  hash: {}", data.hash());
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
            format!("using multiple versions of crate `{}`", name));
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
            let (cnum, _, _) = resolve_crate(e, &None, info.ident,
                                             &info.crate_id, None, true,
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
                None => from_str(ident.get().to_str()).unwrap()
            };
            Some(CrateInfo {
                ident: ident.get().to_str(),
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

fn resolve_crate<'a>(e: &mut Env,
                     root: &Option<CratePaths>,
                     ident: &str,
                     crate_id: &CrateId,
                     hash: Option<&Svh>,
                     should_link: bool,
                     span: Span)
                     -> (ast::CrateNum, @cstore::crate_metadata,
                         cstore::CrateSource) {
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
            let cnum_map = if should_link {
                resolve_crate_deps(e, root, metadata.as_slice(), span)
            } else {
                @RefCell::new(HashMap::new())
            };

            // Claim this crate number and cache it if we're linking to the
            // crate, otherwise it's a syntax-only crate and we don't need to
            // reserve a number
            let cnum = if should_link {
                let n = e.next_crate_num;
                e.next_crate_num += 1;
                n
            } else {
                -1
            };

            let cmeta = @cstore::crate_metadata {
                name: load_ctxt.crate_id.name.to_owned(),
                data: metadata,
                cnum_map: cnum_map,
                cnum: cnum,
                span: span,
            };

            let source = cstore::CrateSource {
                dylib: dylib,
                rlib: rlib,
                cnum: cnum,
            };

            if should_link {
                e.sess.cstore.set_crate_data(cnum, cmeta);
                e.sess.cstore.add_used_crate_source(source.clone());
            }
            (cnum, cmeta, source)
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
    let mut cnum_map = HashMap::new();
    let r = decoder::get_crate_deps(cdata);
    for dep in r.iter() {
        let extrn_cnum = dep.cnum;
        debug!("resolving dep crate {} hash: `{}`", dep.crate_id, dep.hash);
        let (local_cnum, _, _) = resolve_crate(e, root,
                                               dep.crate_id.name.as_slice(),
                                               &dep.crate_id,
                                               Some(&dep.hash),
                                               true,
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
                next_crate_num: sess.cstore.next_crate_num(),
                intr: token::get_ident_interner(),
            }
        }
    }
}

impl<'a> CrateLoader for Loader<'a> {
    fn load_crate(&mut self, krate: &ast::ViewItem) -> MacroCrate {
        let info = extract_crate_info(&self.env, krate).unwrap();
        let (_, data, library) = resolve_crate(&mut self.env, &None,
                                               info.ident, &info.crate_id,
                                               None, info.should_link,
                                               krate.span);
        let macros = decoder::get_exported_macros(data);
        let registrar = decoder::get_macro_registrar_fn(data).map(|id| {
            decoder::get_symbol(data.data.as_slice(), id)
        });
        MacroCrate {
            lib: library.dylib,
            macros: macros.move_iter().collect(),
            registrar_symbol: registrar,
        }
    }
}
