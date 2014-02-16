// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Validates all used crates and extern libraries and loads their metadata

use driver::{driver, session};
use driver::session::Session;
use metadata::csearch;
use metadata::cstore;
use metadata::decoder;
use metadata::loader;
use metadata::loader::Os;

use std::cell::RefCell;
use std::hashmap::HashMap;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{Span, DUMMY_SP};
use syntax::diagnostic::SpanHandler;
use syntax::ext::base::{CrateLoader, MacroCrate};
use syntax::parse::token::{IdentInterner, InternedString};
use syntax::parse::token;
use syntax::crateid::CrateId;
use syntax::visit;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(sess: Session,
                   krate: &ast::Crate,
                   os: loader::Os,
                   intr: @IdentInterner) {
    let mut e = Env {
        sess: sess,
        os: os,
        crate_cache: @RefCell::new(~[]),
        next_crate_num: 1,
        intr: intr
    };
    visit_crate(&e, krate);
    {
        let mut v = ReadCrateVisitor {
            e: &mut e
        };
        visit::walk_crate(&mut v, krate, ());
    }
    let crate_cache = e.crate_cache.borrow();
    dump_crates(*crate_cache.get());
    warn_if_multiple_versions(&mut e, sess.diagnostic(), *crate_cache.get());
}

struct ReadCrateVisitor<'a> {
    e: &'a mut Env,
}

impl<'a> visit::Visitor<()> for ReadCrateVisitor<'a> {
    fn visit_view_item(&mut self, a: &ast::ViewItem, _: ()) {
        visit_view_item(self.e, a);
        visit::walk_view_item(self, a, ());
    }
    fn visit_item(&mut self, a: &ast::Item, _: ()) {
        visit_item(self.e, a);
        visit::walk_item(self, a, ());
    }
}

#[deriving(Clone)]
struct cache_entry {
    cnum: ast::CrateNum,
    span: Span,
    hash: ~str,
    crateid: CrateId,
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
                             diag: @SpanHandler,
                             crate_cache: &[cache_entry]) {
    if crate_cache.len() != 0u {
        let name = crate_cache[crate_cache.len() - 1].crateid.name.clone();

        let (matches, non_matches) = crate_cache.partitioned(|entry|
            name == entry.crateid.name);

        assert!(!matches.is_empty());

        if matches.len() != 1u {
            diag.handler().warn(
                format!("using multiple versions of crate `{}`", name));
            for match_ in matches.iter() {
                diag.span_note(match_.span, "used here");
                loader::note_crateid_attr(diag, &match_.crateid);
            }
        }

        warn_if_multiple_versions(e, diag, non_matches);
    }
}

struct Env {
    sess: Session,
    os: loader::Os,
    crate_cache: @RefCell<~[cache_entry]>,
    next_crate_num: ast::CrateNum,
    intr: @IdentInterner
}

fn visit_crate(e: &Env, c: &ast::Crate) {
    let cstore = e.sess.cstore;

    for a in c.attrs.iter().filter(|m| m.name().equiv(&("link_args"))) {
        match a.value_str() {
          Some(ref linkarg) => cstore.add_used_link_args(linkarg.get()),
          None => { /* fallthrough */ }
        }
    }
}

fn visit_view_item(e: &mut Env, i: &ast::ViewItem) {
    let should_load = i.attrs.iter().all(|attr| {
        attr.name().get() != "phase" ||
            attr.meta_item_list().map_or(false, |phases| {
                attr::contains_name(phases, "link")
            })
    });

    if !should_load {
        return;
    }

    match extract_crate_info(i) {
        Some(info) => {
            let cnum = resolve_crate(e,
                                     info.ident.clone(),
                                     info.name.clone(),
                                     info.version.clone(),
                                     ~"",
                                     i.span);
            e.sess.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
        }
        None => ()
    }
}

struct CrateInfo {
    ident: ~str,
    name: ~str,
    version: ~str,
    id: ast::NodeId,
}

fn extract_crate_info(i: &ast::ViewItem) -> Option<CrateInfo> {
    match i.node {
        ast::ViewItemExternMod(ident, ref path_opt, id) => {
            let ident = token::get_ident(ident);
            debug!("resolving extern crate stmt. ident: {:?} path_opt: {:?}",
                   ident, path_opt);
            let (name, version) = match *path_opt {
                Some((ref path_str, _)) => {
                    let crateid: Option<CrateId> = from_str(path_str.get());
                    match crateid {
                        None => (~"", ~""),
                        Some(crateid) => {
                            let version = match crateid.version {
                                None => ~"",
                                Some(ref ver) => ver.to_str(),
                            };
                            (crateid.name.to_str(), version)
                        }
                    }
                }
                None => (ident.get().to_str(), ~""),
            };
            Some(CrateInfo {
                  ident: ident.get().to_str(),
                  name: name,
                  version: version,
                  id: id,
            })
        }
        _ => None
    }
}

fn visit_item(e: &Env, i: &ast::Item) {
    match i.node {
        ast::ItemForeignMod(ref fm) => {
            if fm.abis.is_rust() || fm.abis.is_intrinsic() {
                return;
            }

            // First, add all of the custom link_args attributes
            let cstore = e.sess.cstore;
            let link_args = i.attrs.iter()
                .filter_map(|at| if at.name().equiv(&("link_args")) {
                    Some(at)
                } else {
                    None
                })
                .to_owned_vec();
            for m in link_args.iter() {
                match m.value_str() {
                    Some(linkarg) => cstore.add_used_link_args(linkarg.get()),
                    None => { /* fallthrough */ }
                }
            }

            // Next, process all of the #[link(..)]-style arguments
            let cstore = e.sess.cstore;
            let link_args = i.attrs.iter()
                .filter_map(|at| if at.name().equiv(&("link")) {
                    Some(at)
                } else {
                    None
                })
                .to_owned_vec();
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
                            cstore.add_used_library(n.get().to_owned(), kind);
                        }
                    }
                    None => {}
                }
            }
        }
        _ => { }
    }
}

fn existing_match(e: &Env, name: &str, version: &str, hash: &str) -> Option<ast::CrateNum> {
    let crate_cache = e.crate_cache.borrow();
    for c in crate_cache.get().iter() {
        let crateid_version = match c.crateid.version {
            None => ~"0.0",
            Some(ref ver) => ver.to_str(),
        };
        if (name.is_empty() || name == c.crateid.name) &&
            (version.is_empty() || version == crateid_version) &&
            (hash.is_empty() || hash == c.hash) {
            return Some(c.cnum);
        }
    }
    None
}

fn resolve_crate(e: &mut Env,
                 ident: ~str,
                 name: ~str,
                 version: ~str,
                 hash: ~str,
                 span: Span)
              -> ast::CrateNum {
    match existing_match(e, name, version, hash) {
      None => {
        let load_ctxt = loader::Context {
            sess: e.sess,
            span: span,
            ident: ident,
            name: name,
            version: version,
            hash: hash,
            os: e.os,
            intr: e.intr
        };
        let loader::Library {
            dylib, rlib, metadata
        } = load_ctxt.load_library_crate();

        let attrs = decoder::get_crate_attributes(metadata.as_slice());
        let crateid = attr::find_crateid(attrs).unwrap();
        let hash = decoder::get_crate_hash(metadata.as_slice());

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        {
            let mut crate_cache = e.crate_cache.borrow_mut();
            crate_cache.get().push(cache_entry {
                cnum: cnum,
                span: span,
                hash: hash,
                crateid: crateid,
            });
        }
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, metadata.as_slice());

        let cmeta = @cstore::crate_metadata {
            name: load_ctxt.name,
            data: metadata,
            cnum_map: cnum_map,
            cnum: cnum
        };

        let cstore = e.sess.cstore;
        cstore.set_crate_data(cnum, cmeta);
        cstore.add_used_crate_source(cstore::CrateSource {
            dylib: dylib,
            rlib: rlib,
            cnum: cnum,
        });
        return cnum;
      }
      Some(cnum) => {
        return cnum;
      }
    }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: &mut Env, cdata: &[u8]) -> cstore::cnum_map {
    debug!("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let mut cnum_map = HashMap::new();
    let r = decoder::get_crate_deps(cdata);
    for dep in r.iter() {
        let extrn_cnum = dep.cnum;
        let cname_str = token::get_ident(dep.name);
        debug!("resolving dep crate {} ver: {} hash: {}",
               cname_str, dep.vers, dep.hash);
        match existing_match(e,
                             cname_str.get(),
                             dep.vers,
                             dep.hash) {
          Some(local_cnum) => {
            debug!("already have it");
            // We've already seen this crate
            cnum_map.insert(extrn_cnum, local_cnum);
          }
          None => {
            debug!("need to load it");
            // This is a new one so we've got to load it
            // FIXME (#2404): Need better error reporting than just a bogus
            // span.
            let fake_span = DUMMY_SP;
            let local_cnum = resolve_crate(e,
                                           cname_str.get().to_str(),
                                           cname_str.get().to_str(),
                                           dep.vers.clone(),
                                           dep.hash.clone(),
                                           fake_span);
            cnum_map.insert(extrn_cnum, local_cnum);
          }
        }
    }
    return @RefCell::new(cnum_map);
}

pub struct Loader {
    priv env: Env,
}

impl Loader {
    pub fn new(sess: Session) -> Loader {
        let os = driver::get_os(driver::host_triple()).unwrap();
        let os = session::sess_os_to_meta_os(os);
        Loader {
            env: Env {
                sess: sess,
                os: os,
                crate_cache: @RefCell::new(~[]),
                next_crate_num: 1,
                intr: token::get_ident_interner(),
            }
        }
    }
}

impl CrateLoader for Loader {
    fn load_crate(&mut self, krate: &ast::ViewItem) -> MacroCrate {
        let info = extract_crate_info(krate).unwrap();
        let cnum = resolve_crate(&mut self.env,
                                 info.ident.clone(),
                                 info.name.clone(),
                                 info.version.clone(),
                                 ~"",
                                 krate.span);
        let library = self.env.sess.cstore.get_used_crate_source(cnum).unwrap();
        MacroCrate {
            lib: library.dylib,
            cnum: cnum
        }
    }

    fn get_exported_macros(&mut self, cnum: ast::CrateNum) -> ~[~str] {
        csearch::get_exported_macros(self.env.sess.cstore, cnum)
    }

    fn get_registrar_symbol(&mut self, cnum: ast::CrateNum) -> Option<~str> {
        let cstore = self.env.sess.cstore;
        csearch::get_macro_registrar_fn(cstore, cnum)
            .map(|did| csearch::get_symbol(cstore, did))
    }
}
