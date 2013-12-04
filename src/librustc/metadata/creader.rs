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

use driver::session::Session;
use metadata::cstore;
use metadata::decoder;
use metadata::loader;

use std::hashmap::HashMap;
use syntax::ast;
use std::vec;
use syntax::abi;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{Span, dummy_sp};
use syntax::diagnostic::span_handler;
use syntax::parse::token;
use syntax::parse::token::ident_interner;
use syntax::visit;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(sess: Session,
                   crate: &ast::Crate,
                   os: loader::Os,
                   intr: @ident_interner) {
    let e = @mut Env {
        sess: sess,
        os: os,
        crate_cache: @mut ~[],
        next_crate_num: 1,
        intr: intr
    };
    let mut v = ReadCrateVisitor{ e:e };
    visit_crate(e, crate);
    visit::walk_crate(&mut v, crate, ());
    dump_crates(*e.crate_cache);
    warn_if_multiple_versions(e, sess.diagnostic(), *e.crate_cache);
}

struct ReadCrateVisitor { e:@mut Env }
impl visit::Visitor<()> for ReadCrateVisitor {
    fn visit_view_item(&mut self, a:&ast::view_item, _:()) {
        visit_view_item(self.e, a);
        visit::walk_view_item(self, a, ());
    }
    fn visit_item(&mut self, a:@ast::item, _:()) {
        visit_item(self.e, a);
        visit::walk_item(self, a, ());
    }
}

#[deriving(Clone)]
struct cache_entry {
    cnum: ast::CrateNum,
    span: Span,
    hash: @str,
    metas: @~[@ast::MetaItem]
}

fn dump_crates(crate_cache: &[cache_entry]) {
    debug!("resolved crates:");
    for entry in crate_cache.iter() {
        debug!("cnum: {:?}", entry.cnum);
        debug!("span: {:?}", entry.span);
        debug!("hash: {:?}", entry.hash);
    }
}

fn warn_if_multiple_versions(e: @mut Env,
                             diag: @mut span_handler,
                             crate_cache: &[cache_entry]) {
    if crate_cache.len() != 0u {
        let name = loader::crate_name_from_metas(
            *crate_cache[crate_cache.len() - 1].metas
        );

        let (matches, non_matches) = crate_cache.partitioned(|entry|
            name == loader::crate_name_from_metas(*entry.metas));

        assert!(!matches.is_empty());

        if matches.len() != 1u {
            diag.handler().warn(
                format!("using multiple versions of crate `{}`", name));
            for match_ in matches.iter() {
                diag.span_note(match_.span, "used here");
                let attrs = ~[
                    attr::mk_attr(attr::mk_list_item(@"link",
                                                     (*match_.metas).clone()))
                ];
                loader::note_linkage_attrs(e.intr, diag, attrs);
            }
        }

        warn_if_multiple_versions(e, diag, non_matches);
    }
}

struct Env {
    sess: Session,
    os: loader::Os,
    crate_cache: @mut ~[cache_entry],
    next_crate_num: ast::CrateNum,
    intr: @ident_interner
}

fn visit_crate(e: &Env, c: &ast::Crate) {
    let cstore = e.sess.cstore;

    for a in c.attrs.iter().filter(|m| "link_args" == m.name()) {
        match a.value_str() {
          Some(ref linkarg) => {
            cstore::add_used_link_args(cstore, *linkarg);
          }
          None => {/* fallthrough */ }
        }
    }
}

fn visit_view_item(e: @mut Env, i: &ast::view_item) {
    match i.node {
      ast::view_item_extern_mod(ident, path_opt, ref meta_items, id) => {
          let ident = token::ident_to_str(&ident);
          let meta_items = match path_opt {
              None => meta_items.clone(),
              Some((p, _path_str_style)) => {
                  let p_path = Path::new(p);
                  match p_path.filestem_str() {
                      None|Some("") =>
                          e.sess.span_bug(i.span, "Bad package path in `extern mod` item"),
                      Some(s) =>
                          vec::append(
                              ~[attr::mk_name_value_item_str(@"package_id", p),
                               attr::mk_name_value_item_str(@"name", s.to_managed())],
                              *meta_items)
                  }
            }
          };
          debug!("resolving extern mod stmt. ident: {:?}, meta: {:?}",
                 ident, meta_items);
          let cnum = resolve_crate(e,
                                   ident,
                                   meta_items,
                                   @"",
                                   i.span);
          cstore::add_extern_mod_stmt_cnum(e.sess.cstore, id, cnum);
      }
      _ => ()
  }
}

fn visit_item(e: &Env, i: @ast::item) {
    match i.node {
        ast::item_foreign_mod(ref fm) => {
            if fm.abis.is_rust() || fm.abis.is_intrinsic() {
                return;
            }

            // First, add all of the custom link_args attributes
            let cstore = e.sess.cstore;
            let link_args = i.attrs.iter()
                .filter_map(|at| if "link_args" == at.name() {Some(at)} else {None})
                .to_owned_vec();
            for m in link_args.iter() {
                match m.value_str() {
                    Some(linkarg) => {
                        cstore::add_used_link_args(cstore, linkarg);
                    }
                    None => { /* fallthrough */ }
                }
            }

            // Next, process all of the #[link(..)]-style arguments
            let cstore = e.sess.cstore;
            let link_args = i.attrs.iter()
                .filter_map(|at| if "link" == at.name() {Some(at)} else {None})
                .to_owned_vec();
            for m in link_args.iter() {
                match m.meta_item_list() {
                    Some(items) => {
                        let kind = items.iter().find(|k| {
                            "kind" == k.name()
                        }).and_then(|a| a.value_str());
                        let kind = match kind {
                            Some(k) => {
                                if "static" == k {
                                    cstore::NativeStatic
                                } else if e.sess.targ_cfg.os == abi::OsMacos &&
                                          "framework" == k {
                                    cstore::NativeFramework
                                } else if "framework" == k {
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
                            "name" == n.name()
                        }).and_then(|a| a.value_str());
                        let n = match n {
                            Some(n) => n,
                            None => {
                                e.sess.span_err(m.span,
                                    "#[link(...)] specified without \
                                     `name = \"foo\"`");
                                @"foo"
                            }
                        };
                        cstore::add_used_library(cstore, n.to_owned(), kind);
                    }
                    None => {}
                }
            }
        }
        _ => { }
    }
}

fn metas_with(ident: @str, key: @str, mut metas: ~[@ast::MetaItem])
    -> ~[@ast::MetaItem] {
    // Check if key isn't there yet.
    if !attr::contains_name(metas, key) {
        metas.push(attr::mk_name_value_item_str(key, ident));
    }
    metas
}

fn metas_with_ident(ident: @str, metas: ~[@ast::MetaItem])
    -> ~[@ast::MetaItem] {
    metas_with(ident, @"name", metas)
}

fn existing_match(e: &Env, metas: &[@ast::MetaItem], hash: &str)
               -> Option<ast::CrateNum> {
    for c in e.crate_cache.iter() {
        if loader::metadata_matches(*c.metas, metas)
            && (hash.is_empty() || c.hash.as_slice() == hash) {
            return Some(c.cnum);
        }
    }
    return None;
}

fn resolve_crate(e: @mut Env,
                 ident: @str,
                 metas: ~[@ast::MetaItem],
                 hash: @str,
                 span: Span)
              -> ast::CrateNum {
    let metas = metas_with_ident(ident, metas);

    match existing_match(e, metas, hash) {
      None => {
        let load_ctxt = loader::Context {
            sess: e.sess,
            span: span,
            ident: ident,
            metas: metas,
            hash: hash,
            os: e.os,
            intr: e.intr
        };
        let loader::Library {
            dylib, rlib, metadata
        } = load_ctxt.load_library_crate();

        let attrs = decoder::get_crate_attributes(metadata);
        let linkage_metas = attr::find_linkage_metas(attrs);
        let hash = decoder::get_crate_hash(metadata);

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        e.crate_cache.push(cache_entry {
            cnum: cnum,
            span: span,
            hash: hash,
            metas: @linkage_metas
        });
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, metadata);

        let cname =
            match attr::last_meta_item_value_str_by_name(load_ctxt.metas,
                                                         "name") {
                Some(v) => v,
                None => ident
            };
        let cmeta = @cstore::crate_metadata {
            name: cname,
            data: metadata,
            cnum_map: cnum_map,
            cnum: cnum
        };

        let cstore = e.sess.cstore;
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_source(cstore, cstore::CrateSource {
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
fn resolve_crate_deps(e: @mut Env, cdata: @~[u8]) -> cstore::cnum_map {
    debug!("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let mut cnum_map = HashMap::new();
    let r = decoder::get_crate_deps(cdata);
    for dep in r.iter() {
        let extrn_cnum = dep.cnum;
        let cname_str = token::ident_to_str(&dep.name);
        let cmetas = metas_with(dep.vers, @"vers", ~[]);
        debug!("resolving dep crate {} ver: {} hash: {}",
               cname_str, dep.vers, dep.hash);
        match existing_match(e,
                             metas_with_ident(cname_str, cmetas.clone()),
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
            let fake_span = dummy_sp();
            let local_cnum = resolve_crate(e, cname_str, cmetas, dep.hash,
                                           fake_span);
            cnum_map.insert(extrn_cnum, local_cnum);
          }
        }
    }
    return @mut cnum_map;
}
