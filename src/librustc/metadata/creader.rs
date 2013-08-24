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


use metadata::cstore;
use metadata::decoder;
use metadata::filesearch::FileSearch;
use metadata::loader;
use metadata::loader::MetadataSection;

use std::hashmap::HashMap;
use syntax::ast;
use std::vec;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{span, dummy_sp};
use syntax::diagnostic::span_handler;
use syntax::parse::token;
use syntax::parse::token::ident_interner;
use syntax::oldvisit;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(diag: @mut span_handler,
                   crate: &ast::Crate,
                   cstore: @mut cstore::CStore,
                   filesearch: @FileSearch,
                   os: loader::os,
                   statik: bool,
                   intr: @ident_interner) {
    let e = @mut Env {
        diag: diag,
        filesearch: filesearch,
        cstore: cstore,
        os: os,
        statik: statik,
        crate_cache: @mut ~[],
        next_crate_num: 1,
        intr: intr
    };
    let v =
        oldvisit::mk_simple_visitor(@oldvisit::SimpleVisitor {
            visit_view_item: |a| visit_view_item(e, a),
            visit_item: |a| visit_item(e, a),
            .. *oldvisit::default_simple_visitor()});
    visit_crate(e, crate);
    oldvisit::visit_crate(crate, ((), v));
    dump_crates(*e.crate_cache);
    warn_if_multiple_versions(e, diag, *e.crate_cache);
}

#[deriving(Clone)]
struct cache_entry {
    cnum: int,
    span: span,
    hash: @str,
    metas: @~[@ast::MetaItem]
}

fn dump_crates(crate_cache: &[cache_entry]) {
    debug!("resolved crates:");
    for entry in crate_cache.iter() {
        debug!("cnum: %?", entry.cnum);
        debug!("span: %?", entry.span);
        debug!("hash: %?", entry.hash);
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
                fmt!("using multiple versions of crate `%s`", name));
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
    diag: @mut span_handler,
    filesearch: @FileSearch,
    cstore: @mut cstore::CStore,
    os: loader::os,
    statik: bool,
    crate_cache: @mut ~[cache_entry],
    next_crate_num: ast::CrateNum,
    intr: @ident_interner
}

fn visit_crate(e: &Env, c: &ast::Crate) {
    let cstore = e.cstore;

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
              Some(p) => {
                  let p_path = Path(p);
                  match p_path.filestem() {
                      Some(s) =>
                          vec::append(
                              ~[attr::mk_name_value_item_str(@"package_id", p),
                               attr::mk_name_value_item_str(@"name", s.to_managed())],
                              *meta_items),
                      None => e.diag.span_bug(i.span, "Bad package path in `extern mod` item")
                  }
            }
          };
          debug!("resolving extern mod stmt. ident: %?, meta: %?",
                 ident, meta_items);
          let cnum = resolve_crate(e,
                                   ident,
                                   meta_items,
                                   @"",
                                   i.span);
          cstore::add_extern_mod_stmt_cnum(e.cstore, id, cnum);
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

        let cstore = e.cstore;
        let mut already_added = false;
        let link_args = i.attrs.iter()
            .filter_map(|at| if "link_args" == at.name() {Some(at)} else {None})
            .collect::<~[&ast::Attribute]>();

        match fm.sort {
            ast::named => {
                let link_name = i.attrs.iter()
                    .find(|at| "link_name" == at.name())
                    .chain(|at| at.value_str());

                let foreign_name = match link_name {
                        Some(nn) => {
                            if nn.is_empty() {
                                e.diag.span_fatal(
                                    i.span,
                                    "empty #[link_name] not allowed; use \
                                     #[nolink].");
                            }
                            nn
                        }
                        None => token::ident_to_str(&i.ident)
                    };
                if !attr::contains_name(i.attrs, "nolink") {
                    already_added =
                        !cstore::add_used_library(cstore, foreign_name);
                }
                if !link_args.is_empty() && already_added {
                    e.diag.span_fatal(i.span, ~"library '" + foreign_name +
                               "' already added: can't specify link_args.");
                }
            }
            ast::anonymous => { /* do nothing */ }
        }

        for m in link_args.iter() {
            match m.value_str() {
                Some(linkarg) => {
                    cstore::add_used_link_args(cstore, linkarg);
                }
                None => { /* fallthrough */ }
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
               -> Option<int> {
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
                 span: span)
              -> ast::CrateNum {
    let metas = metas_with_ident(ident, metas);

    match existing_match(e, metas, hash) {
      None => {
        let load_ctxt = loader::Context {
            diag: e.diag,
            filesearch: e.filesearch,
            span: span,
            ident: ident,
            metas: metas,
            hash: hash,
            os: e.os,
            is_static: e.statik,
            intr: e.intr
        };
        let (lident, ldata) = loader::load_library_crate(&load_ctxt);

        let cfilename = Path(lident);
        let cdata = ldata;

        let attrs = decoder::get_crate_attributes(cdata);
        let linkage_metas = attr::find_linkage_metas(attrs);
        let hash = decoder::get_crate_hash(cdata);

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
        let cnum_map = resolve_crate_deps(e, cdata);

        let cname =
            match attr::last_meta_item_value_str_by_name(load_ctxt.metas,
                                                         "name") {
                Some(v) => v,
                None => ident
            };
        let cmeta = @cstore::crate_metadata {
            name: cname,
            data: cdata,
            cnum_map: cnum_map,
            cnum: cnum
        };

        let cstore = e.cstore;
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, &cfilename);
        return cnum;
      }
      Some(cnum) => {
        return cnum;
      }
    }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: @mut Env, cdata: MetadataSection) -> cstore::cnum_map {
    debug!("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let mut cnum_map = HashMap::new();
    let r = decoder::get_crate_deps(cdata);
    for dep in r.iter() {
        let extrn_cnum = dep.cnum;
        let cname_str = token::ident_to_str(&dep.name);
        let cmetas = metas_with(dep.vers, @"vers", ~[]);
        debug!("resolving dep crate %s ver: %s hash: %s",
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
