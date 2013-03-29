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

use core::prelude::*;

use metadata::cstore;
use metadata::decoder;
use metadata::filesearch::FileSearch;
use metadata::loader;

use core::either;
use core::hashmap::linear::LinearMap;
use core::vec;
use syntax::attr;
use syntax::codemap::{span, dummy_sp};
use syntax::diagnostic::span_handler;
use syntax::parse::token::ident_interner;
use syntax::visit;
use syntax::ast;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
pub fn read_crates(diag: @span_handler,
                   crate: ast::crate,
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
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_view_item: |a| visit_view_item(e, a),
            visit_item: |a| visit_item(e, a),
            .. *visit::default_simple_visitor()});
    visit_crate(e, crate);
    visit::visit_crate(crate, (), v);
    dump_crates(e.crate_cache);
    warn_if_multiple_versions(e, diag, e.crate_cache);
}

struct cache_entry {
    cnum: int,
    span: span,
    hash: @~str,
    metas: @~[@ast::meta_item]
}

fn dump_crates(crate_cache: @mut ~[cache_entry]) {
    debug!("resolved crates:");
    for crate_cache.each |entry| {
        debug!("cnum: %?", entry.cnum);
        debug!("span: %?", entry.span);
        debug!("hash: %?", entry.hash);
    }
}

fn warn_if_multiple_versions(e: @mut Env,
                             diag: @span_handler,
                             crate_cache: @mut ~[cache_entry]) {
    use core::either::*;

    let crate_cache = &mut *crate_cache;

    if crate_cache.len() != 0u {
        let name = loader::crate_name_from_metas(
            *crate_cache[crate_cache.len() - 1].metas
        );

        let (matches, non_matches) =
            partition(crate_cache.map_to_vec(|&entry| {
                let othername = loader::crate_name_from_metas(
                    copy *entry.metas);
                if name == othername {
                    Left(entry)
                } else {
                    Right(entry)
                }
            }));

        assert!(!matches.is_empty());

        if matches.len() != 1u {
            diag.handler().warn(
                fmt!("using multiple versions of crate `%s`", *name));
            for matches.each |match_| {
                diag.span_note(match_.span, ~"used here");
                let attrs = ~[
                    attr::mk_attr(attr::mk_list_item(
                        @~"link", /*bad*/copy *match_.metas))
                ];
                loader::note_linkage_attrs(e.intr, diag, attrs);
            }
        }

        warn_if_multiple_versions(e, diag, @mut non_matches);
    }
}

struct Env {
    diag: @span_handler,
    filesearch: @FileSearch,
    cstore: @mut cstore::CStore,
    os: loader::os,
    statik: bool,
    crate_cache: @mut ~[cache_entry],
    next_crate_num: ast::crate_num,
    intr: @ident_interner
}

fn visit_crate(e: @mut Env, c: ast::crate) {
    let cstore = e.cstore;
    let link_args = attr::find_attrs_by_name(c.node.attrs, "link_args");

    for link_args.each |a| {
        match attr::get_meta_item_value_str(attr::attr_meta(*a)) {
          Some(ref linkarg) => {
            cstore::add_used_link_args(cstore, **linkarg);
          }
          None => {/* fallthrough */ }
        }
    }
}

fn visit_view_item(e: @mut Env, i: @ast::view_item) {
    match i.node {
      ast::view_item_extern_mod(ident, /*bad*/copy meta_items, id) => {
        debug!("resolving extern mod stmt. ident: %?, meta: %?",
               ident, meta_items);
        let cnum = resolve_crate(e, ident, meta_items, @~"", i.span);
        cstore::add_extern_mod_stmt_cnum(e.cstore, id, cnum);
      }
      _ => ()
    }
}

fn visit_item(e: @mut Env, i: @ast::item) {
    match i.node {
      ast::item_foreign_mod(ref fm) => {
        match attr::foreign_abi(i.attrs) {
          either::Right(abi) => {
            if abi != ast::foreign_abi_cdecl &&
               abi != ast::foreign_abi_stdcall { return; }
          }
          either::Left(ref msg) => e.diag.span_fatal(i.span, (*msg))
        }

        let cstore = e.cstore;
        let mut already_added = false;
        let link_args = attr::find_attrs_by_name(i.attrs, "link_args");

        match fm.sort {
            ast::named => {
                let foreign_name =
                    match attr::first_attr_value_str_by_name(i.attrs,
                                                            ~"link_name") {
                        Some(nn) => {
                            if *nn == ~"" {
                                e.diag.span_fatal(
                                    i.span,
                                    ~"empty #[link_name] not allowed; use " +
                                    ~"#[nolink].");
                            }
                            nn
                        }
                        None => e.intr.get(i.ident)
                    };
                if attr::find_attrs_by_name(i.attrs, ~"nolink").is_empty() {
                    already_added =
                        !cstore::add_used_library(cstore, foreign_name);
                }
                if !link_args.is_empty() && already_added {
                    e.diag.span_fatal(i.span, ~"library '" + *foreign_name +
                               ~"' already added: can't specify link_args.");
                }
            }
            ast::anonymous => { /* do nothing */ }
        }

        for link_args.each |a| {
            match attr::get_meta_item_value_str(attr::attr_meta(*a)) {
                Some(linkarg) => {
                    cstore::add_used_link_args(cstore, *linkarg);
                }
                None => { /* fallthrough */ }
            }
        }
      }
      _ => { }
    }
}

fn metas_with(ident: @~str, key: @~str, +metas: ~[@ast::meta_item])
    -> ~[@ast::meta_item] {
    let name_items = attr::find_meta_items_by_name(metas, *key);
    if name_items.is_empty() {
        vec::append_one(metas, attr::mk_name_value_item_str(key, ident))
    } else {
        metas
    }
}

fn metas_with_ident(ident: @~str, +metas: ~[@ast::meta_item])
    -> ~[@ast::meta_item] {
    metas_with(ident, @~"name", metas)
}

fn existing_match(e: @mut Env, metas: &[@ast::meta_item], hash: @~str)
               -> Option<int> {
    for e.crate_cache.each |c| {
        if loader::metadata_matches(*c.metas, metas)
            && (hash.is_empty() || c.hash == hash) {
            return Some(c.cnum);
        }
    }
    return None;
}

fn resolve_crate(e: @mut Env,
                 ident: ast::ident,
                 +metas: ~[@ast::meta_item],
                 hash: @~str,
                 span: span)
              -> ast::crate_num {
    let metas = metas_with_ident(@/*bad*/copy *e.intr.get(ident), metas);

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
        let (lident, ldata) = loader::load_library_crate(load_ctxt);

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
                                                         ~"name") {
                Some(v) => v,
                None => e.intr.get(ident),
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
fn resolve_crate_deps(e: @mut Env, cdata: @~[u8]) -> cstore::cnum_map {
    debug!("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let mut cnum_map = LinearMap::new();
    for decoder::get_crate_deps(e.intr, cdata).each |dep| {
        let extrn_cnum = dep.cnum;
        let cname = dep.name;
        let cmetas = metas_with(dep.vers, @~"vers", ~[]);
        debug!("resolving dep crate %s ver: %s hash: %s",
               *e.intr.get(dep.name), *dep.vers, *dep.hash);
        match existing_match(e, metas_with_ident(e.intr.get(cname),
                                                 copy cmetas),
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
            let local_cnum = resolve_crate(e, cname, cmetas, dep.hash,
                                           fake_span);
            cnum_map.insert(extrn_cnum, local_cnum);
          }
        }
    }
    return @mut cnum_map;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
