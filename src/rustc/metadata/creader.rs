//! Validates all used crates and extern libraries and loads their metadata

import syntax::diagnostic::span_handler;
import syntax::{ast, ast_util};
import syntax::attr;
import syntax::visit;
import syntax::codemap::span;
import std::map::{hashmap, int_hash};
import syntax::print::pprust;
import filesearch::filesearch;
import common::*;
import dvec::{dvec, extensions};

export read_crates;

// Traverses an AST, reading all the information about use'd crates and extern
// libraries necessary for later resolving, typechecking, linking, etc.
fn read_crates(diag: span_handler, crate: ast::crate,
               cstore: cstore::cstore, filesearch: filesearch,
               os: loader::os, static: bool) {
    let e = @{diag: diag,
              filesearch: filesearch,
              cstore: cstore,
              os: os,
              static: static,
              crate_cache: dvec(),
              mut next_crate_num: 1};
    let v =
        visit::mk_simple_visitor(@{visit_view_item:
                                       |a| visit_view_item(e, a),
                                   visit_item: |a| visit_item(e, a)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), v);
    dump_crates(e.crate_cache);
    warn_if_multiple_versions(diag, e.crate_cache.get());
}

type cache_entry = {
    cnum: int,
    span: span,
    hash: @str,
    metas: @~[@ast::meta_item]
};

fn dump_crates(crate_cache: dvec<cache_entry>) {
    #debug("resolved crates:");
    for crate_cache.each |entry| {
        #debug("cnum: %?", entry.cnum);
        #debug("span: %?", entry.span);
        #debug("hash: %?", entry.hash);
        let attrs = ~[
            attr::mk_attr(attr::mk_list_item(@"link", *entry.metas))
        ];
        for attr::find_linkage_attrs(attrs).each |attr| {
            #debug("meta: %s", pprust::attr_to_str(attr));
        }
    }
}

fn warn_if_multiple_versions(diag: span_handler,
                             crate_cache: ~[cache_entry]) {
    import either::*;

    if crate_cache.len() != 0u {
        let name = loader::crate_name_from_metas(*crate_cache.last().metas);
        let {lefts: matches, rights: non_matches} =
            partition(crate_cache.map_to_vec(|entry| {
                let othername = loader::crate_name_from_metas(*entry.metas);
                if name == othername {
                    left(entry)
                } else {
                    right(entry)
                }
            }));

        assert matches.is_not_empty();

        if matches.len() != 1u {
            diag.handler().warn(
                #fmt("using multiple versions of crate `%s`", *name));
            for matches.each |match| {
                diag.span_note(match.span, "used here");
                let attrs = ~[
                    attr::mk_attr(attr::mk_list_item(@"link", *match.metas))
                ];
                loader::note_linkage_attrs(diag, attrs);
            }
        }

        warn_if_multiple_versions(diag, non_matches);
    }
}

type env = @{diag: span_handler,
             filesearch: filesearch,
             cstore: cstore::cstore,
             os: loader::os,
             static: bool,
             crate_cache: dvec<cache_entry>,
             mut next_crate_num: ast::crate_num};

fn visit_view_item(e: env, i: @ast::view_item) {
    alt i.node {
      ast::view_item_use(ident, meta_items, id) {
        #debug("resolving use stmt. ident: %?, meta: %?", ident, meta_items);
        let cnum = resolve_crate(e, ident, meta_items, "", i.span);
        cstore::add_use_stmt_cnum(e.cstore, id, cnum);
      }
      _ { }
    }
}

fn visit_item(e: env, i: @ast::item) {
    alt i.node {
      ast::item_foreign_mod(m) {
        alt attr::foreign_abi(i.attrs) {
          either::right(abi) {
            if abi != ast::foreign_abi_cdecl &&
               abi != ast::foreign_abi_stdcall { ret; }
          }
          either::left(msg) { e.diag.span_fatal(i.span, msg); }
        }

        let cstore = e.cstore;
        let foreign_name =
            alt attr::first_attr_value_str_by_name(i.attrs, "link_name") {
              some(nn) {
                if *nn == "" {
                    e.diag.span_fatal(
                        i.span,
                        "empty #[link_name] not allowed; use #[nolink].");
                }
                nn
              }
              none { i.ident }
            };
        let mut already_added = false;
        if vec::len(attr::find_attrs_by_name(i.attrs, "nolink")) == 0u {
            already_added = !cstore::add_used_library(cstore, *foreign_name);
        }
        let link_args = attr::find_attrs_by_name(i.attrs, "link_args");
        if vec::len(link_args) > 0u && already_added {
            e.diag.span_fatal(i.span, "library '" + *foreign_name +
                              "' already added: can't specify link_args.");
        }
        for link_args.each |a| {
            alt attr::get_meta_item_value_str(attr::attr_meta(a)) {
              some(linkarg) {
                cstore::add_used_link_args(cstore, *linkarg);
              }
              none {/* fallthrough */ }
            }
        }
      }
      _ { }
    }
}

fn metas_with(ident: ast::ident, key: ast::ident,
                    metas: ~[@ast::meta_item]) -> ~[@ast::meta_item] {
    let name_items = attr::find_meta_items_by_name(metas, *key);
    if name_items.is_empty() {
        vec::append_one(metas, attr::mk_name_value_item_str(key, *ident))
    } else {
        metas
    }
}

fn metas_with_ident(ident: ast::ident,
                    metas: ~[@ast::meta_item]) -> ~[@ast::meta_item] {
    metas_with(ident, @"name", metas)
}

fn existing_match(e: env, metas: ~[@ast::meta_item], hash: str) ->
    option<int> {

    for e.crate_cache.each |c| {
        if loader::metadata_matches(*c.metas, metas)
            && (hash.is_empty() || *c.hash == hash) {
            ret some(c.cnum);
        }
    }
    ret none;
}

fn resolve_crate(e: env, ident: ast::ident, metas: ~[@ast::meta_item],
                 hash: str, span: span) -> ast::crate_num {
    let metas = metas_with_ident(ident, metas);

    alt existing_match(e, metas, hash) {
      none {
        let load_ctxt: loader::ctxt = {
            diag: e.diag,
            filesearch: e.filesearch,
            span: span,
            ident: ident,
            metas: metas,
            hash: hash,
            os: e.os,
            static: e.static
        };
        let cinfo = loader::load_library_crate(load_ctxt);

        let cfilename = cinfo.ident;
        let cdata = cinfo.data;

        let attrs = decoder::get_crate_attributes(cdata);
        let linkage_metas = attr::find_linkage_metas(attrs);
        let hash = decoder::get_crate_hash(cdata);

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        e.crate_cache.push({cnum: cnum, span: span,
                            hash: hash, metas: @linkage_metas});
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, cdata);

        let cname =
            alt attr::last_meta_item_value_str_by_name(metas, "name") {
              option::some(v) { v }
              option::none { ident }
            };
        let cmeta = @{name: *cname, data: cdata,
                      cnum_map: cnum_map, cnum: cnum};

        let cstore = e.cstore;
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, cfilename);
        ret cnum;
      }
      some(cnum) {
        ret cnum;
      }
    }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: env, cdata: @~[u8]) -> cstore::cnum_map {
    #debug("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let cnum_map = int_hash::<ast::crate_num>();
    for decoder::get_crate_deps(cdata).each |dep| {
        let extrn_cnum = dep.cnum;
        let cname = dep.name;
        let cmetas = metas_with(dep.vers, @"vers", ~[]);
        #debug("resolving dep crate %s ver: %s hash: %s",
               *dep.name, *dep.vers, *dep.hash);
        alt existing_match(e, metas_with_ident(cname, cmetas), *dep.hash) {
          some(local_cnum) {
            #debug("already have it");
            // We've already seen this crate
            cnum_map.insert(extrn_cnum, local_cnum);
          }
          none {
            #debug("need to load it");
            // This is a new one so we've got to load it
            // FIXME (#2404): Need better error reporting than just a bogus
            // span.
            let fake_span = ast_util::dummy_sp();
            let local_cnum =
                resolve_crate(e, cname, cmetas, *dep.hash, fake_span);
            cnum_map.insert(extrn_cnum, local_cnum);
          }
        }
    }
    ret cnum_map;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
