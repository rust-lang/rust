// Extracting metadata from crate files

import driver::session;
import session::session;
import syntax::{ast, ast_util};
import lib::llvm::{False, llvm, mk_object_file, mk_section_iter};
import syntax::attr;
import syntax::visit;
import syntax::codemap::span;
import util::{filesearch};
import io::writer_util;
import std::map::{hashmap, int_hash};
import syntax::print::pprust;
import common::*;

export read_crates;
export list_file_metadata;

// Traverses an AST, reading all the information about use'd crates and native
// libraries necessary for later resolving, typechecking, linking, etc.
fn read_crates(sess: session::session, crate: ast::crate) {
    let e = @{sess: sess,
              mut crate_cache: [],
              mut next_crate_num: 1};
    let v =
        visit::mk_simple_visitor(@{visit_view_item:
                                       bind visit_view_item(e, _),
                                   visit_item: bind visit_item(e, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), v);
    dump_crates(e.crate_cache);
    warn_if_multiple_versions(sess, copy e.crate_cache);
}

type cache_entry = {
    cnum: int,
    span: span,
    hash: str,
    metas: @[@ast::meta_item]
};

fn dump_crates(crate_cache: [cache_entry]) {
    #debug("resolved crates:");
    for crate_cache.each {|entry|
        #debug("cnum: %?", entry.cnum);
        #debug("span: %?", entry.span);
        #debug("hash: %?", entry.hash);
        let attrs = [
            attr::mk_attr(attr::mk_list_item("link", *entry.metas))
        ];
        for attr::find_linkage_attrs(attrs).each {|attr|
            #debug("meta: %s", pprust::attr_to_str(attr));
        }
    }
}

fn warn_if_multiple_versions(sess: session::session,
                             crate_cache: [cache_entry]) {
    import either::*;

    if crate_cache.is_not_empty() {
        let name = crate_name_from_metas(*crate_cache.last().metas);
        let {lefts: matches, rights: non_matches} =
            partition(crate_cache.map {|entry|
                let othername = crate_name_from_metas(*entry.metas);
                if name == othername {
                    left(entry)
                } else {
                    right(entry)
                }
            });

        assert matches.is_not_empty();

        if matches.len() != 1u {
            sess.warn(#fmt("using multiple versions of crate `%s`", name));
            for matches.each {|match|
                sess.span_note(match.span, "used here");
                let attrs = [
                    attr::mk_attr(attr::mk_list_item("link", *match.metas))
                ];
                note_linkage_attrs(sess, attrs);
            }
        }

        warn_if_multiple_versions(sess, non_matches);
    }
}

type env = @{sess: session::session,
             mut crate_cache: [cache_entry],
             mut next_crate_num: ast::crate_num};

fn visit_view_item(e: env, i: @ast::view_item) {
    alt i.node {
      ast::view_item_use(ident, meta_items, id) {
        #debug("resolving use stmt. ident: %?, meta: %?", ident, meta_items);
        let cnum = resolve_crate(e, ident, meta_items, "", i.span);
        cstore::add_use_stmt_cnum(e.sess.cstore, id, cnum);
      }
      _ { }
    }
}

fn visit_item(e: env, i: @ast::item) {
    alt i.node {
      ast::item_native_mod(m) {
        alt attr::native_abi(i.attrs) {
          either::right(abi) {
            if abi != ast::native_abi_cdecl &&
               abi != ast::native_abi_stdcall { ret; }
          }
          either::left(msg) { e.sess.span_fatal(i.span, msg); }
        }

        let cstore = e.sess.cstore;
        let native_name =
            alt attr::get_meta_item_value_str_by_name(i.attrs, "link_name") {
              some(nn) {
                if nn == "" {
                    e.sess.span_fatal(
                        i.span,
                        "empty #[link_name] not allowed; use #[nolink].");
                }
                nn
              }
              none { i.ident }
            };
        let mut already_added = false;
        if vec::len(attr::find_attrs_by_name(i.attrs, "nolink")) == 0u {
            already_added = !cstore::add_used_library(cstore, native_name);
        }
        let link_args = attr::find_attrs_by_name(i.attrs, "link_args");
        if vec::len(link_args) > 0u && already_added {
            e.sess.span_fatal(i.span, "library '" + native_name +
                              "' already added: can't specify link_args.");
        }
        for link_args.each {|a|
            alt attr::get_meta_item_value_str(attr::attr_meta(a)) {
              some(linkarg) {
                cstore::add_used_link_args(cstore, linkarg);
              }
              none {/* fallthrough */ }
            }
        }
      }
      _ { }
    }
}

// A diagnostic function for dumping crate metadata to an output stream
fn list_file_metadata(sess: session::session, path: str, out: io::writer) {
    alt get_metadata_section(sess, path) {
      option::some(bytes) { decoder::list_crate_metadata(bytes, out); }
      option::none {
        out.write_str("could not find metadata in " + path + ".\n");
      }
    }
}

fn crate_matches(crate_data: @[u8], metas: [@ast::meta_item], hash: str) ->
    bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);
    if hash.is_not_empty() {
        let chash = decoder::get_crate_hash(crate_data);
        if chash != hash { ret false; }
    }
    metadata_matches(linkage_metas, metas)
}

fn metadata_matches(extern_metas: [@ast::meta_item],
                    local_metas: [@ast::meta_item]) -> bool {

    #debug("matching %u metadata requirements against %u items",
           vec::len(local_metas), vec::len(extern_metas));

    #debug("crate metadata:");
    for extern_metas.each {|have|
        #debug("  %s", pprust::meta_item_to_str(*have));
    }

    for local_metas.each {|needed|
        #debug("looking for %s", pprust::meta_item_to_str(*needed));
        if !attr::contains(extern_metas, needed) {
            #debug("missing %s", pprust::meta_item_to_str(*needed));
            ret false;
        }
    }
    ret true;
}

fn default_native_lib_naming(sess: session::session, static: bool) ->
   {prefix: str, suffix: str} {
    if static { ret {prefix: "lib", suffix: ".rlib"}; }
    alt sess.targ_cfg.os {
      session::os_win32 { ret {prefix: "", suffix: ".dll"}; }
      session::os_macos { ret {prefix: "lib", suffix: ".dylib"}; }
      session::os_linux { ret {prefix: "lib", suffix: ".so"}; }
      session::os_freebsd { ret {prefix: "lib", suffix: ".so"}; }
    }
}

fn crate_name_from_metas(metas: [@ast::meta_item]) -> str {
    let name_items = attr::find_meta_items_by_name(metas, "name");
    alt vec::last_opt(name_items) {
      some(i) {
        alt attr::get_meta_item_value_str(i) {
          some(n) { n }
          // FIXME: Probably want a warning here since the user
          // is using the wrong type of meta item
          _ { fail }
        }
      }
      none { fail "expected to find the crate name" }
    }
}

fn find_library_crate(sess: session::session, span: span,
                      metas: [@ast::meta_item], hash: str)
   -> option<{ident: str, data: @[u8]}> {

    attr::require_unique_names(sess.diagnostic(), metas);
    let metas = metas;

    let nn = default_native_lib_naming(sess, sess.opts.static);
    let x =
        find_library_crate_aux(sess, span, nn,
                               metas, hash, sess.filesearch);
    if x != none || sess.opts.static { ret x; }
    let nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(sess, span, nn2, metas, hash,
                               sess.filesearch);
}

fn find_library_crate_aux(sess: session::session,
                          span: span,
                          nn: {prefix: str, suffix: str},
                          metas: [@ast::meta_item],
                          hash: str,
                          filesearch: filesearch::filesearch) ->
   option<{ident: str, data: @[u8]}> {
    let crate_name = crate_name_from_metas(metas);
    let prefix: str = nn.prefix + crate_name + "-";
    let suffix: str = nn.suffix;

    let mut matches = [];
    filesearch::search(filesearch, { |path|
        #debug("inspecting file %s", path);
        let f: str = path::basename(path);
        if !(str::starts_with(f, prefix) && str::ends_with(f, suffix)) {
            #debug("skipping %s, doesn't look like %s*%s", path, prefix,
                   suffix);
            option::none
        } else {
            #debug("%s is a candidate", path);
            alt get_metadata_section(sess, path) {
              option::some(cvec) {
                if !crate_matches(cvec, metas, hash) {
                    #debug("skipping %s, metadata doesn't match", path);
                    option::none
                } else {
                    #debug("found %s with matching metadata", path);
                    matches += [{ident: path, data: cvec}];
                    option::none
                }
              }
              _ {
                #debug("could not load metadata for %s", path);
                option::none
              }
            }
        }
    });

    if matches.is_empty() {
        none
    } else if matches.len() == 1u {
        some(matches[0])
    } else {
        sess.span_err(
            span, #fmt("multiple matching crates for `%s`", crate_name));
        sess.note("candidates:");
        for matches.each {|match|
            sess.note(#fmt("path: %s", match.ident));
            let attrs = decoder::get_crate_attributes(match.data);
            note_linkage_attrs(sess, attrs);
        }
        sess.abort_if_errors();
        none
    }
}

fn note_linkage_attrs(sess: session::session, attrs: [ast::attribute]) {
    for attr::find_linkage_attrs(attrs).each {|attr|
        sess.note(#fmt("meta: %s", pprust::attr_to_str(attr)));
    }
}

fn get_metadata_section(sess: session::session,
                        filename: str) -> option<@[u8]> unsafe {
    let mb = str::as_c_str(filename, {|buf|
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
                                   });
    if mb as int == 0 { ret option::none::<@[u8]>; }
    let of = alt mk_object_file(mb) {
        option::some(of) { of }
        _ { ret option::none::<@[u8]>; }
    };
    let si = mk_section_iter(of.llof);
    while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
        let name_buf = llvm::LLVMGetSectionName(si.llsi);
        let name = unsafe { str::unsafe::from_c_str(name_buf) };
        if str::eq(name, sess.targ_cfg.target_strs.meta_sect_name) {
            let cbuf = llvm::LLVMGetSectionContents(si.llsi);
            let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
            unsafe {
                let cvbuf: *u8 = unsafe::reinterpret_cast(cbuf);
                ret some(@vec::unsafe::from_buf(cvbuf, csz));
            }
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none::<@[u8]>;
}

fn load_library_crate(sess: session::session, ident: ast::ident, span: span,
                      metas: [@ast::meta_item], hash: str)
   -> {ident: str, data: @[u8]} {


    alt find_library_crate(sess, span, metas, hash) {
      some(t) { ret t; }
      none {
        sess.span_fatal(span, #fmt["can't find crate for '%s'", ident]);
      }
    }
}

fn metas_with(ident: ast::ident, key: str,
                    metas: [@ast::meta_item]) -> [@ast::meta_item] {
    let name_items = attr::find_meta_items_by_name(metas, key);
    if name_items.is_empty() {
        metas + [attr::mk_name_value_item_str(key, ident)]
    } else {
        metas
    }
}

fn metas_with_ident(ident: ast::ident,
                    metas: [@ast::meta_item]) -> [@ast::meta_item] {
    metas_with(ident, "name", metas)
}

fn existing_match(e: env, metas: [@ast::meta_item], hash: str) ->
    option<int> {
    let maybe_entry = e.crate_cache.find {|c|
        metadata_matches(*c.metas, metas) &&
            (hash.is_empty() || c.hash == hash)
    };

    maybe_entry.map {|c| c.cnum }
}

fn resolve_crate(e: env, ident: ast::ident, metas: [@ast::meta_item],
                 hash: str, span: span) -> ast::crate_num {
    let metas = metas_with_ident(ident, metas);

    alt existing_match(e, metas, hash) {
      none {
        let cinfo =
            load_library_crate(e.sess, ident, span, metas, hash);

        let cfilename = cinfo.ident;
        let cdata = cinfo.data;

        let attrs = decoder::get_crate_attributes(cdata);
        let linkage_metas = attr::find_linkage_metas(attrs);
        let hash = decoder::get_crate_hash(cdata);

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        e.crate_cache += [{cnum: cnum, span: span,
                           hash: hash, metas: @linkage_metas}];
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, cdata);

        let cname = alt attr::meta_item_value_from_list(metas, "name") {
          option::some(v) { v }
          option::none { ident }
        };
        let cmeta = @{name: cname, data: cdata,
                      cnum_map: cnum_map, cnum: cnum};

        let cstore = e.sess.cstore;
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
fn resolve_crate_deps(e: env, cdata: @[u8]) -> cstore::cnum_map {
    #debug("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let cnum_map = int_hash::<ast::crate_num>();
    for decoder::get_crate_deps(cdata).each {|dep|
        let extrn_cnum = dep.cnum;
        let cname = dep.name;
        let cmetas = metas_with(dep.vers, "vers", []);
        #debug("resolving dep crate %s ver: %s hash: %s",
               dep.name, dep.vers, dep.hash);
        alt existing_match(e, metas_with_ident(cname, cmetas), dep.hash) {
          some(local_cnum) {
            #debug("already have it");
            // We've already seen this crate
            cnum_map.insert(extrn_cnum, local_cnum);
          }
          none {
            #debug("need to load it");
            // This is a new one so we've got to load it
            // FIXME: Need better error reporting than just a bogus span
            let fake_span = ast_util::dummy_sp();
            let local_cnum =
                resolve_crate(e, cname, cmetas, dep.hash, fake_span);
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
