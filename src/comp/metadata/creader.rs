// Extracting metadata from crate files

import driver::session;
import syntax::ast;
import syntax::ast_util;
import lib::llvm::False;
import lib::llvm::llvm;
import lib::llvm::mk_object_file;
import lib::llvm::mk_section_iter;
import front::attr;
import middle::resolve;
import syntax::visit;
import syntax::codemap::span;
import back::x86;
import util::common;
import std::vec;
import std::str;
import std::istr;
import std::fs;
import std::io;
import std::option;
import std::option::none;
import std::option::some;
import std::map::hashmap;
import std::map::new_int_hash;
import syntax::print::pprust;
import common::*;

export read_crates;
export list_file_metadata;

// Traverses an AST, reading all the information about use'd crates and native
// libraries necessary for later resolving, typechecking, linking, etc.
fn read_crates(sess: session::session, crate: &ast::crate) {
    let e =
        @{sess: sess,
          crate_cache: @std::map::new_str_hash::<int>(),
          library_search_paths: sess.get_opts().library_search_paths,
          mutable next_crate_num: 1};
    let v =
        visit::mk_simple_visitor(@{visit_view_item:
                                       bind visit_view_item(e, _),
                                   visit_item: bind visit_item(e, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), v);
}

type env =
    @{sess: session::session,
      crate_cache: @hashmap<istr, int>,
      library_search_paths: [str],
      mutable next_crate_num: ast::crate_num};

fn visit_view_item(e: env, i: &@ast::view_item) {
    alt i.node {
      ast::view_item_use(ident, meta_items, id) {
        let cnum = resolve_crate(e, ident, meta_items, i.span);
        cstore::add_use_stmt_cnum(e.sess.get_cstore(), id, cnum);
      }
      _ { }
    }
}

fn visit_item(e: env, i: &@ast::item) {
    alt i.node {
      ast::item_native_mod(m) {
        if m.abi != ast::native_abi_rust && m.abi != ast::native_abi_cdecl {
            ret;
        }
        let cstore = e.sess.get_cstore();
        if !cstore::add_used_library(cstore, m.native_name) { ret; }
        for a: ast::attribute in
            attr::find_attrs_by_name(i.attrs, ~"link_args") {
            alt attr::get_meta_item_value_str(attr::attr_meta(a)) {
              some(linkarg) { cstore::add_used_link_args(cstore, linkarg); }
              none. {/* fallthrough */ }
            }
        }
      }
      _ { }
    }
}

// A diagnostic function for dumping crate metadata to an output stream
fn list_file_metadata(path: str, out: io::writer) {
    alt get_metadata_section(path) {
      option::some(bytes) { decoder::list_crate_metadata(bytes, out); }
      option::none. {
        out.write_str(
            istr::from_estr("Could not find metadata in " + path + ".\n"));
      }
    }
}

fn metadata_matches(crate_data: &@[u8], metas: &[@ast::meta_item]) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);

    log #fmt["matching %u metadata requirements against %u items",
             vec::len(metas), vec::len(linkage_metas)];

    for needed: @ast::meta_item in metas {
        if !attr::contains(linkage_metas, needed) {
            log #fmt["missing %s", pprust::meta_item_to_str(*needed)];
            ret false;
        }
    }
    ret true;
}

fn default_native_lib_naming(sess: session::session, static: bool) ->
   {prefix: str, suffix: str} {
    if static { ret {prefix: "lib", suffix: ".rlib"}; }
    alt sess.get_targ_cfg().os {
      session::os_win32. { ret {prefix: "", suffix: ".dll"}; }
      session::os_macos. { ret {prefix: "lib", suffix: ".dylib"}; }
      session::os_linux. { ret {prefix: "lib", suffix: ".so"}; }
    }
}

fn find_library_crate(sess: &session::session, ident: &ast::ident,
                      metas: &[@ast::meta_item], library_search_paths: &[str])
   -> option::t<{ident: str, data: @[u8]}> {

    attr::require_unique_names(sess, metas);

    // FIXME: Probably want a warning here since the user
    // is using the wrong type of meta item
    let crate_name =
        {
            let name_items = attr::find_meta_items_by_name(metas, ~"name");
            alt vec::last(name_items) {
              some(i) {
                alt attr::get_meta_item_value_str(i) {
                  some(n) { n }
                  _ { istr::to_estr(ident) }
                }
              }
              none. { istr::to_estr(ident) }
            }
        };

    let nn = default_native_lib_naming(sess, sess.get_opts().static);
    let x =
        find_library_crate_aux(nn, crate_name, metas, library_search_paths);
    if x != none || sess.get_opts().static { ret x; }
    let nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(nn2, crate_name, metas, library_search_paths);
}

fn find_library_crate_aux(nn: &{prefix: str, suffix: str}, crate_name: str,
                          metas: &[@ast::meta_item],
                          library_search_paths: &[str]) ->
   option::t<{ident: str, data: @[u8]}> {
    let prefix: istr = istr::from_estr(nn.prefix + crate_name);
    let suffix: istr = istr::from_estr(nn.suffix);
    // FIXME: we could probably use a 'glob' function in std::fs but it will
    // be much easier to write once the unsafe module knows more about FFI
    // tricks. Currently the glob(3) interface is a bit more than we can
    // stomach from here, and writing a C++ wrapper is more work than just
    // manually filtering fs::list_dir here.

    for library_search_path: str in library_search_paths {
        log #fmt["searching %s", library_search_path];
        let library_search_path = istr::from_estr(library_search_path);
        for path: istr in fs::list_dir(library_search_path) {
            log #fmt["searching %s", istr::to_estr(path)];
            let f: istr = fs::basename(path);
            let path = istr::to_estr(path);
            if !(istr::starts_with(f, prefix) && istr::ends_with(f, suffix))
               {
                log #fmt["skipping %s, doesn't look like %s*%s",
                         path,
                         istr::to_estr(prefix),
                         istr::to_estr(suffix)];
                cont;
            }
            alt get_metadata_section(path) {
              option::some(cvec) {
                if !metadata_matches(cvec, metas) {
                    log #fmt["skipping %s, metadata doesn't match", path];
                    cont;
                }
                log #fmt["found %s with matching metadata", path];
                ret some({ident: path, data: cvec});
              }
              _ { }
            }
        }
    }
    ret none;
}

fn get_metadata_section(filename: str) -> option::t<@[u8]> {
    let b = str::buf(filename);
    let mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(b);
    if mb as int == 0 { ret option::none::<@[u8]>; }
    let of = mk_object_file(mb);
    let si = mk_section_iter(of.llof);
    while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
        let name_buf = llvm::LLVMGetSectionName(si.llsi);
        let name = str::str_from_cstr(name_buf);
        if str::eq(name, istr::to_estr(x86::get_meta_sect_name())) {
            let cbuf = llvm::LLVMGetSectionContents(si.llsi);
            let csz = llvm::LLVMGetSectionSize(si.llsi);
            let cvbuf: *u8 = std::unsafe::reinterpret_cast(cbuf);
            ret option::some::<@[u8]>(@vec::unsafe::from_buf(cvbuf, csz));
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none::<@[u8]>;
}

fn load_library_crate(sess: &session::session, span: span, ident: &ast::ident,
                      metas: &[@ast::meta_item], library_search_paths: &[str])
   -> {ident: str, data: @[u8]} {


    alt find_library_crate(sess, ident, metas, library_search_paths) {
      some(t) { ret t; }
      none. {
        sess.span_fatal(span, #fmt["can't find crate for '%s'",
                                   istr::to_estr(ident)]);
      }
    }
}

fn resolve_crate(e: env, ident: &ast::ident, metas: [@ast::meta_item],
                 span: span) -> ast::crate_num {
    if !e.crate_cache.contains_key(ident) {
        let cinfo =
            load_library_crate(e.sess, span, ident, metas,
                               e.library_search_paths);

        let cfilename = cinfo.ident;
        let cdata = cinfo.data;

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        e.crate_cache.insert(ident, cnum);
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, cdata);

        let cmeta = {name: istr::to_estr(ident),
                     data: cdata, cnum_map: cnum_map};

        let cstore = e.sess.get_cstore();
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, cfilename);
        ret cnum;
    } else { ret e.crate_cache.get(ident); }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: env, cdata: &@[u8]) -> cstore::cnum_map {
    log "resolving deps of external crate";
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let cnum_map = new_int_hash::<ast::crate_num>();
    for dep: decoder::crate_dep in decoder::get_crate_deps(cdata) {
        let extrn_cnum = dep.cnum;
        let cname = dep.ident;
        log #fmt["resolving dep %s", cname];
        if e.crate_cache.contains_key(istr::from_estr(cname)) {
            log "already have it";
            // We've already seen this crate
            let local_cnum = e.crate_cache.get(istr::from_estr(cname));
            cnum_map.insert(extrn_cnum, local_cnum);
        } else {
            log "need to load it";
            // This is a new one so we've got to load it
            // FIXME: Need better error reporting than just a bogus span
            let fake_span = ast_util::dummy_sp();
            let local_cnum = resolve_crate(e,
                                           istr::from_estr(cname),
                                           [], fake_span);
            cnum_map.insert(extrn_cnum, local_cnum);
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
