// Extracting metadata from crate files

import driver::session;
import syntax::{ast, ast_util};
import lib::llvm::{False, llvm, mk_object_file, mk_section_iter};
import front::attr;
import middle::resolve;
import syntax::visit;
import syntax::codemap::span;
import back::x86;
import util::{common, filesearch};
import std::{vec, str, fs, io, option};
import std::option::{none, some};
import std::map::{hashmap, new_int_hash};
import syntax::print::pprust;
import common::*;

export read_crates;
export list_file_metadata;

// Traverses an AST, reading all the information about use'd crates and native
// libraries necessary for later resolving, typechecking, linking, etc.
fn read_crates(sess: session::session, crate: ast::crate) {
    let e =
        @{sess: sess,
          crate_cache: @std::map::new_str_hash::<int>(),
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
      crate_cache: @hashmap<str, int>,
      mutable next_crate_num: ast::crate_num};

fn visit_view_item(e: env, i: @ast::view_item) {
    alt i.node {
      ast::view_item_use(ident, meta_items, id) {
        let cnum = resolve_crate(e, ident, meta_items, i.span);
        cstore::add_use_stmt_cnum(e.sess.get_cstore(), id, cnum);
      }
      _ { }
    }
}

fn visit_item(e: env, i: @ast::item) {
    alt i.node {
      ast::item_native_mod(m) {
        if m.abi != ast::native_abi_rust && m.abi != ast::native_abi_cdecl &&
                m.abi != ast::native_abi_c_stack_cdecl &&
                m.abi != ast::native_abi_c_stack_stdcall {
            ret;
        }
        let cstore = e.sess.get_cstore();
        if !cstore::add_used_library(cstore, m.native_name) { ret; }
        for a: ast::attribute in
            attr::find_attrs_by_name(i.attrs, "link_args") {

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
        out.write_str("Could not find metadata in " + path + ".\n");
      }
    }
}

fn metadata_matches(crate_data: @[u8], metas: [@ast::meta_item]) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);

    log #fmt["matching %u metadata requirements against %u items",
             vec::len(metas), vec::len(linkage_metas)];

    log #fmt("crate metadata:");
    for have: @ast::meta_item in linkage_metas {
        log #fmt("  %s", pprust::meta_item_to_str(*have));
    }

    for needed: @ast::meta_item in metas {
        log #fmt["looking for %s", pprust::meta_item_to_str(*needed)];
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

fn find_library_crate(sess: session::session, ident: ast::ident,
                      metas: [@ast::meta_item])
   -> option::t<{ident: str, data: @[u8]}> {

    attr::require_unique_names(sess, metas);

    let crate_name =
        {
            let name_items = attr::find_meta_items_by_name(metas, "name");
            alt vec::last(name_items) {
              some(i) {
                alt attr::get_meta_item_value_str(i) {
                  some(n) { n }
                  // FIXME: Probably want a warning here since the user
                  // is using the wrong type of meta item
                  _ { ident }
                }
              }
              none. { ident }
            }
        };

    let nn = default_native_lib_naming(sess, sess.get_opts().static);
    let x =
        find_library_crate_aux(nn, crate_name, metas,
                               sess.filesearch());
    if x != none || sess.get_opts().static { ret x; }
    let nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(nn2, crate_name, metas,
                               sess.filesearch());
}

fn find_library_crate_aux(nn: {prefix: str, suffix: str}, crate_name: str,
                          metas: [@ast::meta_item],
                          filesearch: filesearch::filesearch) ->
   option::t<{ident: str, data: @[u8]}> {
    let prefix: str = nn.prefix + crate_name;
    let suffix: str = nn.suffix;

    ret filesearch::search(filesearch, { |path|
        log #fmt("inspecting file %s", path);
        let f: str = fs::basename(path);
        if !(str::starts_with(f, prefix) && str::ends_with(f, suffix)) {
            log #fmt["skipping %s, doesn't look like %s*%s", path, prefix,
                     suffix];
            option::none
        } else {
            log #fmt("%s is a candidate", path);
            alt get_metadata_section(path) {
              option::some(cvec) {
                if !metadata_matches(cvec, metas) {
                    log #fmt["skipping %s, metadata doesn't match", path];
                    option::none
                } else {
                    log #fmt["found %s with matching metadata", path];
                    option::some({ident: path, data: cvec})
                }
              }
              _ {
                log #fmt("could not load metadata for %s", path);
                option::none
              }
            }
        }
    });
}

fn get_metadata_section(filename: str) -> option::t<@[u8]> unsafe {
    let mb = str::as_buf(filename, {|buf|
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
                                   });
    if mb as int == 0 { ret option::none::<@[u8]>; }
    let of = mk_object_file(mb);
    let si = mk_section_iter(of.llof);
    while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
        let name_buf = llvm::LLVMGetSectionName(si.llsi);
        let name = str::str_from_cstr(name_buf);
        if str::eq(name, x86::get_meta_sect_name()) {
            let cbuf = llvm::LLVMGetSectionContents(si.llsi);
            let csz = llvm::LLVMGetSectionSize(si.llsi);
            let cvbuf: *u8 = std::unsafe::reinterpret_cast(cbuf);
            ret option::some::<@[u8]>(@vec::unsafe::from_buf(cvbuf, csz));
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none::<@[u8]>;
}

fn load_library_crate(sess: session::session, span: span, ident: ast::ident,
                      metas: [@ast::meta_item])
   -> {ident: str, data: @[u8]} {


    alt find_library_crate(sess, ident, metas) {
      some(t) { ret t; }
      none. {
        sess.span_fatal(span, #fmt["can't find crate for '%s'", ident]);
      }
    }
}

fn resolve_crate(e: env, ident: ast::ident, metas: [@ast::meta_item],
                 span: span) -> ast::crate_num {
    if !e.crate_cache.contains_key(ident) {
        let cinfo =
            load_library_crate(e.sess, span, ident, metas);

        let cfilename = cinfo.ident;
        let cdata = cinfo.data;

        // Claim this crate number and cache it
        let cnum = e.next_crate_num;
        e.crate_cache.insert(ident, cnum);
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        let cnum_map = resolve_crate_deps(e, cdata);

        let cmeta = {name: ident, data: cdata, cnum_map: cnum_map};

        let cstore = e.sess.get_cstore();
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, cfilename);
        ret cnum;
    } else { ret e.crate_cache.get(ident); }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: env, cdata: @[u8]) -> cstore::cnum_map {
    log "resolving deps of external crate";
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let cnum_map = new_int_hash::<ast::crate_num>();
    for dep: decoder::crate_dep in decoder::get_crate_deps(cdata) {
        let extrn_cnum = dep.cnum;
        let cname = dep.ident;
        log #fmt["resolving dep %s", cname];
        if e.crate_cache.contains_key(cname) {
            log "already have it";
            // We've already seen this crate
            let local_cnum = e.crate_cache.get(cname);
            cnum_map.insert(extrn_cnum, local_cnum);
        } else {
            log "need to load it";
            // This is a new one so we've got to load it
            // FIXME: Need better error reporting than just a bogus span
            let fake_span = ast_util::dummy_sp();
            let local_cnum = resolve_crate(e, cname, [], fake_span);
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
