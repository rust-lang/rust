// Extracting metadata from crate files

import driver::session;
import syntax::{ast, ast_util};
import lib::llvm::{False, llvm, mk_object_file, mk_section_iter};
import front::attr;
import syntax::visit;
import syntax::codemap::span;
import util::{filesearch};
import std::{io, fs};
import io::writer_util;
import option::{none, some};
import std::map::{hashmap, new_int_hash};
import syntax::print::pprust;
import common::*;

export read_crates;
export list_file_metadata;

// Traverses an AST, reading all the information about use'd crates and native
// libraries necessary for later resolving, typechecking, linking, etc.
fn read_crates(sess: session::session, crate: ast::crate) {
    let e = @{sess: sess,
              crate_cache: std::map::new_str_hash::<int>(),
              mutable next_crate_num: 1};
    let v =
        visit::mk_simple_visitor(@{visit_view_item:
                                       bind visit_view_item(e, _),
                                   visit_item: bind visit_item(e, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), v);
}

type env = @{sess: session::session,
             crate_cache: hashmap<str, int>,
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
        alt attr::native_abi(i.attrs) {
          either::right(abi) {
            if abi != ast::native_abi_cdecl &&
               abi != ast::native_abi_stdcall { ret; }
          }
          either::left(msg) { e.sess.span_fatal(i.span, msg); }
        }

        let cstore = e.sess.get_cstore();
        let native_name = i.ident;
        let already_added = false;
        if vec::len(attr::find_attrs_by_name(i.attrs, "nolink")) == 0u {
            alt attr::get_meta_item_value_str_by_name(i.attrs, "link_name") {
              some(nn) { native_name = nn; }
              none. { }
            }
            if native_name == "" {
                e.sess.span_fatal(i.span,
                    "empty #[link_name] not allowed; use #[nolink].");
            }
            already_added = !cstore::add_used_library(cstore, native_name);
        }
        let link_args = attr::find_attrs_by_name(i.attrs, "link_args");
        if vec::len(link_args) > 0u && already_added {
            e.sess.span_fatal(i.span, "library '" + native_name +
                              "' already added: can't specify link_args.");
        }
        for a: ast::attribute in link_args {
            alt attr::get_meta_item_value_str(attr::attr_meta(a)) {
              some(linkarg) {
                cstore::add_used_link_args(cstore, linkarg);
              }
              none. {/* fallthrough */ }
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
      option::none. {
        out.write_str("Could not find metadata in " + path + ".\n");
      }
    }
}

fn metadata_matches(crate_data: @[u8], metas: [@ast::meta_item]) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);

    #debug("matching %u metadata requirements against %u items",
           vec::len(metas), vec::len(linkage_metas));

    #debug("crate metadata:");
    for have: @ast::meta_item in linkage_metas {
        #debug("  %s", pprust::meta_item_to_str(*have));
    }

    for needed: @ast::meta_item in metas {
        #debug("looking for %s", pprust::meta_item_to_str(*needed));
        if !attr::contains(linkage_metas, needed) {
            #debug("missing %s", pprust::meta_item_to_str(*needed));
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
      session::os_freebsd. { ret {prefix: "lib", suffix: ".so"}; }
    }
}

fn find_library_crate(sess: session::session, ident: ast::ident,
                      metas: [@ast::meta_item])
   -> option::t<{ident: str, data: @[u8]}> {

    attr::require_unique_names(sess, metas);
    let metas = metas;

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
        find_library_crate_aux(sess, nn, crate_name,
                               metas, sess.filesearch());
    if x != none || sess.get_opts().static { ret x; }
    let nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(sess, nn2, crate_name, metas,
                               sess.filesearch());
}

fn find_library_crate_aux(sess: session::session,
                          nn: {prefix: str, suffix: str},
                          crate_name: str,
                          metas: [@ast::meta_item],
                          filesearch: filesearch::filesearch) ->
   option::t<{ident: str, data: @[u8]}> {
    let prefix: str = nn.prefix + crate_name + "-";
    let suffix: str = nn.suffix;

    ret filesearch::search(filesearch, { |path|
        #debug("inspecting file %s", path);
        let f: str = fs::basename(path);
        if !(str::starts_with(f, prefix) && str::ends_with(f, suffix)) {
            #debug("skipping %s, doesn't look like %s*%s", path, prefix,
                   suffix);
            option::none
        } else {
            #debug("%s is a candidate", path);
            alt get_metadata_section(sess, path) {
              option::some(cvec) {
                if !metadata_matches(cvec, metas) {
                    #debug("skipping %s, metadata doesn't match", path);
                    option::none
                } else {
                    #debug("found %s with matching metadata", path);
                    option::some({ident: path, data: cvec})
                }
              }
              _ {
                #debug("could not load metadata for %s", path);
                option::none
              }
            }
        }
    });
}

fn get_metadata_section(sess: session::session,
                        filename: str) -> option::t<@[u8]> unsafe {
    let mb = str::as_buf(filename, {|buf|
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
        let name = unsafe { str::from_cstr(name_buf) };
        if str::eq(name, sess.get_targ_cfg().target_strs.meta_sect_name) {
            let cbuf = llvm::LLVMGetSectionContents(si.llsi);
            let csz = llvm::LLVMGetSectionSize(si.llsi);
            unsafe {
                let cvbuf: *u8 = unsafe::reinterpret_cast(cbuf);
                ret option::some::<@[u8]>(@vec::unsafe::from_buf(cvbuf, csz));
            }
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

        let cmeta = @{name: ident, data: cdata,
                      cnum_map: cnum_map, cnum: cnum};

        let cstore = e.sess.get_cstore();
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, cfilename);
        ret cnum;
    } else { ret e.crate_cache.get(ident); }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(e: env, cdata: @[u8]) -> cstore::cnum_map {
    #debug("resolving deps of external crate");
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    let cnum_map = new_int_hash::<ast::crate_num>();
    for dep: decoder::crate_dep in decoder::get_crate_deps(cdata) {
        let extrn_cnum = dep.cnum;
        let cname = dep.ident;
        #debug("resolving dep %s", cname);
        if e.crate_cache.contains_key(cname) {
            #debug("already have it");
            // We've already seen this crate
            let local_cnum = e.crate_cache.get(cname);
            cnum_map.insert(extrn_cnum, local_cnum);
        } else {
            #debug("need to load it");
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
// End:
