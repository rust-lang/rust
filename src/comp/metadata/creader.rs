// Extracting metadata from crate files

import driver::session;
import syntax::ast;
import lib::llvm::False;
import lib::llvm::llvm;
import lib::llvm::mk_object_file;
import lib::llvm::mk_section_iter;
import front::attr;
import middle::resolve;
import syntax::walk;
import syntax::codemap::span;
import back::x86;
import util::common;
import std::ivec;
import std::str;
import std::vec;
import std::fs;
import std::ioivec;
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
fn read_crates(session::session sess,
               &ast::crate crate) {
    auto e =
        @rec(sess=sess,
             crate_cache=@std::map::new_str_hash[int](),
             library_search_paths=sess.get_opts().library_search_paths,
             mutable next_crate_num=1);
    auto v =
        rec(visit_view_item_pre=bind visit_view_item(e, _),
            visit_item_pre=bind visit_item(e, _)
            with walk::default_visitor());
    walk::walk_crate(v, crate);
}

type env =
    @rec(session::session sess,
         @hashmap[str, int] crate_cache,
         vec[str] library_search_paths,
         mutable ast::crate_num next_crate_num);

fn visit_view_item(env e, &@ast::view_item i) {
    alt (i.node) {
        case (ast::view_item_use(?ident, ?meta_items, ?id)) {
            auto cnum = resolve_crate(e, ident, meta_items, i.span);
            cstore::add_use_stmt_cnum(e.sess.get_cstore(), id, cnum);
        }
        case (_) { }
    }
}

fn visit_item(env e, &@ast::item i) {
    alt (i.node) {
        case (ast::item_native_mod(?m)) {
            if (m.abi != ast::native_abi_rust &&
                m.abi != ast::native_abi_cdecl) {
                ret;
            }
            auto cstore = e.sess.get_cstore();
            if (!cstore::add_used_library(cstore, m.native_name)) {
                ret;
            }
            for (ast::attribute a in
                     attr::find_attrs_by_name(i.attrs, "link_args")) {
                alt (attr::get_meta_item_value_str(attr::attr_meta(a))) {
                    case (some(?linkarg)) {
                        cstore::add_used_link_args(cstore, linkarg);
                    }
                    case (none) { /* fallthrough */ }
                }
            }
        }
        case (_) {
        }
    }
}

// A diagnostic function for dumping crate metadata to an output stream
fn list_file_metadata(str path, ioivec::writer out) {
    alt (get_metadata_section(path)) {
        case (option::some(?bytes)) {
            decoder::list_crate_metadata(bytes, out);
        }
        case (option::none) {
            out.write_str("Could not find metadata in " + path + ".\n");
        }
    }
}

fn metadata_matches(&@u8[] crate_data, &(@ast::meta_item)[] metas) -> bool {
    auto attrs = decoder::get_crate_attributes(crate_data);
    auto linkage_metas = attr::find_linkage_metas(attrs);

    log #fmt("matching %u metadata requirements against %u items",
             ivec::len(metas), ivec::len(linkage_metas));

    for (@ast::meta_item needed in metas) {
        if (!attr::contains(linkage_metas, needed)) {
            log #fmt("missing %s", pprust::meta_item_to_str(*needed));
            ret false;
        }
    }
    ret true;
}

fn default_native_lib_naming(session::session sess, bool static) ->
   rec(str prefix, str suffix) {
    if (static) {
        ret rec(prefix="lib", suffix=".rlib");
    }
    alt (sess.get_targ_cfg().os) {
        case (session::os_win32) { ret rec(prefix="", suffix=".dll"); }
        case (session::os_macos) { ret rec(prefix="lib", suffix=".dylib"); }
        case (session::os_linux) { ret rec(prefix="lib", suffix=".so"); }
    }
}

fn find_library_crate(&session::session sess, &ast::ident ident,
                      &(@ast::meta_item)[] metas,
                      &vec[str] library_search_paths)
        -> option::t[tup(str, @u8[])] {

    attr::require_unique_names(sess, metas);

    auto crate_name = {
        auto name_items = attr::find_meta_items_by_name(metas, "name");
        alt (ivec::last(name_items)) {
            case (some(?i)) {
                alt (attr::get_meta_item_value_str(i)) {
                    case (some(?n)) { n }
                    case (_) {
                        // FIXME: Probably want a warning here since the user
                        // is using the wrong type of meta item
                        ident
                    }
                }
            }
            case (none) { ident }
        }
    };

    auto nn = default_native_lib_naming(sess, sess.get_opts().static);
    auto x = find_library_crate_aux(nn, crate_name, metas,
                                    library_search_paths);
    if (x != none || sess.get_opts().static) {
        ret x;
    }
    auto nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(nn2, crate_name, metas, library_search_paths);
}

fn find_library_crate_aux(&rec(str prefix, str suffix) nn, str crate_name,
                          &(@ast::meta_item)[] metas,
                          &vec[str] library_search_paths) ->
                          option::t[tup(str, @u8[])] {
    let str prefix = nn.prefix + crate_name;
    // FIXME: we could probably use a 'glob' function in std::fs but it will
    // be much easier to write once the unsafe module knows more about FFI
    // tricks. Currently the glob(3) interface is a bit more than we can
    // stomach from here, and writing a C++ wrapper is more work than just
    // manually filtering fs::list_dir here.

    for (str library_search_path in library_search_paths) {
        log #fmt("searching %s", library_search_path);
        for (str path in fs::list_dir(library_search_path)) {
            log #fmt("searching %s", path);
            let str f = fs::basename(path);
            if (!(str::starts_with(f, prefix) &&
                      str::ends_with(f, nn.suffix))) {
                log #fmt("skipping %s, doesn't look like %s*%s", path, prefix,
                         nn.suffix);
                cont;
            }
            alt (get_metadata_section(path)) {
                case (option::some(?cvec)) {
                    if (!metadata_matches(cvec, metas)) {
                        log #fmt("skipping %s, metadata doesn't match", path);
                        cont;
                    }
                    log #fmt("found %s with matching metadata", path);
                    ret some(tup(path, cvec));
                }
                case (_) { }
            }
        }
    }
    ret none;
}

fn get_metadata_section(str filename) -> option::t[@u8[]] {
    auto b = str::buf(filename);
    auto mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(b);
    if (mb as int == 0) { ret option::none[@u8[]]; }
    auto of = mk_object_file(mb);
    auto si = mk_section_iter(of.llof);
    while (llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False) {
        auto name_buf = llvm::LLVMGetSectionName(si.llsi);
        auto name = str::str_from_cstr(name_buf);
        if (str::eq(name, x86::get_meta_sect_name())) {
            auto cbuf = llvm::LLVMGetSectionContents(si.llsi);
            auto csz = llvm::LLVMGetSectionSize(si.llsi);
            let *u8 cvbuf = std::unsafe::reinterpret_cast(cbuf);
            ret option::some[@u8[]](@ivec::unsafe::from_buf(cvbuf, csz));
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none[@u8[]];
}

fn load_library_crate(&session::session sess, span span,
                      &ast::ident ident, &(@ast::meta_item)[] metas,
                      &vec[str] library_search_paths) -> tup(str, @u8[]) {

    alt (find_library_crate(sess, ident, metas, library_search_paths)) {
        case (some(?t)) {
            ret t;
        }
        case (none) {
            sess.span_fatal(span, #fmt("can't find crate for '%s'", ident));
        }
    }
}

fn resolve_crate(env e, ast::ident ident, (@ast::meta_item)[] metas,
                 span span) -> ast::crate_num {
    if (!e.crate_cache.contains_key(ident)) {
        auto cinfo = load_library_crate(e.sess, span, ident, metas,
                                        e.library_search_paths);

        auto cfilename = cinfo._0;
        auto cdata = cinfo._1;

        // Claim this crate number and cache it
        auto cnum = e.next_crate_num;
        e.crate_cache.insert(ident, cnum);
        e.next_crate_num += 1;

        // Now resolve the crates referenced by this crate
        auto cnum_map = resolve_crate_deps(e, cdata);

        auto cmeta = rec(name=ident,
                         data=cdata,
                         cnum_map=cnum_map);

        auto cstore = e.sess.get_cstore();
        cstore::set_crate_data(cstore, cnum, cmeta);
        cstore::add_used_crate_file(cstore, cfilename);
        ret cnum;
    } else {
        ret e.crate_cache.get(ident);
    }
}

// Go through the crate metadata and load any crates that it references
fn resolve_crate_deps(env e, &@u8[] cdata) -> cstore::cnum_map {
    log "resolving deps of external crate";
    // The map from crate numbers in the crate we're resolving to local crate
    // numbers
    auto cnum_map = new_int_hash[ast::crate_num]();
    for (decoder::crate_dep dep in decoder::get_crate_deps(cdata)) {
        auto extrn_cnum = dep._0;
        auto cname = dep._1;
        log #fmt("resolving dep %s", cname);
        if (e.crate_cache.contains_key(cname)) {
            log "already have it";
            // We've already seen this crate
            auto local_cnum = e.crate_cache.get(cname);
            cnum_map.insert(extrn_cnum, local_cnum);
        } else {
            log "need to load it";
            // This is a new one so we've got to load it
            // FIXME: Need better error reporting than just a bogus span
            auto fake_span = rec(lo=0u,hi=0u);
            auto local_cnum = resolve_crate(e, cname, ~[], fake_span);
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
