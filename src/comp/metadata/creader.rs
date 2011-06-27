

// -*- rust -*-
import driver::session;
import front::ast;
import lib::llvm::False;
import lib::llvm::llvm;
import lib::llvm::mk_object_file;
import lib::llvm::mk_section_iter;
import middle::resolve;
import middle::walk;
import cwriter;
import middle::trans;
import middle::ty;
import back::x86;
import util::common;
import std::str;
import std::uint;
import std::vec;
import std::ebml;
import std::fs;
import std::io;
import std::option;
import std::option::none;
import std::option::some;
import std::os;
import std::map::hashmap;
import defs::*;
import tydecode::parse_def_id;
import tydecode::parse_ty_data;

export get_symbol;
export get_tag_variants;
export get_type;
export read_crates;
export lookup_defs;
export get_type;
export list_file_metadata;


fn lookup_hash(&ebml::doc d, fn(vec[u8]) -> bool  eq_fn, uint hash) ->
   vec[ebml::doc] {
    auto index = ebml::get_doc(d, tag_index);
    auto table = ebml::get_doc(index, tag_index_table);
    auto hash_pos = table.start + hash % 256u * 4u;
    auto pos = ebml::be_uint_from_bytes(d.data, hash_pos, 4u);
    auto bucket = ebml::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet

    let vec[ebml::doc] result = [];
    auto belt = tag_index_buckets_bucket_elt;
    for each (ebml::doc elt in ebml::tagged_docs(bucket, belt)) {
        auto pos = ebml::be_uint_from_bytes(elt.data, elt.start, 4u);
        if (eq_fn(vec::slice[u8](elt.data, elt.start + 4u, elt.end))) {
            vec::push(result, ebml::doc_at(d.data, pos));
        }
    }
    ret result;
}


// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
fn resolve_path(vec[ast::ident] path, vec[u8] data) -> vec[ast::def_id] {
    fn eq_item(vec[u8] data, str s) -> bool {
        ret str::eq(str::unsafe_from_bytes(data), s);
    }
    auto s = str::connect(path, "::");
    auto md = ebml::new_doc(data);
    auto paths = ebml::get_doc(md, tag_paths);
    auto eqer = bind eq_item(_, s);
    let vec[ast::def_id] result = [];
    for (ebml::doc doc in lookup_hash(paths, eqer, encoder::hash_path(s))) {
        auto did_doc = ebml::get_doc(doc, tag_def_id);
        vec::push(result, parse_def_id(ebml::doc_data(did_doc)));
    }
    ret result;
}

fn maybe_find_item(int item_id, &ebml::doc items) -> option::t[ebml::doc] {
    fn eq_item(vec[u8] bytes, int item_id) -> bool {
        ret ebml::be_uint_from_bytes(bytes, 0u, 4u) as int == item_id;
    }
    auto eqer = bind eq_item(_, item_id);
    auto found = lookup_hash(items, eqer, encoder::hash_def_id(item_id));
    if (vec::len(found) == 0u) {
        ret option::none[ebml::doc];
    } else { ret option::some[ebml::doc](found.(0)); }
}

fn find_item(int item_id, &ebml::doc items) -> ebml::doc {
    ret option::get(maybe_find_item(item_id, items));
}


// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item(int item_id, vec[u8] data) -> ebml::doc {
    auto items = ebml::get_doc(ebml::new_doc(data), tag_items);
    ret find_item(item_id, items);
}

fn item_kind(&ebml::doc item) -> u8 {
    auto kind = ebml::get_doc(item, tag_items_data_item_kind);
    ret ebml::doc_as_uint(kind) as u8;
}

fn item_symbol(&ebml::doc item) -> str {
    auto sym = ebml::get_doc(item, tag_items_data_item_symbol);
    ret str::unsafe_from_bytes(ebml::doc_data(sym));
}

fn variant_tag_id(&ebml::doc d) -> ast::def_id {
    auto tagdoc = ebml::get_doc(d, tag_items_data_item_tag_id);
    ret parse_def_id(ebml::doc_data(tagdoc));
}

fn item_type(&ebml::doc item, int this_cnum, ty::ctxt tcx) -> ty::t {
    fn parse_external_def_id(int this_cnum, str s) -> ast::def_id {
        // FIXME: This is completely wrong when linking against a crate
        // that, in turn, links against another crate. We need a mapping
        // from crate ID to crate "meta" attributes as part of the crate
        // metadata:

        auto buf = str::bytes(s);
        auto external_def_id = parse_def_id(buf);
        ret tup(this_cnum, external_def_id._1);
    }
    auto tp = ebml::get_doc(item, tag_items_data_item_type);
    auto s = str::unsafe_from_bytes(ebml::doc_data(tp));
    ret parse_ty_data(item.data, this_cnum, tp.start, tp.end - tp.start,
                      bind parse_external_def_id(this_cnum, _), tcx);
}

fn item_ty_param_count(&ebml::doc item, int this_cnum) -> uint {
    let uint ty_param_count = 0u;
    auto tp = tag_items_data_item_ty_param_count;
    for each (ebml::doc p in ebml::tagged_docs(item, tp)) {
        ty_param_count = ebml::vint_at(ebml::doc_data(p), 0u)._0;
    }
    ret ty_param_count;
}

fn tag_variant_ids(&ebml::doc item, int this_cnum) -> vec[ast::def_id] {
    let vec[ast::def_id] ids = [];
    auto v = tag_items_data_item_variant;
    for each (ebml::doc p in ebml::tagged_docs(item, v)) {
        auto ext = parse_def_id(ebml::doc_data(p));
        vec::push[ast::def_id](ids, tup(this_cnum, ext._1));
    }
    ret ids;
}

fn get_exported_metadata(&session::session sess, &str path, &vec[u8] data) ->
   hashmap[str, str] {
    auto meta_items =
        ebml::get_doc(ebml::new_doc(data), tag_meta_export);
    auto mm = common::new_str_hash[str]();
    for each (ebml::doc m in
             ebml::tagged_docs(meta_items, tag_meta_item)) {
        auto kd = ebml::get_doc(m, tag_meta_item_key);
        auto vd = ebml::get_doc(m, tag_meta_item_value);
        auto k = str::unsafe_from_bytes(ebml::doc_data(kd));
        auto v = str::unsafe_from_bytes(ebml::doc_data(vd));
        log #fmt("metadata in %s: %s = %s", path, k, v);
        if (!mm.insert(k, v)) {
            sess.warn(#fmt("Duplicate metadata item in %s: %s", path, k));
        }
    }
    ret mm;
}

fn metadata_matches(hashmap[str, str] mm, &vec[@ast::meta_item] metas) ->
   bool {
    log #fmt("matching %u metadata requirements against %u metadata items",
             vec::len(metas), mm.size());
    for (@ast::meta_item mi in metas) {
        alt (mi.node) {
            case (ast::meta_key_value(?key, ?value)) {
                alt (mm.find(key)) {
                    case (some(?v)) {
                        if (v == value) {
                            log #fmt("matched '%s': '%s'", key,
                                     value);
                        } else {
                            log #fmt("missing '%s': '%s' (got '%s')",
                                     key,
                                     value, v);
                            ret false;
                        }
                    }
                    case (none) {
                        log #fmt("missing '%s': '%s'",
                                 key, value);
                        ret false;
                    }
                }
            }
            case (_) {
                // FIXME (#487): Support all forms of meta_item
                log_err "unimplemented meta_item variant in metadata_matches";
                ret false;
            }
        }
    }
    ret true;
}

fn default_native_lib_naming(session::session sess) ->
   rec(str prefix, str suffix) {
    alt (sess.get_targ_cfg().os) {
        case (session::os_win32) { ret rec(prefix="", suffix=".dll"); }
        case (session::os_macos) { ret rec(prefix="lib", suffix=".dylib"); }
        case (session::os_linux) { ret rec(prefix="lib", suffix=".so"); }
    }
}

fn find_library_crate(&session::session sess, &ast::ident ident,
                      &vec[@ast::meta_item] metas,
                      &vec[str] library_search_paths) ->
   option::t[tup(str, vec[u8])] {
    let str crate_name = ident;
    for (@ast::meta_item mi in metas) {
        alt (mi.node) {
            case (ast::meta_key_value(?key, ?value)) {
                if (key == "name") {
                    crate_name = value;
                    break;
                }
            }
            case (_) {
                // FIXME (#487)
                sess.unimpl("meta_item variant")
            }
        }
    }
    auto nn = default_native_lib_naming(sess);
    let str prefix = nn.prefix + crate_name;
    // FIXME: we could probably use a 'glob' function in std::fs but it will
    // be much easier to write once the unsafe module knows more about FFI
    // tricks. Currently the glob(3) interface is a bit more than we can
    // stomach from here, and writing a C++ wrapper is more work than just
    // manually filtering fs::list_dir here.

    for (str library_search_path in library_search_paths) {
        for (str path in fs::list_dir(library_search_path)) {
            let str f = fs::basename(path);
            if (!(str::starts_with(f, prefix) &&
                      str::ends_with(f, nn.suffix))) {
                log #fmt("skipping %s, doesn't look like %s*%s", path, prefix,
                         nn.suffix);
                cont;
            }
            alt (get_metadata_section(path)) {
                case (option::some(?cvec)) {
                    auto mm = get_exported_metadata(sess, path, cvec);
                    if (!metadata_matches(mm, metas)) {
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

fn get_metadata_section(str filename) -> option::t[vec[u8]] {
    auto b = str::buf(filename);
    auto mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(b);
    if (mb as int == 0) { ret option::none[vec[u8]]; }
    auto of = mk_object_file(mb);
    auto si = mk_section_iter(of.llof);
    while (llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False) {
        auto name_buf = llvm::LLVMGetSectionName(si.llsi);
        auto name = str::str_from_cstr(name_buf);
        if (str::eq(name, x86::get_meta_sect_name())) {
            auto cbuf = llvm::LLVMGetSectionContents(si.llsi);
            auto csz = llvm::LLVMGetSectionSize(si.llsi);
            auto cvbuf = cbuf as vec::vbuf;
            ret option::some[vec[u8]](vec::vec_from_vbuf[u8](cvbuf, csz));
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none[vec[u8]];
}

fn load_library_crate(&session::session sess, int cnum, &ast::ident ident,
                      &vec[@ast::meta_item] metas,
                      &vec[str] library_search_paths) {
    alt (find_library_crate(sess, ident, metas, library_search_paths)) {
        case (some(?t)) {
            sess.set_external_crate(cnum, rec(name=ident, data=t._1));
            sess.add_used_crate_file(t._0);
            ret;
        }
        case (_) { }
    }
    log_err #fmt("can't find crate for '%s'", ident);
    fail;
}

type env =
    @rec(session::session sess,
         resolve::crate_map crate_map,
         @hashmap[str, int] crate_cache,
         vec[str] library_search_paths,
         mutable int next_crate_num);

fn visit_view_item(env e, &@ast::view_item i) {
    alt (i.node) {
        case (ast::view_item_use(?ident, ?meta_items, ?id)) {
            auto cnum;
            if (!e.crate_cache.contains_key(ident)) {
                cnum = e.next_crate_num;
                load_library_crate(e.sess, cnum, ident, meta_items,
                                   e.library_search_paths);
                e.crate_cache.insert(ident, e.next_crate_num);
                e.next_crate_num += 1;
            } else { cnum = e.crate_cache.get(ident); }
            e.crate_map.insert(id, cnum);
        }
        case (_) { }
    }
}

fn visit_item(env e, &@ast::item i) {
    alt (i.node) {
        case (ast::item_native_mod(?m)) {
            auto name;
            if (m.native_name == "" ) {
                name = i.ident;
            } else {
                name = m.native_name;
            }
            alt (m.abi) {
                case (ast::native_abi_rust) {
                    e.sess.add_used_library(name);
                }
                case (ast::native_abi_cdecl) {
                    e.sess.add_used_library(name);
                }
                case (ast::native_abi_llvm) {
                }
                case (ast::native_abi_rust_intrinsic) {
                }
            }
        }
        case (_) {
        }
    }
}

// Reads external crates referenced by "use" directives.
fn read_crates(session::session sess, resolve::crate_map crate_map,
               &ast::crate crate) {
    auto e =
        @rec(sess=sess,
             crate_map=crate_map,
             crate_cache=@common::new_str_hash[int](),
             library_search_paths=sess.get_opts().library_search_paths,
             mutable next_crate_num=1);
    auto v =
        rec(visit_view_item_pre=bind visit_view_item(e, _),
            visit_item_pre=bind visit_item(e, _)
            with walk::default_visitor());
    walk::walk_crate(v, crate);
}

fn kind_has_type_params(u8 kind_ch) -> bool {
    ret alt (kind_ch as char) {
            case ('c') { false }
            case ('f') { true }
            case ('p') { true }
            case ('F') { true }
            case ('y') { true }
            case ('t') { true }
            case ('T') { false }
            case ('m') { false }
            case ('n') { false }
            case ('v') { true }
        };
}


// Crate metadata queries
fn lookup_defs(session::session sess, int cnum, vec[ast::ident] path) ->
   vec[ast::def] {
    auto data = sess.get_external_crate(cnum).data;
    ret vec::map(bind lookup_def(cnum, data, _), resolve_path(path, data));
}


// FIXME doesn't yet handle re-exported externals
fn lookup_def(int cnum, vec[u8] data, &ast::def_id did_) -> ast::def {
    auto item = lookup_item(did_._1, data);
    auto kind_ch = item_kind(item);
    auto did = tup(cnum, did_._1);
    auto def =
        alt (kind_ch as char) {
            case ('c') { ast::def_const(did) }
            case ('f') { ast::def_fn(did, ast::impure_fn) }
            case ('p') { ast::def_fn(did, ast::pure_fn) }
            case ('F') { ast::def_native_fn(did) }
            case ('y') { ast::def_ty(did) }
            case ('T') { ast::def_native_ty(did) }
            // We treat references to tags as references to types.
            case ('t') { ast::def_ty(did) }
            case ('m') { ast::def_mod(did) }
            case ('n') { ast::def_native_mod(did) }
            case ('v') {
                auto tid = variant_tag_id(item);
                tid = tup(cnum, tid._1);
                ast::def_variant(tid, did)
            }
        };
    ret def;
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    auto external_crate_id = def._0;
    auto data = tcx.sess.get_external_crate(external_crate_id).data;
    auto item = lookup_item(def._1, data);
    auto t = item_type(item, external_crate_id, tcx);
    auto tp_count;
    auto kind_ch = item_kind(item);
    auto has_ty_params = kind_has_type_params(kind_ch);
    if (has_ty_params) {
        tp_count = item_ty_param_count(item, external_crate_id);
    } else { tp_count = 0u; }
    ret tup(tp_count, t);
}

fn get_symbol(session::session sess, ast::def_id def) -> str {
    auto external_crate_id = def._0;
    auto data = sess.get_external_crate(external_crate_id).data;
    ret item_symbol(lookup_item(def._1, data));
}

fn get_tag_variants(ty::ctxt tcx, ast::def_id def) -> vec[ty::variant_info] {
    auto external_crate_id = def._0;
    auto data = tcx.sess.get_external_crate(external_crate_id).data;
    auto items = ebml::get_doc(ebml::new_doc(data), tag_items);
    auto item = find_item(def._1, items);
    let vec[ty::variant_info] infos = [];
    auto variant_ids = tag_variant_ids(item, external_crate_id);
    for (ast::def_id did in variant_ids) {
        auto item = find_item(did._1, items);
        auto ctor_ty = item_type(item, external_crate_id, tcx);
        let vec[ty::t] arg_tys = [];
        alt (ty::struct(tcx, ctor_ty)) {
            case (ty::ty_fn(_, ?args, _, _, _)) {
                for (ty::arg a in args) { arg_tys += [a.ty]; }
            }
            case (_) {
                // Nullary tag variant.

            }
        }
        infos += [rec(args=arg_tys, ctor_ty=ctor_ty, id=did)];
    }
    ret infos;
}

fn list_file_metadata(str path, io::writer out) {
    alt (get_metadata_section(path)) {
        case (option::some(?bytes)) { list_crate_metadata(bytes, out); }
        case (option::none) {
            out.write_str("Could not find metadata in " + path + ".\n");
        }
    }
}

fn read_path(&ebml::doc d) -> tup(str, uint) {
    auto desc = ebml::doc_data(d);
    auto pos = ebml::be_uint_from_bytes(desc, 0u, 4u);
    auto pathbytes = vec::slice[u8](desc, 4u, vec::len[u8](desc));
    auto path = str::unsafe_from_bytes(pathbytes);
    ret tup(path, pos);
}

fn list_crate_metadata(vec[u8] bytes, io::writer out) {
    auto md = ebml::new_doc(bytes);
    auto paths = ebml::get_doc(md, tag_paths);
    auto items = ebml::get_doc(md, tag_items);
    auto index = ebml::get_doc(paths, tag_index);
    auto bs = ebml::get_doc(index, tag_index_buckets);
    for each (ebml::doc bucket in
             ebml::tagged_docs(bs, tag_index_buckets_bucket)) {
        auto et = tag_index_buckets_bucket_elt;
        for each (ebml::doc elt in ebml::tagged_docs(bucket, et)) {
            auto data = read_path(elt);
            auto def = ebml::doc_at(bytes, data._1);
            auto did_doc = ebml::get_doc(def, tag_def_id);
            auto did = parse_def_id(ebml::doc_data(did_doc));
            out.write_str(#fmt("%s (%s)\n", data._0,
                               describe_def(items, did)));
        }
    }
}

fn describe_def(&ebml::doc items, ast::def_id id) -> str {
    if (id._0 != 0) { ret "external"; }
    ret item_kind_to_str(item_kind(find_item(id._1, items)));
}

fn item_kind_to_str(u8 kind) -> str {
    alt (kind as char) {
        case ('c') { ret "const"; }
        case ('f') { ret "fn"; }
        case ('p') { ret "pred"; }
        case ('F') { ret "native fn"; }
        case ('y') { ret "type"; }
        case ('T') { ret "native type"; }
        case ('t') { ret "type"; }
        case ('m') { ret "mod"; }
        case ('n') { ret "native mod"; }
        case ('v') { ret "tag"; }
    }
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
