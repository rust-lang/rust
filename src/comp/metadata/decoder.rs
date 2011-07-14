// Decoding metadata from a single crate's metadata

import std::ebmlivec;
import std::ivec;
import std::option;
import std::vec;
import std::str;
import std::ioivec;
import std::map::hashmap;
import syntax::ast;
import front::attr;
import middle::ty;
import common::*;
import tydecode::parse_def_id;
import tydecode::parse_ty_data;
import driver::session;
import syntax::print::pprust;
import cstore;

export get_symbol;
export get_tag_variants;
export get_type;
export get_type_param_count;
export lookup_defs;
export get_crate_attributes;
export list_crate_metadata;
export crate_dep;
export get_crate_deps;
export external_resolver;

// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.
type external_resolver = fn(&ast::def_id def_id) -> ast::def_id;

fn lookup_hash(&ebmlivec::doc d, fn(&u8[]) -> bool  eq_fn, uint hash) ->
   (ebmlivec::doc)[] {
    auto index = ebmlivec::get_doc(d, tag_index);
    auto table = ebmlivec::get_doc(index, tag_index_table);
    auto hash_pos = table.start + hash % 256u * 4u;
    auto pos = ebmlivec::be_uint_from_bytes(d.data, hash_pos, 4u);
    auto bucket = ebmlivec::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet

    let (ebmlivec::doc)[] result = ~[];
    auto belt = tag_index_buckets_bucket_elt;
    for each (ebmlivec::doc elt in ebmlivec::tagged_docs(bucket, belt)) {
        auto pos = ebmlivec::be_uint_from_bytes(elt.data, elt.start, 4u);
        if (eq_fn(ivec::slice[u8](*elt.data, elt.start + 4u, elt.end))) {
            result += ~[ebmlivec::doc_at(d.data, pos)];
        }
    }
    ret result;
}

fn maybe_find_item(int item_id, &ebmlivec::doc items)
    -> option::t[ebmlivec::doc] {
    fn eq_item(&u8[] bytes, int item_id) -> bool {
        ret ebmlivec::be_uint_from_bytes(@bytes, 0u, 4u) as int == item_id;
    }
    auto eqer = bind eq_item(_, item_id);
    auto found = lookup_hash(items, eqer, hash_node_id(item_id));
    if (ivec::len(found) == 0u) {
        ret option::none[ebmlivec::doc];
    } else { ret option::some[ebmlivec::doc](found.(0)); }
}

fn find_item(int item_id, &ebmlivec::doc items) -> ebmlivec::doc {
    ret option::get(maybe_find_item(item_id, items));
}

// Looks up an item in the given metadata and returns an ebmlivec doc pointing
// to the item data.
fn lookup_item(int item_id, &@u8[] data) -> ebmlivec::doc {
    auto items = ebmlivec::get_doc(ebmlivec::new_doc(data), tag_items);
    ret find_item(item_id, items);
}

fn item_kind(&ebmlivec::doc item) -> u8 {
    auto kind = ebmlivec::get_doc(item, tag_items_data_item_kind);
    ret ebmlivec::doc_as_uint(kind) as u8;
}

fn item_symbol(&ebmlivec::doc item) -> str {
    auto sym = ebmlivec::get_doc(item, tag_items_data_item_symbol);
    ret str::unsafe_from_bytes_ivec(ebmlivec::doc_data(sym));
}

fn variant_tag_id(&ebmlivec::doc d) -> ast::def_id {
    auto tagdoc = ebmlivec::get_doc(d, tag_items_data_item_tag_id);
    ret parse_def_id(ebmlivec::doc_data(tagdoc));
}

fn item_type(&ebmlivec::doc item, ast::crate_num this_cnum,
             ty::ctxt tcx, &external_resolver extres) -> ty::t {
    fn parse_external_def_id(ast::crate_num this_cnum,
                             &external_resolver extres,
                             str s) -> ast::def_id {
        auto buf = str::bytes_ivec(s);
        auto external_def_id = parse_def_id(buf);

        // This item was defined in the crate we're searching if it's has the
        // local crate number, otherwise we need to search a different crate
        if (external_def_id._0 == ast::local_crate) {
            ret tup(this_cnum, external_def_id._1);
        } else {
            ret extres(external_def_id);
        }
    }
    auto tp = ebmlivec::get_doc(item, tag_items_data_item_type);
    auto def_parser = bind parse_external_def_id(this_cnum, extres, _);
    ret parse_ty_data(item.data, this_cnum, tp.start, tp.end - tp.start,
                      def_parser, tcx);
}

fn item_ty_param_count(&ebmlivec::doc item) -> uint {
    let uint ty_param_count = 0u;
    auto tp = tag_items_data_item_ty_param_count;
    for each (ebmlivec::doc p in ebmlivec::tagged_docs(item, tp)) {
        ty_param_count = ebmlivec::vint_at(ebmlivec::doc_data(p), 0u)._0;
    }
    ret ty_param_count;
}

fn tag_variant_ids(&ebmlivec::doc item,
                   ast::crate_num this_cnum) -> vec[ast::def_id] {
    let vec[ast::def_id] ids = [];
    auto v = tag_items_data_item_variant;
    for each (ebmlivec::doc p in ebmlivec::tagged_docs(item, v)) {
        auto ext = parse_def_id(ebmlivec::doc_data(p));
        vec::push[ast::def_id](ids, tup(this_cnum, ext._1));
    }
    ret ids;
}

// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
fn resolve_path(vec[ast::ident] path, @u8[] data) -> vec[ast::def_id] {
    fn eq_item(&u8[] data, str s) -> bool {
        ret str::eq(str::unsafe_from_bytes_ivec(data), s);
    }
    auto s = str::connect(path, "::");
    auto md = ebmlivec::new_doc(data);
    auto paths = ebmlivec::get_doc(md, tag_paths);
    auto eqer = bind eq_item(_, s);
    let vec[ast::def_id] result = [];
    for (ebmlivec::doc doc in lookup_hash(paths, eqer, hash_path(s))) {
        auto did_doc = ebmlivec::get_doc(doc, tag_def_id);
        vec::push(result, parse_def_id(ebmlivec::doc_data(did_doc)));
    }
    ret result;
}

// Crate metadata queries
fn lookup_defs(&@u8[] data, ast::crate_num cnum,
               vec[ast::ident] path) -> vec[ast::def] {
    ret vec::map(bind lookup_def(cnum, data, _), resolve_path(path, data));
}


// FIXME doesn't yet handle re-exported externals
fn lookup_def(ast::crate_num cnum, @u8[] data, &ast::def_id did_)
        -> ast::def {
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

fn get_type(@u8[] data, ast::def_id def, &ty::ctxt tcx,
            &external_resolver extres) -> ty::ty_param_count_and_ty {
    auto this_cnum = def._0;
    auto node_id = def._1;
    auto item = lookup_item(node_id, data);
    auto t = item_type(item, this_cnum, tcx, extres);
    auto tp_count;
    auto kind_ch = item_kind(item);
    auto has_ty_params = kind_has_type_params(kind_ch);
    if (has_ty_params) {
        tp_count = item_ty_param_count(item);
    } else { tp_count = 0u; }
    ret tup(tp_count, t);
}

fn get_type_param_count(@u8[] data, ast::node_id id) -> uint {
    ret item_ty_param_count(lookup_item(id, data));
}

fn get_symbol(@u8[] data, ast::node_id id) -> str {
    ret item_symbol(lookup_item(id, data));
}

fn get_tag_variants(&@u8[] data, ast::def_id def,
                    &ty::ctxt tcx,
                    &external_resolver extres) -> ty::variant_info[] {
    auto external_crate_id = def._0;
    auto data = cstore::get_crate_data(tcx.sess.get_cstore(),
                                       external_crate_id).data;
    auto items = ebmlivec::get_doc(ebmlivec::new_doc(data), tag_items);
    auto item = find_item(def._1, items);
    let ty::variant_info[] infos = ~[];
    auto variant_ids = tag_variant_ids(item, external_crate_id);
    for (ast::def_id did in variant_ids) {
        auto item = find_item(did._1, items);
        auto ctor_ty = item_type(item, external_crate_id, tcx, extres);
        let ty::t[] arg_tys = ~[];
        alt (ty::struct(tcx, ctor_ty)) {
            case (ty::ty_fn(_, ?args, _, _, _)) {
                for (ty::arg a in args) { arg_tys += ~[a.ty]; }
            }
            case (_) {
                // Nullary tag variant.

            }
        }
        infos += ~[rec(args=arg_tys, ctor_ty=ctor_ty, id=did)];
    }
    ret infos;
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

fn read_path(&ebmlivec::doc d) -> tup(str, uint) {
    auto desc = ebmlivec::doc_data(d);
    auto pos = ebmlivec::be_uint_from_bytes(@desc, 0u, 4u);
    auto pathbytes = ivec::slice[u8](desc, 4u, ivec::len[u8](desc));
    auto path = str::unsafe_from_bytes_ivec(pathbytes);
    ret tup(path, pos);
}

fn describe_def(&ebmlivec::doc items, ast::def_id id) -> str {
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

fn get_meta_items(&ebmlivec::doc md) -> (@ast::meta_item)[] {
    let (@ast::meta_item)[] items = ~[];
    for each (ebmlivec::doc meta_item_doc in
              ebmlivec::tagged_docs(md, tag_meta_item_word)) {
        auto nd = ebmlivec::get_doc(meta_item_doc, tag_meta_item_name);
        auto n = str::unsafe_from_bytes_ivec(ebmlivec::doc_data(nd));
        items += ~[attr::mk_word_item(n)];
    }
    for each (ebmlivec::doc meta_item_doc in
              ebmlivec::tagged_docs(md, tag_meta_item_name_value)) {
        auto nd = ebmlivec::get_doc(meta_item_doc, tag_meta_item_name);
        auto vd = ebmlivec::get_doc(meta_item_doc, tag_meta_item_value);
        auto n = str::unsafe_from_bytes_ivec(ebmlivec::doc_data(nd));
        auto v = str::unsafe_from_bytes_ivec(ebmlivec::doc_data(vd));
        // FIXME (#611): Should be able to decode meta_name_value variants,
        // but currently they can't be encoded
        items += ~[attr::mk_name_value_item_str(n, v)];
    }
    for each (ebmlivec::doc meta_item_doc in
              ebmlivec::tagged_docs(md, tag_meta_item_list)) {
        auto nd = ebmlivec::get_doc(meta_item_doc, tag_meta_item_name);
        auto n = str::unsafe_from_bytes_ivec(ebmlivec::doc_data(nd));
        auto subitems = get_meta_items(meta_item_doc);
        items += ~[attr::mk_list_item(n, subitems)];
    }
    ret items;
}

fn get_attributes(&ebmlivec::doc md) -> ast::attribute[] {
    let ast::attribute[] attrs = ~[];
    alt (ebmlivec::maybe_get_doc(md, tag_attributes)) {
        case (option::some(?attrs_d)) {
            for each (ebmlivec::doc attr_doc in
                      ebmlivec::tagged_docs(attrs_d, tag_attribute)) {
                auto meta_items = get_meta_items(attr_doc);
                // Currently it's only possible to have a single meta item on
                // an attribute
                assert (ivec::len(meta_items) == 1u);
                auto meta_item = meta_items.(0);
                attrs += ~[rec(node=rec(style=ast::attr_outer,
                                        value=*meta_item),
                               span=rec(lo=0u, hi=0u))];
            }
        }
        case (option::none) { }
    }
    ret attrs;
}

fn list_meta_items(&ebmlivec::doc meta_items, ioivec::writer out) {
    for (@ast::meta_item mi in get_meta_items(meta_items)) {
        out.write_str(#fmt("%s\n", pprust::meta_item_to_str(*mi)));
    }
}

fn list_crate_attributes(&ebmlivec::doc md, ioivec::writer out) {
    out.write_str("=Crate Attributes=\n");

    for (ast::attribute attr in get_attributes(md)) {
        out.write_str(#fmt("%s\n", pprust::attribute_to_str(attr)));
    }

    out.write_str("\n\n");
}

fn get_crate_attributes(@u8[] data) -> ast::attribute[] {
    ret get_attributes(ebmlivec::new_doc(data));
}

type crate_dep = tup(ast::crate_num, str);

fn get_crate_deps(@u8[] data) -> vec[crate_dep] {
    let vec[crate_dep] deps = [];
    auto cratedoc = ebmlivec::new_doc(data);
    auto depsdoc = ebmlivec::get_doc(cratedoc, tag_crate_deps);
    auto crate_num = 1;
    for each (ebmlivec::doc depdoc in
              ebmlivec::tagged_docs(depsdoc, tag_crate_dep)) {
        auto depname =
            str::unsafe_from_bytes_ivec(ebmlivec::doc_data(depdoc));
        deps += [tup(crate_num, depname)];
        crate_num += 1;
    }
    ret deps;
}

fn list_crate_deps(@u8[] data, ioivec::writer out) {
    out.write_str("=External Dependencies=\n");

    for (crate_dep dep in get_crate_deps(data)) {
        out.write_str(#fmt("%d %s\n", dep._0, dep._1));
    }

    out.write_str("\n");
}

fn list_crate_items(&@u8[] bytes, &ebmlivec::doc md, ioivec::writer out) {
    out.write_str("=Items=\n");
    auto paths = ebmlivec::get_doc(md, tag_paths);
    auto items = ebmlivec::get_doc(md, tag_items);
    auto index = ebmlivec::get_doc(paths, tag_index);
    auto bs = ebmlivec::get_doc(index, tag_index_buckets);
    for each (ebmlivec::doc bucket in
             ebmlivec::tagged_docs(bs, tag_index_buckets_bucket)) {
        auto et = tag_index_buckets_bucket_elt;
        for each (ebmlivec::doc elt in ebmlivec::tagged_docs(bucket, et)) {
            auto data = read_path(elt);
            auto def = ebmlivec::doc_at(bytes, data._1);
            auto did_doc = ebmlivec::get_doc(def, tag_def_id);
            auto did = parse_def_id(ebmlivec::doc_data(did_doc));
            out.write_str(#fmt("%s (%s)\n", data._0,
                               describe_def(items, did)));
        }
    }
    out.write_str("\n");
}

fn list_crate_metadata(&@u8[] bytes, ioivec::writer out) {
    auto md = ebmlivec::new_doc(bytes);
    list_crate_attributes(md, out);
    list_crate_deps(bytes, out);
    list_crate_items(bytes, md, out);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
