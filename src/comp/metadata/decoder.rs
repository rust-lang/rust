// Decoding metadata from a single crate's metadata

import std::{ebml, vec, option, str, io};
import std::map::hashmap;
import syntax::{ast, ast_util};
import front::attr;
import middle::ty;
import common::*;
import tydecode::{parse_def_id, parse_ty_data};
import driver::session;
import syntax::print::pprust;
import cstore;

export get_symbol;
export get_tag_variants;
export get_type;
export get_type_param_count;
export get_type_param_kinds;
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
type external_resolver = fn(ast::def_id) -> ast::def_id;

fn lookup_hash(d: ebml::doc, eq_fn: fn([u8]) -> bool, hash: uint) ->
   [ebml::doc] {
    let index = ebml::get_doc(d, tag_index);
    let table = ebml::get_doc(index, tag_index_table);
    let hash_pos = table.start + hash % 256u * 4u;
    let pos = ebml::be_uint_from_bytes(d.data, hash_pos, 4u);
    let bucket = ebml::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet

    let result: [ebml::doc] = [];
    let belt = tag_index_buckets_bucket_elt;
    for each elt: ebml::doc in ebml::tagged_docs(bucket, belt) {
        let pos = ebml::be_uint_from_bytes(elt.data, elt.start, 4u);
        if eq_fn(vec::slice::<u8>(*elt.data, elt.start + 4u, elt.end)) {
            result += [ebml::doc_at(d.data, pos)];
        }
    }
    ret result;
}

fn maybe_find_item(item_id: int, items: ebml::doc) -> option::t<ebml::doc> {
    fn eq_item(bytes: [u8], item_id: int) -> bool {
        ret ebml::be_uint_from_bytes(@bytes, 0u, 4u) as int == item_id;
    }
    let eqer = bind eq_item(_, item_id);
    let found = lookup_hash(items, eqer, hash_node_id(item_id));
    if vec::len(found) == 0u {
        ret option::none::<ebml::doc>;
    } else { ret option::some::<ebml::doc>(found[0]); }
}

fn find_item(item_id: int, items: ebml::doc) -> ebml::doc {
    ret option::get(maybe_find_item(item_id, items));
}

// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item(item_id: int, data: @[u8]) -> ebml::doc {
    let items = ebml::get_doc(ebml::new_doc(data), tag_items);
    ret find_item(item_id, items);
}

fn item_family(item: ebml::doc) -> u8 {
    let fam = ebml::get_doc(item, tag_items_data_item_family);
    ret ebml::doc_as_uint(fam) as u8;
}

fn item_symbol(item: ebml::doc) -> str {
    let sym = ebml::get_doc(item, tag_items_data_item_symbol);
    ret str::unsafe_from_bytes(ebml::doc_data(sym));
}

fn variant_tag_id(d: ebml::doc) -> ast::def_id {
    let tagdoc = ebml::get_doc(d, tag_items_data_item_tag_id);
    ret parse_def_id(ebml::doc_data(tagdoc));
}

fn item_type(item: ebml::doc, this_cnum: ast::crate_num, tcx: ty::ctxt,
             extres: external_resolver) -> ty::t {
    fn parse_external_def_id(this_cnum: ast::crate_num,
                             extres: external_resolver, s: str) ->
       ast::def_id {
        let buf = str::bytes(s);
        let external_def_id = parse_def_id(buf);


        // This item was defined in the crate we're searching if it's has the
        // local crate number, otherwise we need to search a different crate
        if external_def_id.crate == ast::local_crate {
            ret {crate: this_cnum, node: external_def_id.node};
        } else { ret extres(external_def_id); }
    }
    let tp = ebml::get_doc(item, tag_items_data_item_type);
    let def_parser = bind parse_external_def_id(this_cnum, extres, _);
    ret parse_ty_data(item.data, this_cnum, tp.start, tp.end - tp.start,
                      def_parser, tcx);
}

fn item_ty_param_kinds(item: ebml::doc) -> [ast::kind] {
    let ks: [ast::kind] = [];
    let tp = tag_items_data_item_ty_param_kinds;
    for each p: ebml::doc in ebml::tagged_docs(item, tp) {
        let dat: [u8] = ebml::doc_data(p);
        let vi = ebml::vint_at(dat, 0u);
        let i = 0u;
        while i < vi.val {
            let k =
                alt dat[vi.next + i] as char {
                  'u' { ast::kind_unique }
                  's' { ast::kind_shared }
                  'p' { ast::kind_pinned }
                };
            ks += [k];
            i += 1u;
        }
    }
    ret ks;
}

fn tag_variant_ids(item: ebml::doc, this_cnum: ast::crate_num) ->
   [ast::def_id] {
    let ids: [ast::def_id] = [];
    let v = tag_items_data_item_variant;
    for each p: ebml::doc in ebml::tagged_docs(item, v) {
        let ext = parse_def_id(ebml::doc_data(p));
        ids += [{crate: this_cnum, node: ext.node}];
    }
    ret ids;
}

// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
fn resolve_path(path: [ast::ident], data: @[u8]) -> [ast::def_id] {
    fn eq_item(data: [u8], s: str) -> bool {
        ret str::eq(str::unsafe_from_bytes(data), s);
    }
    let s = str::connect(path, "::");
    let md = ebml::new_doc(data);
    let paths = ebml::get_doc(md, tag_paths);
    let eqer = bind eq_item(_, s);
    let result: [ast::def_id] = [];
    for doc: ebml::doc in lookup_hash(paths, eqer, hash_path(s)) {
        let did_doc = ebml::get_doc(doc, tag_def_id);
        result += [parse_def_id(ebml::doc_data(did_doc))];
    }
    ret result;
}

// Crate metadata queries
fn lookup_defs(data: @[u8], cnum: ast::crate_num, path: [ast::ident]) ->
   [ast::def] {
    ret vec::map(bind lookup_def(cnum, data, _), resolve_path(path, data));
}


// FIXME doesn't yet handle re-exported externals
fn lookup_def(cnum: ast::crate_num, data: @[u8], did_: ast::def_id) ->
   ast::def {
    let item = lookup_item(did_.node, data);
    let fam_ch = item_family(item);
    let did = {crate: cnum, node: did_.node};
    // We treat references to tags as references to types.
    let def =
        alt fam_ch as char {
          'c' { ast::def_const(did) }
          'u' { ast::def_fn(did, ast::unsafe_fn) }
          'f' { ast::def_fn(did, ast::impure_fn) }
          'p' { ast::def_fn(did, ast::pure_fn) }
          'F' { ast::def_native_fn(did) }
          'y' { ast::def_ty(did) }
          'T' { ast::def_native_ty(did) }
          't' { ast::def_ty(did) }
          'm' { ast::def_mod(did) }
          'n' { ast::def_native_mod(did) }
          'v' {
            let tid = variant_tag_id(item);
            tid = {crate: cnum, node: tid.node};
            ast::def_variant(tid, did)
          }
        };
    ret def;
}

fn get_type(data: @[u8], def: ast::def_id, tcx: ty::ctxt,
            extres: external_resolver) -> ty::ty_param_kinds_and_ty {
    let this_cnum = def.crate;
    let node_id = def.node;
    let item = lookup_item(node_id, data);
    let t = item_type(item, this_cnum, tcx, extres);
    let tp_kinds: [ast::kind];
    let fam_ch = item_family(item);
    let has_ty_params = family_has_type_params(fam_ch);
    if has_ty_params {
        tp_kinds = item_ty_param_kinds(item);
    } else { tp_kinds = []; }
    ret {kinds: tp_kinds, ty: t};
}

fn get_type_param_count(data: @[u8], id: ast::node_id) -> uint {
    ret vec::len(get_type_param_kinds(data, id));
}

fn get_type_param_kinds(data: @[u8], id: ast::node_id) -> [ast::kind] {
    ret item_ty_param_kinds(lookup_item(id, data));
}

fn get_symbol(data: @[u8], id: ast::node_id) -> str {
    ret item_symbol(lookup_item(id, data));
}

fn get_tag_variants(_data: @[u8], def: ast::def_id, tcx: ty::ctxt,
                    extres: external_resolver) -> [ty::variant_info] {
    let external_crate_id = def.crate;
    let data =
        cstore::get_crate_data(tcx.sess.get_cstore(), external_crate_id).data;
    let items = ebml::get_doc(ebml::new_doc(data), tag_items);
    let item = find_item(def.node, items);
    let infos: [ty::variant_info] = [];
    let variant_ids = tag_variant_ids(item, external_crate_id);
    for did: ast::def_id in variant_ids {
        let item = find_item(did.node, items);
        let ctor_ty = item_type(item, external_crate_id, tcx, extres);
        let arg_tys: [ty::t] = [];
        alt ty::struct(tcx, ctor_ty) {
          ty::ty_fn(_, args, _, _, _) {
            for a: ty::arg in args { arg_tys += [a.ty]; }
          }
          _ {
            // Nullary tag variant.

          }
        }
        infos += [{args: arg_tys, ctor_ty: ctor_ty, id: did}];
    }
    ret infos;
}

fn family_has_type_params(fam_ch: u8) -> bool {
    ret alt fam_ch as char {
          'c' { false }
          'f' { true }
          'u' { true }
          'p' { true }
          'F' { true }
          'y' { true }
          't' { true }
          'T' { false }
          'm' { false }
          'n' { false }
          'v' { true }
        };
}

fn read_path(d: ebml::doc) -> {path: str, pos: uint} {
    let desc = ebml::doc_data(d);
    let pos = ebml::be_uint_from_bytes(@desc, 0u, 4u);
    let pathbytes = vec::slice::<u8>(desc, 4u, vec::len::<u8>(desc));
    let path = str::unsafe_from_bytes(pathbytes);
    ret {path: path, pos: pos};
}

fn describe_def(items: ebml::doc, id: ast::def_id) -> str {
    if id.crate != ast::local_crate { ret "external"; }
    ret item_family_to_str(item_family(find_item(id.node, items)));
}

fn item_family_to_str(fam: u8) -> str {
    alt fam as char {
      'c' { ret "const"; }
      'f' { ret "fn"; }
      'u' { ret "unsafe fn"; }
      'p' { ret "pure fn"; }
      'F' { ret "native fn"; }
      'y' { ret "type"; }
      'T' { ret "native type"; }
      't' { ret "type"; }
      'm' { ret "mod"; }
      'n' { ret "native mod"; }
      'v' { ret "tag"; }
    }
}

fn get_meta_items(md: ebml::doc) -> [@ast::meta_item] {
    let items: [@ast::meta_item] = [];
    for each meta_item_doc: ebml::doc in
             ebml::tagged_docs(md, tag_meta_item_word) {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        items += [attr::mk_word_item(n)];
    }
    for each meta_item_doc: ebml::doc in
             ebml::tagged_docs(md, tag_meta_item_name_value) {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = ebml::get_doc(meta_item_doc, tag_meta_item_value);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        let v = str::unsafe_from_bytes(ebml::doc_data(vd));
        // FIXME (#611): Should be able to decode meta_name_value variants,
        // but currently they can't be encoded
        items += [attr::mk_name_value_item_str(n, v)];
    }
    for each meta_item_doc: ebml::doc in
             ebml::tagged_docs(md, tag_meta_item_list) {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        let subitems = get_meta_items(meta_item_doc);
        items += [attr::mk_list_item(n, subitems)];
    }
    ret items;
}

fn get_attributes(md: ebml::doc) -> [ast::attribute] {
    let attrs: [ast::attribute] = [];
    alt ebml::maybe_get_doc(md, tag_attributes) {
      option::some(attrs_d) {
        for each attr_doc: ebml::doc in
                 ebml::tagged_docs(attrs_d, tag_attribute) {
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert (vec::len(meta_items) == 1u);
            let meta_item = meta_items[0];
            attrs +=
                [{node: {style: ast::attr_outer, value: *meta_item},
                  span: ast_util::dummy_sp()}];
        }
      }
      option::none. { }
    }
    ret attrs;
}

fn list_meta_items(meta_items: ebml::doc, out: io::writer) {
    for mi: @ast::meta_item in get_meta_items(meta_items) {
        out.write_str(#fmt["%s\n", pprust::meta_item_to_str(*mi)]);
    }
}

fn list_crate_attributes(md: ebml::doc, out: io::writer) {
    out.write_str("=Crate Attributes=\n");

    for attr: ast::attribute in get_attributes(md) {
        out.write_str(#fmt["%s\n", pprust::attribute_to_str(attr)]);
    }

    out.write_str("\n\n");
}

fn get_crate_attributes(data: @[u8]) -> [ast::attribute] {
    ret get_attributes(ebml::new_doc(data));
}

type crate_dep = {cnum: ast::crate_num, ident: str};

fn get_crate_deps(data: @[u8]) -> [crate_dep] {
    let deps: [crate_dep] = [];
    let cratedoc = ebml::new_doc(data);
    let depsdoc = ebml::get_doc(cratedoc, tag_crate_deps);
    let crate_num = 1;
    for each depdoc: ebml::doc in ebml::tagged_docs(depsdoc, tag_crate_dep) {
        let depname = str::unsafe_from_bytes(ebml::doc_data(depdoc));
        deps += [{cnum: crate_num, ident: depname}];
        crate_num += 1;
    }
    ret deps;
}

fn list_crate_deps(data: @[u8], out: io::writer) {
    out.write_str("=External Dependencies=\n");

    for dep: crate_dep in get_crate_deps(data) {
        out.write_str(#fmt["%d %s\n", dep.cnum, dep.ident]);
    }

    out.write_str("\n");
}

fn list_crate_items(bytes: @[u8], md: ebml::doc, out: io::writer) {
    out.write_str("=Items=\n");
    let paths = ebml::get_doc(md, tag_paths);
    let items = ebml::get_doc(md, tag_items);
    let index = ebml::get_doc(paths, tag_index);
    let bs = ebml::get_doc(index, tag_index_buckets);
    for each bucket: ebml::doc in
             ebml::tagged_docs(bs, tag_index_buckets_bucket) {
        let et = tag_index_buckets_bucket_elt;
        for each elt: ebml::doc in ebml::tagged_docs(bucket, et) {
            let data = read_path(elt);
            let def = ebml::doc_at(bytes, data.pos);
            let did_doc = ebml::get_doc(def, tag_def_id);
            let did = parse_def_id(ebml::doc_data(did_doc));
            out.write_str(#fmt["%s (%s)\n", data.path,
                               describe_def(items, did)]);
        }
    }
    out.write_str("\n");
}

fn list_crate_metadata(bytes: @[u8], out: io::writer) {
    let md = ebml::new_doc(bytes);
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
