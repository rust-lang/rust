// Decoding metadata from a single crate's metadata

import std::{ebml, io};
import io::writer_util;
import syntax::{ast, ast_util};
import front::attr;
import middle::ty;
import common::*;
import tydecode::{parse_ty_data, parse_def_id, parse_bounds_data};
import syntax::print::pprust;
import cmd=cstore::crate_metadata;

export get_symbol;
export get_tag_variants;
export get_type;
export get_type_param_count;
export get_impl_iface;
export lookup_def;
export lookup_item_name;
export get_impl_iface;
export resolve_path;
export get_crate_attributes;
export list_crate_metadata;
export crate_dep;
export get_crate_deps;
export get_crate_hash;
export get_impls_for_mod;
export get_iface_methods;
// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.

fn lookup_hash(d: ebml::doc, eq_fn: fn@([u8]) -> bool, hash: uint) ->
   [ebml::doc] {
    let index = ebml::get_doc(d, tag_index);
    let table = ebml::get_doc(index, tag_index_table);
    let hash_pos = table.start + hash % 256u * 4u;
    let pos = ebml::be_uint_from_bytes(d.data, hash_pos, 4u);
    let bucket = ebml::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet

    let result: [ebml::doc] = [];
    let belt = tag_index_buckets_bucket_elt;
    ebml::tagged_docs(bucket, belt) {|elt|
        let pos = ebml::be_uint_from_bytes(elt.data, elt.start, 4u);
        if eq_fn(vec::slice::<u8>(*elt.data, elt.start + 4u, elt.end)) {
            result += [ebml::doc_at(d.data, pos)];
        }
    };
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

fn variant_disr_val(d: ebml::doc) -> option::t<int> {
    alt ebml::maybe_get_doc(d, tag_disr_val) {
      some(val_doc) {
        let val_buf = ebml::doc_data(val_doc);
        let val = int::parse_buf(val_buf, 10u);
        ret some(val);
      }
      _ { ret none;}
    }
}

fn doc_type(doc: ebml::doc, tcx: ty::ctxt, cdata: cmd) -> ty::t {
    let tp = ebml::get_doc(doc, tag_items_data_item_type);
    parse_ty_data(tp.data, cdata.cnum, tp.start, tcx, {|did|
        translate_def_id(cdata, did)
    })
}

fn item_type(item: ebml::doc, tcx: ty::ctxt, cdata: cmd) -> ty::t {
    let t = doc_type(item, tcx, cdata);
    if family_names_type(item_family(item)) {
        ty::mk_named(tcx, t, @item_name(item))
    } else { t }
}

fn item_impl_iface(item: ebml::doc, tcx: ty::ctxt, cdata: cmd)
    -> option::t<ty::t> {
    let result = none;
    ebml::tagged_docs(item, tag_impl_iface) {|ity|
        let t = parse_ty_data(ity.data, cdata.cnum, ity.start, tcx, {|did|
            translate_def_id(cdata, did)
        });
        result = some(t);
    }
    result
}

fn item_ty_param_bounds(item: ebml::doc, tcx: ty::ctxt, cdata: cmd)
    -> @[ty::param_bounds] {
    let bounds = [];
    ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds) {|p|
        let bd = parse_bounds_data(p.data, p.start, cdata.cnum, tcx, {|did|
            translate_def_id(cdata, did)
        });
        bounds += [bd];
    }
    @bounds
}

fn item_ty_param_count(item: ebml::doc) -> uint {
    let n = 0u;
    ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds,
                      {|_p| n += 1u; });
    n
}

fn tag_variant_ids(item: ebml::doc, cdata: cmd) -> [ast::def_id] {
    let ids: [ast::def_id] = [];
    let v = tag_items_data_item_variant;
    ebml::tagged_docs(item, v) {|p|
        let ext = parse_def_id(ebml::doc_data(p));
        ids += [{crate: cdata.cnum, node: ext.node}];
    };
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

fn item_name(item: ebml::doc) -> ast::ident {
    let name = ebml::get_doc(item, tag_paths_data_name);
    str::unsafe_from_bytes(ebml::doc_data(name))
}

fn lookup_item_name(data: @[u8], id: ast::node_id) -> ast::ident {
    item_name(lookup_item(id, data))
}

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
          'U' { ast::def_native_fn(did, ast::unsafe_fn) }
          'F' { ast::def_native_fn(did, ast::impure_fn) }
          'P' { ast::def_native_fn(did, ast::pure_fn) }
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
          'I' { ast::def_ty(did) }
        };
    ret def;
}

fn get_type(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> ty::ty_param_bounds_and_ty {
    let item = lookup_item(id, cdata.data);
    let t = item_type(item, tcx, cdata);
    let tp_bounds = if family_has_type_params(item_family(item)) {
        item_ty_param_bounds(item, tcx, cdata)
    } else { @[] };
    ret {bounds: tp_bounds, ty: t};
}

fn get_type_param_count(data: @[u8], id: ast::node_id) -> uint {
    item_ty_param_count(lookup_item(id, data))
}

fn get_impl_iface(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> option::t<ty::t> {
    item_impl_iface(lookup_item(id, cdata.data), tcx, cdata)
}

fn get_symbol(data: @[u8], id: ast::node_id) -> str {
    ret item_symbol(lookup_item(id, data));
}

fn get_tag_variants(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> [ty::variant_info] {
    let data = cdata.data;
    let items = ebml::get_doc(ebml::new_doc(data), tag_items);
    let item = find_item(id, items);
    let infos: [ty::variant_info] = [];
    let variant_ids = tag_variant_ids(item, cdata);
    let disr_val = 0;
    for did: ast::def_id in variant_ids {
        let item = find_item(did.node, items);
        let ctor_ty = item_type(item, tcx, cdata);
        let name = item_name(item);
        let arg_tys: [ty::t] = [];
        alt ty::struct(tcx, ctor_ty) {
          ty::ty_fn(f) {
            for a: ty::arg in f.inputs { arg_tys += [a.ty]; }
          }
          _ { /* Nullary tag variant. */ }
        }
        alt variant_disr_val(item) {
          some(val) { disr_val = val; }
          _         { /* empty */ }
        }
        infos += [@{args: arg_tys, ctor_ty: ctor_ty, name: name,
                    id: did, disr_val: disr_val}];
        disr_val += 1;
    }
    ret infos;
}

fn item_impl_methods(cdata: cmd, item: ebml::doc, base_tps: uint)
    -> [@middle::resolve::method_info] {
    let rslt = [];
    ebml::tagged_docs(item, tag_item_method) {|doc|
        let m_did = parse_def_id(ebml::doc_data(doc));
        let mth_item = lookup_item(m_did.node, cdata.data);
        rslt += [@{did: translate_def_id(cdata, m_did),
                   n_tps: item_ty_param_count(mth_item) - base_tps,
                   ident: item_name(mth_item)}];
    }
    rslt
}

fn get_impls_for_mod(cdata: cmd, m_id: ast::node_id,
                     name: option::t<ast::ident>)
    -> @[@middle::resolve::_impl] {
    let data = cdata.data;
    let mod_item = lookup_item(m_id, data), result = [];
    ebml::tagged_docs(mod_item, tag_mod_impl) {|doc|
        let did = translate_def_id(cdata, parse_def_id(ebml::doc_data(doc)));
        let item = lookup_item(did.node, data), nm = item_name(item);
        if alt name { some(n) { n == nm } none. { true } } {
            let base_tps = item_ty_param_count(doc);
            result += [@{did: did, ident: nm,
                         methods: item_impl_methods(cdata, item, base_tps)}];
        }
    }
    @result
}

fn get_iface_methods(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> @[ty::method] {
    let data = cdata.data;
    let item = lookup_item(id, data), result = [];
    ebml::tagged_docs(item, tag_item_method) {|mth|
        let bounds = item_ty_param_bounds(mth, tcx, cdata);
        let name = item_name(mth);
        let ty = doc_type(mth, tcx, cdata);
        let fty = alt ty::struct(tcx, ty) { ty::ty_fn(f) { f } };
        result += [{ident: name, tps: bounds, fty: fty}];
    }
    @result
}

fn family_has_type_params(fam_ch: u8) -> bool {
    alt fam_ch as char {
      'c' | 'T' | 'm' | 'n' { false }
      'f' | 'u' | 'p' | 'F' | 'U' | 'P' | 'y' | 't' | 'v' | 'i' | 'I' { true }
    }
}

fn family_names_type(fam_ch: u8) -> bool {
    alt fam_ch as char { 'y' | 't' | 'I' { true } _ { false } }
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
      'U' { ret "unsafe native fn"; }
      'P' { ret "pure native fn"; }
      'y' { ret "type"; }
      'T' { ret "native type"; }
      't' { ret "type"; }
      'm' { ret "mod"; }
      'n' { ret "native mod"; }
      'v' { ret "tag"; }
      'i' { ret "impl"; }
      'I' { ret "iface"; }
    }
}

fn get_meta_items(md: ebml::doc) -> [@ast::meta_item] {
    let items: [@ast::meta_item] = [];
    ebml::tagged_docs(md, tag_meta_item_word) {|meta_item_doc|
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        items += [attr::mk_word_item(n)];
    };
    ebml::tagged_docs(md, tag_meta_item_name_value) {|meta_item_doc|
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = ebml::get_doc(meta_item_doc, tag_meta_item_value);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        let v = str::unsafe_from_bytes(ebml::doc_data(vd));
        // FIXME (#611): Should be able to decode meta_name_value variants,
        // but currently they can't be encoded
        items += [attr::mk_name_value_item_str(n, v)];
    };
    ebml::tagged_docs(md, tag_meta_item_list) {|meta_item_doc|
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::unsafe_from_bytes(ebml::doc_data(nd));
        let subitems = get_meta_items(meta_item_doc);
        items += [attr::mk_list_item(n, subitems)];
    };
    ret items;
}

fn get_attributes(md: ebml::doc) -> [ast::attribute] {
    let attrs: [ast::attribute] = [];
    alt ebml::maybe_get_doc(md, tag_attributes) {
      option::some(attrs_d) {
        ebml::tagged_docs(attrs_d, tag_attribute) {|attr_doc|
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert (vec::len(meta_items) == 1u);
            let meta_item = meta_items[0];
            attrs +=
                [{node: {style: ast::attr_outer, value: *meta_item},
                  span: ast_util::dummy_sp()}];
        };
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

fn list_crate_attributes(md: ebml::doc, hash: str, out: io::writer) {
    out.write_str(#fmt("=Crate Attributes (%s)=\n", hash));

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
    ebml::tagged_docs(depsdoc, tag_crate_dep) {|depdoc|
        let depname = str::unsafe_from_bytes(ebml::doc_data(depdoc));
        deps += [{cnum: crate_num, ident: depname}];
        crate_num += 1;
    };
    ret deps;
}

fn list_crate_deps(data: @[u8], out: io::writer) {
    out.write_str("=External Dependencies=\n");

    for dep: crate_dep in get_crate_deps(data) {
        out.write_str(#fmt["%d %s\n", dep.cnum, dep.ident]);
    }

    out.write_str("\n");
}

fn get_crate_hash(data: @[u8]) -> str {
    let cratedoc = ebml::new_doc(data);
    let hashdoc = ebml::get_doc(cratedoc, tag_crate_hash);
    ret str::unsafe_from_bytes(ebml::doc_data(hashdoc));
}

fn list_crate_items(bytes: @[u8], md: ebml::doc, out: io::writer) {
    out.write_str("=Items=\n");
    let paths = ebml::get_doc(md, tag_paths);
    let items = ebml::get_doc(md, tag_items);
    let index = ebml::get_doc(paths, tag_index);
    let bs = ebml::get_doc(index, tag_index_buckets);
    ebml::tagged_docs(bs, tag_index_buckets_bucket) {|bucket|
        let et = tag_index_buckets_bucket_elt;
        ebml::tagged_docs(bucket, et) {|elt|
            let data = read_path(elt);
            let def = ebml::doc_at(bytes, data.pos);
            let did_doc = ebml::get_doc(def, tag_def_id);
            let did = parse_def_id(ebml::doc_data(did_doc));
            out.write_str(#fmt["%s (%s)\n", data.path,
                               describe_def(items, did)]);
        };
    };
    out.write_str("\n");
}

fn list_crate_metadata(bytes: @[u8], out: io::writer) {
    let hash = get_crate_hash(bytes);
    let md = ebml::new_doc(bytes);
    list_crate_attributes(md, hash, out);
    list_crate_deps(bytes, out);
    list_crate_items(bytes, md, out);
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
fn translate_def_id(cdata: cmd, did: ast::def_id) -> ast::def_id {
    if did.crate == ast::local_crate {
        ret {crate: cdata.cnum, node: did.node};
    }

    alt cdata.cnum_map.find(did.crate) {
      option::some(n) { ret {crate: n, node: did.node}; }
      option::none. { fail "didn't find a crate in the cnum_map"; }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
