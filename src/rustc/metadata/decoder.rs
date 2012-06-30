// Decoding metadata from a single crate's metadata

import std::{ebml, map};
import std::map::hashmap;
import io::writer_util;
import syntax::{ast, ast_util};
import syntax::attr;
import middle::ty;
import syntax::ast_map;
import tydecode::{parse_ty_data, parse_def_id, parse_bounds_data,
        parse_ident};
import syntax::print::pprust;
import cmd=cstore::crate_metadata;
import util::ppaux::ty_to_str;
import ebml::deserializer;
import syntax::diagnostic::span_handler;
import common::*;

export class_dtor;
export get_class_fields;
export get_symbol;
export get_enum_variants;
export get_type;
export get_type_param_count;
export get_impl_iface;
export get_class_method;
export get_impl_method;
export lookup_def;
export lookup_item_name;
export resolve_path;
export get_crate_attributes;
export list_crate_metadata;
export crate_dep;
export get_crate_deps;
export get_crate_hash;
export get_crate_vers;
export get_impls_for_mod;
export get_iface_methods;
export get_crate_module_paths;
export get_item_path;
export maybe_find_item; // sketchy
export item_type; // sketchy
export maybe_get_item_ast;
export decode_inlined_item;
export method_info, _impl;

// Used internally by astencode:
export translate_def_id;

// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.

fn lookup_hash(d: ebml::doc, eq_fn: fn@(~[u8]) -> bool, hash: uint) ->
   ~[ebml::doc] {
    let index = ebml::get_doc(d, tag_index);
    let table = ebml::get_doc(index, tag_index_table);
    let hash_pos = table.start + hash % 256u * 4u;
    let pos = io::u64_from_be_bytes(*d.data, hash_pos, 4u) as uint;
    let {tag:_, doc:bucket} = ebml::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet

    let mut result: ~[ebml::doc] = ~[];
    let belt = tag_index_buckets_bucket_elt;
    do ebml::tagged_docs(bucket, belt) |elt| {
        let pos = io::u64_from_be_bytes(*elt.data, elt.start, 4u) as uint;
        if eq_fn(vec::slice::<u8>(*elt.data, elt.start + 4u, elt.end)) {
            vec::push(result, ebml::doc_at(d.data, pos).doc);
        }
    };
    ret result;
}

fn maybe_find_item(item_id: int, items: ebml::doc) -> option<ebml::doc> {
    fn eq_item(bytes: ~[u8], item_id: int) -> bool {
        ret io::u64_from_be_bytes(bytes, 0u, 4u) as int == item_id;
    }
    let eqer = |a| eq_item(a, item_id);
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
fn lookup_item(item_id: int, data: @~[u8]) -> ebml::doc {
    let items = ebml::get_doc(ebml::doc(data), tag_items);
    alt maybe_find_item(item_id, items) {
       none { fail(#fmt("lookup_item: id not found: %d", item_id)); }
       some(d) { d }
    }
}

fn item_family(item: ebml::doc) -> char {
    let fam = ebml::get_doc(item, tag_items_data_item_family);
    ebml::doc_as_u8(fam) as char
}

fn item_symbol(item: ebml::doc) -> str {
    let sym = ebml::get_doc(item, tag_items_data_item_symbol);
    ret str::from_bytes(ebml::doc_data(sym));
}

fn item_parent_item(d: ebml::doc) -> option<ast::def_id> {
    let mut found = none;
    do ebml::tagged_docs(d, tag_items_data_parent_item) |did| {
        found = some(parse_def_id(ebml::doc_data(did)));
    }
    found
}

fn class_member_id(d: ebml::doc, cdata: cmd) -> ast::def_id {
    let tagdoc = ebml::get_doc(d, tag_def_id);
    ret translate_def_id(cdata, parse_def_id(ebml::doc_data(tagdoc)));
}

fn field_mutability(d: ebml::doc) -> ast::class_mutability {
    // Use maybe_get_doc in case it's a method
    option::map_default(
        ebml::maybe_get_doc(d, tag_class_mut),
        ast::class_immutable,
        |d| {
            alt ebml::doc_as_u8(d) as char {
              'm' { ast::class_mutable }
              _   { ast::class_immutable }
            }
        })
}

fn variant_disr_val(d: ebml::doc) -> option<int> {
    do option::chain(ebml::maybe_get_doc(d, tag_disr_val)) |val_doc| {
        int::parse_buf(ebml::doc_data(val_doc), 10u)
    }
}

fn doc_type(doc: ebml::doc, tcx: ty::ctxt, cdata: cmd) -> ty::t {
    let tp = ebml::get_doc(doc, tag_items_data_item_type);
    parse_ty_data(tp.data, cdata.cnum, tp.start, tcx, |did| {
        translate_def_id(cdata, did)
    })
}

fn item_type(item_id: ast::def_id, item: ebml::doc,
             tcx: ty::ctxt, cdata: cmd) -> ty::t {
    let t = doc_type(item, tcx, cdata);
    if family_names_type(item_family(item)) {
        ty::mk_with_id(tcx, t, item_id)
    } else { t }
}

fn item_impl_iface(item: ebml::doc, tcx: ty::ctxt, cdata: cmd)
    -> option<ty::t> {
    let mut result = none;
    do ebml::tagged_docs(item, tag_impl_iface) |ity| {
        result = some(doc_type(ity, tcx, cdata));
    };
    result
}

fn item_ty_param_bounds(item: ebml::doc, tcx: ty::ctxt, cdata: cmd)
    -> @~[ty::param_bounds] {
    let mut bounds = ~[];
    do ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds) |p| {
        let bd = parse_bounds_data(p.data, p.start, cdata.cnum, tcx, |did| {
            translate_def_id(cdata, did)
        });
        vec::push(bounds, bd);
    }
    @bounds
}

fn item_ty_region_param(item: ebml::doc) -> ast::region_param {
    alt ebml::maybe_get_doc(item, tag_region_param) {
      some(rp_doc) {
        let dsr = ebml::ebml_deserializer(rp_doc);
        ast::deserialize_region_param(dsr)
      }
      none { // not all families of items have region params
        ast::rp_none
      }
    }
}

fn item_ty_param_count(item: ebml::doc) -> uint {
    let mut n = 0u;
    ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds,
                      |_p| n += 1u );
    n
}

fn enum_variant_ids(item: ebml::doc, cdata: cmd) -> ~[ast::def_id] {
    let mut ids: ~[ast::def_id] = ~[];
    let v = tag_items_data_item_variant;
    do ebml::tagged_docs(item, v) |p| {
        let ext = parse_def_id(ebml::doc_data(p));
        vec::push(ids, {crate: cdata.cnum, node: ext.node});
    };
    ret ids;
}

// Given a path and serialized crate metadata, returns the IDs of the
// definitions the path may refer to.
fn resolve_path(path: ~[ast::ident], data: @~[u8]) -> ~[ast::def_id] {
    fn eq_item(data: ~[u8], s: str) -> bool {
        ret str::eq(str::from_bytes(data), s);
    }
    let s = ast_util::path_name_i(path);
    let md = ebml::doc(data);
    let paths = ebml::get_doc(md, tag_paths);
    let eqer = |a| eq_item(a, s);
    let mut result: ~[ast::def_id] = ~[];
    #debug("resolve_path: looking up %s", s);
    for lookup_hash(paths, eqer, hash_path(s)).each |doc| {
        let did_doc = ebml::get_doc(doc, tag_def_id);
        vec::push(result, parse_def_id(ebml::doc_data(did_doc)));
    }
    ret result;
}

fn item_path(item_doc: ebml::doc) -> ast_map::path {
    let path_doc = ebml::get_doc(item_doc, tag_path);

    let len_doc = ebml::get_doc(path_doc, tag_path_len);
    let len = ebml::doc_as_u32(len_doc) as uint;

    let mut result = ~[];
    vec::reserve(result, len);

    do ebml::docs(path_doc) |tag, elt_doc| {
        if tag == tag_path_elt_mod {
            let str = ebml::doc_as_str(elt_doc);
            vec::push(result, ast_map::path_mod(@str));
        } else if tag == tag_path_elt_name {
            let str = ebml::doc_as_str(elt_doc);
            vec::push(result, ast_map::path_name(@str));
        } else {
            // ignore tag_path_len element
        }
    }

    ret result;
}

fn item_name(item: ebml::doc) -> ast::ident {
    let name = ebml::get_doc(item, tag_paths_data_name);
    @str::from_bytes(ebml::doc_data(name))
}

fn lookup_item_name(data: @~[u8], id: ast::node_id) -> ast::ident {
    item_name(lookup_item(id, data))
}

fn lookup_def(cnum: ast::crate_num, data: @~[u8], did_: ast::def_id) ->
   ast::def {
    let item = lookup_item(did_.node, data);
    let fam_ch = item_family(item);
    let did = {crate: cnum, node: did_.node};
    // We treat references to enums as references to types.
    alt check fam_ch {
      'c' { ast::def_const(did) }
      'C' { ast::def_class(did) }
      'u' { ast::def_fn(did, ast::unsafe_fn) }
      'f' { ast::def_fn(did, ast::impure_fn) }
      'p' { ast::def_fn(did, ast::pure_fn) }
      'y' { ast::def_ty(did) }
      't' { ast::def_ty(did) }
      'm' { ast::def_mod(did) }
      'n' { ast::def_foreign_mod(did) }
      'v' {
        let mut tid = option::get(item_parent_item(item));
        tid = {crate: cnum, node: tid.node};
        ast::def_variant(tid, did)
      }
      'I' { ast::def_ty(did) }
    }
}

fn get_type(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> ty::ty_param_bounds_and_ty {

    let item = lookup_item(id, cdata.data);
    let t = item_type({crate: cdata.cnum, node: id}, item, tcx, cdata);
    let tp_bounds = if family_has_type_params(item_family(item)) {
        item_ty_param_bounds(item, tcx, cdata)
    } else { @~[] };
    let rp = item_ty_region_param(item);
    ret {bounds: tp_bounds, rp: rp, ty: t};
}

fn get_type_param_count(data: @~[u8], id: ast::node_id) -> uint {
    item_ty_param_count(lookup_item(id, data))
}

fn get_impl_iface(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> option<ty::t> {
    item_impl_iface(lookup_item(id, cdata.data), tcx, cdata)
}

fn get_impl_method(cdata: cmd, id: ast::node_id,
                   name: ast::ident) -> ast::def_id {
    let items = ebml::get_doc(ebml::doc(cdata.data), tag_items);
    let mut found = none;
    do ebml::tagged_docs(find_item(id, items), tag_item_impl_method) |mid| {
        let m_did = parse_def_id(ebml::doc_data(mid));
        if item_name(find_item(m_did.node, items)) == name {
            found = some(translate_def_id(cdata, m_did));
        }
    }
    option::get(found)
}

fn get_class_method(cdata: cmd, id: ast::node_id,
                    name: ast::ident) -> ast::def_id {
    let items = ebml::get_doc(ebml::doc(cdata.data), tag_items);
    let mut found = none;
    let cls_items = alt maybe_find_item(id, items) {
            some(it) { it }
            none { fail (#fmt("get_class_method: class id not found \
             when looking up method %s", *name)) }};
    do ebml::tagged_docs(cls_items, tag_item_iface_method) |mid| {
        let m_did = class_member_id(mid, cdata);
        if item_name(mid) == name {
            found = some(m_did);
        }
    }
    alt found {
      some(found) { found }
      none { fail (#fmt("get_class_method: no method named %s", *name)) }
    }
}

fn class_dtor(cdata: cmd, id: ast::node_id) -> option<ast::def_id> {
    let items = ebml::get_doc(ebml::doc(cdata.data), tag_items);
    let mut found = none;
    let cls_items = alt maybe_find_item(id, items) {
            some(it) { it }
            none     { fail (#fmt("class_dtor: class id not found \
              when looking up dtor for %d", id)); }
    };
    do ebml::tagged_docs(cls_items, tag_item_dtor) |doc| {
         let doc1 = ebml::get_doc(doc, tag_def_id);
         let did = parse_def_id(ebml::doc_data(doc1));
         found = some(translate_def_id(cdata, did));
    };
    found
}

fn get_symbol(data: @~[u8], id: ast::node_id) -> str {
    ret item_symbol(lookup_item(id, data));
}

fn get_item_path(cdata: cmd, id: ast::node_id) -> ast_map::path {
    item_path(lookup_item(id, cdata.data))
}

type decode_inlined_item = fn(
    cdata: cstore::crate_metadata,
    tcx: ty::ctxt,
    path: ast_map::path,
    par_doc: ebml::doc) -> option<ast::inlined_item>;

fn maybe_get_item_ast(cdata: cmd, tcx: ty::ctxt,
                      id: ast::node_id,
                      decode_inlined_item: decode_inlined_item
                     ) -> csearch::found_ast {
    #debug("Looking up item: %d", id);
    let item_doc = lookup_item(id, cdata.data);
    let path = vec::init(item_path(item_doc));
    alt decode_inlined_item(cdata, tcx, path, item_doc) {
      some(ii) { csearch::found(ii) }
      none {
        alt item_parent_item(item_doc) {
          some(did) {
            let did = translate_def_id(cdata, did);
            let parent_item = lookup_item(did.node, cdata.data);
            alt decode_inlined_item(cdata, tcx, path,
                                               parent_item) {
              some(ii) { csearch::found_parent(did, ii) }
              none { csearch::not_found }
            }
          }
          none { csearch::not_found }
        }
      }
    }
}

fn get_enum_variants(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> ~[ty::variant_info] {
    let data = cdata.data;
    let items = ebml::get_doc(ebml::doc(data), tag_items);
    let item = find_item(id, items);
    let mut infos: ~[ty::variant_info] = ~[];
    let variant_ids = enum_variant_ids(item, cdata);
    let mut disr_val = 0;
    for variant_ids.each |did| {
        let item = find_item(did.node, items);
        let ctor_ty = item_type({crate: cdata.cnum, node: id}, item,
                                tcx, cdata);
        let name = item_name(item);
        let mut arg_tys: ~[ty::t] = ~[];
        alt ty::get(ctor_ty).struct {
          ty::ty_fn(f) {
            for f.inputs.each |a| { vec::push(arg_tys, a.ty); }
          }
          _ { /* Nullary enum variant. */ }
        }
        alt variant_disr_val(item) {
          some(val) { disr_val = val; }
          _         { /* empty */ }
        }
        vec::push(infos, @{args: arg_tys, ctor_ty: ctor_ty, name: name,
                           id: did, disr_val: disr_val});
        disr_val += 1;
    }
    ret infos;
}

// NB: These types are duplicated in resolve.rs
type method_info = {did: ast::def_id, n_tps: uint, ident: ast::ident};
type _impl = {did: ast::def_id, ident: ast::ident, methods: ~[@method_info]};

fn item_impl_methods(cdata: cmd, item: ebml::doc, base_tps: uint)
    -> ~[@method_info] {
    let mut rslt = ~[];
    do ebml::tagged_docs(item, tag_item_impl_method) |doc| {
        let m_did = parse_def_id(ebml::doc_data(doc));
        let mth_item = lookup_item(m_did.node, cdata.data);
        vec::push(rslt, @{did: translate_def_id(cdata, m_did),
                    /* FIXME (maybe #2323) tjc: take a look at this. */
                   n_tps: item_ty_param_count(mth_item) - base_tps,
                   ident: item_name(mth_item)});
    }
    rslt
}

fn get_impls_for_mod(cdata: cmd, m_id: ast::node_id,
                     name: option<ast::ident>,
                     get_cdata: fn(ast::crate_num) -> cmd)
    -> @~[@_impl] {
    let data = cdata.data;
    let mod_item = lookup_item(m_id, data);
    let mut result = ~[];
    do ebml::tagged_docs(mod_item, tag_mod_impl) |doc| {
        let did = parse_def_id(ebml::doc_data(doc));
        let local_did = translate_def_id(cdata, did);
          // The impl may be defined in a different crate. Ask the caller
          // to give us the metadata
        let impl_cdata = get_cdata(local_did.crate);
        let impl_data = impl_cdata.data;
        let item = lookup_item(local_did.node, impl_data);
        let nm = item_name(item);
        if alt name { some(n) { n == nm } none { true } } {
           let base_tps = item_ty_param_count(item);
           vec::push(result, @{
                did: local_did, ident: nm,
                methods: item_impl_methods(impl_cdata, item, base_tps)
            });
        };
    }
    @result
}

/* Works for both classes and ifaces */
fn get_iface_methods(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> @~[ty::method] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    do ebml::tagged_docs(item, tag_item_iface_method) |mth| {
        let bounds = item_ty_param_bounds(mth, tcx, cdata);
        let name = item_name(mth);
        let ty = doc_type(mth, tcx, cdata);
        let fty = alt ty::get(ty).struct { ty::ty_fn(f) { f }
          _ {
            tcx.diag.handler().bug(
                "get_iface_methods: id has non-function type");
        } };
        vec::push(result, {ident: name, tps: bounds, fty: fty,
                    purity: alt check item_family(mth) {
                      'u' { ast::unsafe_fn }
                      'f' { ast::impure_fn }
                      'p' { ast::pure_fn }
                    }, vis: ast::public});
    }
    @result
}

// Helper function that gets either fields or methods
fn get_class_members(cdata: cmd, id: ast::node_id,
                     p: fn(char) -> bool) -> ~[ty::field_ty] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    do ebml::tagged_docs(item, tag_item_field) |an_item| {
       let f = item_family(an_item);
       if p(f) {
          let name = item_name(an_item);
          let did = class_member_id(an_item, cdata);
          let mt = field_mutability(an_item);
          vec::push(result, {ident: name, id: did, vis:
                  family_to_visibility(f), mutability: mt});
       }
    }
    result
}

pure fn family_to_visibility(family: char) -> ast::visibility {
    alt family {
      'g' { ast::public }
      _   { ast::private }
    }
}

/* 'g' for public field, 'j' for private field */
fn get_class_fields(cdata: cmd, id: ast::node_id) -> ~[ty::field_ty] {
    get_class_members(cdata, id, |f| f == 'g' || f == 'j')
}

fn family_has_type_params(fam_ch: char) -> bool {
    alt check fam_ch {
      'c' | 'T' | 'm' | 'n' | 'g' | 'h' | 'j' { false }
      'f' | 'u' | 'p' | 'F' | 'U' | 'P' | 'y' | 't' | 'v' | 'i' | 'I' | 'C'
          | 'a'
          { true }
    }
}

fn family_names_type(fam_ch: char) -> bool {
    alt fam_ch { 'y' | 't' | 'I' { true } _ { false } }
}

fn read_path(d: ebml::doc) -> {path: str, pos: uint} {
    let desc = ebml::doc_data(d);
    let pos = io::u64_from_be_bytes(desc, 0u, 4u) as uint;
    let pathbytes = vec::slice::<u8>(desc, 4u, vec::len::<u8>(desc));
    let path = str::from_bytes(pathbytes);
    ret {path: path, pos: pos};
}

fn describe_def(items: ebml::doc, id: ast::def_id) -> str {
    if id.crate != ast::local_crate { ret "external"; }
    let it = alt maybe_find_item(id.node, items) {
        some(it) { it }
        none { fail (#fmt("describe_def: item not found %?", id)); }
    };
    ret item_family_to_str(item_family(it));
}

fn item_family_to_str(fam: char) -> str {
    alt check fam {
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
      'v' { ret "enum"; }
      'i' { ret "impl"; }
      'I' { ret "iface"; }
      'C' { ret "class"; }
      'g' { ret "public field"; }
      'j' { ret "private field"; }
    }
}

fn get_meta_items(md: ebml::doc) -> ~[@ast::meta_item] {
    let mut items: ~[@ast::meta_item] = ~[];
    do ebml::tagged_docs(md, tag_meta_item_word) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(ebml::doc_data(nd));
        vec::push(items, attr::mk_word_item(@n));
    };
    do ebml::tagged_docs(md, tag_meta_item_name_value) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = ebml::get_doc(meta_item_doc, tag_meta_item_value);
        let n = str::from_bytes(ebml::doc_data(nd));
        let v = str::from_bytes(ebml::doc_data(vd));
        // FIXME (#623): Should be able to decode meta_name_value variants,
        // but currently the encoder just drops them
        vec::push(items, attr::mk_name_value_item_str(@n, v));
    };
    do ebml::tagged_docs(md, tag_meta_item_list) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(ebml::doc_data(nd));
        let subitems = get_meta_items(meta_item_doc);
        vec::push(items, attr::mk_list_item(@n, subitems));
    };
    ret items;
}

fn get_attributes(md: ebml::doc) -> ~[ast::attribute] {
    let mut attrs: ~[ast::attribute] = ~[];
    alt ebml::maybe_get_doc(md, tag_attributes) {
      option::some(attrs_d) {
        do ebml::tagged_docs(attrs_d, tag_attribute) |attr_doc| {
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert (vec::len(meta_items) == 1u);
            let meta_item = meta_items[0];
            vec::push(attrs,
                      {node: {style: ast::attr_outer, value: *meta_item},
                       span: ast_util::dummy_sp()});
        };
      }
      option::none { }
    }
    ret attrs;
}

fn list_meta_items(meta_items: ebml::doc, out: io::writer) {
    for get_meta_items(meta_items).each |mi| {
        out.write_str(#fmt["%s\n", pprust::meta_item_to_str(*mi)]);
    }
}

fn list_crate_attributes(md: ebml::doc, hash: @str, out: io::writer) {
    out.write_str(#fmt("=Crate Attributes (%s)=\n", *hash));

    for get_attributes(md).each |attr| {
        out.write_str(#fmt["%s\n", pprust::attribute_to_str(attr)]);
    }

    out.write_str("\n\n");
}

fn get_crate_attributes(data: @~[u8]) -> ~[ast::attribute] {
    ret get_attributes(ebml::doc(data));
}

type crate_dep = {cnum: ast::crate_num, name: ast::ident,
                  vers: @str, hash: @str};

fn get_crate_deps(data: @~[u8]) -> ~[crate_dep] {
    let mut deps: ~[crate_dep] = ~[];
    let cratedoc = ebml::doc(data);
    let depsdoc = ebml::get_doc(cratedoc, tag_crate_deps);
    let mut crate_num = 1;
    fn docstr(doc: ebml::doc, tag_: uint) -> str {
        str::from_bytes(ebml::doc_data(ebml::get_doc(doc, tag_)))
    }
    do ebml::tagged_docs(depsdoc, tag_crate_dep) |depdoc| {
        vec::push(deps, {cnum: crate_num,
                  name: @docstr(depdoc, tag_crate_dep_name),
                  vers: @docstr(depdoc, tag_crate_dep_vers),
                  hash: @docstr(depdoc, tag_crate_dep_hash)});
        crate_num += 1;
    };
    ret deps;
}

fn list_crate_deps(data: @~[u8], out: io::writer) {
    out.write_str("=External Dependencies=\n");

    for get_crate_deps(data).each |dep| {
        out.write_str(#fmt["%d %s-%s-%s\n",
                           dep.cnum, *dep.name, *dep.hash, *dep.vers]);
    }

    out.write_str("\n");
}

fn get_crate_hash(data: @~[u8]) -> @str {
    let cratedoc = ebml::doc(data);
    let hashdoc = ebml::get_doc(cratedoc, tag_crate_hash);
    ret @str::from_bytes(ebml::doc_data(hashdoc));
}

fn get_crate_vers(data: @~[u8]) -> @str {
    let attrs = decoder::get_crate_attributes(data);
    ret alt attr::last_meta_item_value_str_by_name(
        attr::find_linkage_metas(attrs), "vers") {
      some(ver) { ver }
      none { @"0.0" }
    };
}

fn list_crate_items(bytes: @~[u8], md: ebml::doc, out: io::writer) {
    out.write_str("=Items=\n");
    let items = ebml::get_doc(md, tag_items);
    do iter_crate_items(bytes) |path, did| {
        out.write_str(#fmt["%s (%s)\n", path, describe_def(items, did)]);
    }
    out.write_str("\n");
}

fn iter_crate_items(bytes: @~[u8], proc: fn(str, ast::def_id)) {
    let md = ebml::doc(bytes);
    let paths = ebml::get_doc(md, tag_paths);
    let index = ebml::get_doc(paths, tag_index);
    let bs = ebml::get_doc(index, tag_index_buckets);
    do ebml::tagged_docs(bs, tag_index_buckets_bucket) |bucket| {
        let et = tag_index_buckets_bucket_elt;
        do ebml::tagged_docs(bucket, et) |elt| {
            let data = read_path(elt);
            let {tag:_, doc:def} = ebml::doc_at(bytes, data.pos);
            let did_doc = ebml::get_doc(def, tag_def_id);
            let did = parse_def_id(ebml::doc_data(did_doc));
            proc(data.path, did);
        };
    };
}

fn get_crate_module_paths(bytes: @~[u8]) -> ~[(ast::def_id, str)] {
    fn mod_of_path(p: str) -> str {
        str::connect(vec::init(str::split_str(p, "::")), "::")
    }

    // find all module (path, def_ids), which are not
    // fowarded path due to renamed import or reexport
    let mut res = ~[];
    let mods = map::str_hash();
    do iter_crate_items(bytes) |path, did| {
        let m = mod_of_path(path);
        if str::is_not_empty(m) {
            // if m has a sub-item, it must be a module
            mods.insert(m, true);
        }
        // Collect everything by now. There might be multiple
        // paths pointing to the same did. Those will be
        // unified later by using the mods map
        vec::push(res, (did, path));
    }
    ret do vec::filter(res) |x| {
        let (_, xp) = x;
        mods.contains_key(xp)
    }
}

fn list_crate_metadata(bytes: @~[u8], out: io::writer) {
    let hash = get_crate_hash(bytes);
    let md = ebml::doc(bytes);
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
      option::none { fail "didn't find a crate in the cnum_map"; }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
