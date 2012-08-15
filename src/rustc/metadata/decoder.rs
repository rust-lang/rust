// Decoding metadata from a single crate's metadata

import std::{ebml, map};
import std::map::{hashmap, str_hash};
import io::WriterUtil;
import dvec::{DVec, dvec};
import syntax::{ast, ast_util};
import syntax::attr;
import middle::ty;
import syntax::ast_map;
import tydecode::{parse_ty_data, parse_def_id, parse_bounds_data,
        parse_ident};
import syntax::print::pprust;
import cmd=cstore::crate_metadata;
import util::ppaux::ty_to_str;
import syntax::diagnostic::span_handler;
import common::*;

export class_dtor;
export get_class_fields;
export get_symbol;
export get_enum_variants;
export get_type;
export get_region_param;
export get_type_param_count;
export get_impl_traits;
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
export get_trait_methods;
export get_method_names_if_trait;
export get_item_attrs;
export get_crate_module_paths;
export def_like;
export dl_def;
export dl_impl;
export dl_field;
export path_entry;
export each_path;
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

fn lookup_hash(d: ebml::doc, eq_fn: fn(x:&[u8]) -> bool, hash: uint) ->
   option<ebml::doc> {
    let index = ebml::get_doc(d, tag_index);
    let table = ebml::get_doc(index, tag_index_table);
    let hash_pos = table.start + hash % 256u * 4u;
    let pos = io::u64_from_be_bytes(*d.data, hash_pos, 4u) as uint;
    let {tag:_, doc:bucket} = ebml::doc_at(d.data, pos);

    let belt = tag_index_buckets_bucket_elt;
    for ebml::tagged_docs(bucket, belt) |elt| {
        let pos = io::u64_from_be_bytes(*elt.data, elt.start, 4u) as uint;
        if eq_fn(vec::slice(*elt.data, elt.start + 4u, elt.end)) {
            return some(ebml::doc_at(d.data, pos).doc);
        }
    };
    none
}

fn maybe_find_item(item_id: int, items: ebml::doc) -> option<ebml::doc> {
    fn eq_item(bytes: &[u8], item_id: int) -> bool {
        return io::u64_from_be_bytes(vec::slice(bytes, 0u, 4u), 0u, 4u) as int
            == item_id;
    }
    lookup_hash(items,
                |a| eq_item(a, item_id),
                hash_node_id(item_id))
}

fn find_item(item_id: int, items: ebml::doc) -> ebml::doc {
    return option::get(maybe_find_item(item_id, items));
}

// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item(item_id: int, data: @~[u8]) -> ebml::doc {
    let items = ebml::get_doc(ebml::doc(data), tag_items);
    match maybe_find_item(item_id, items) {
       none => fail(fmt!{"lookup_item: id not found: %d", item_id}),
       some(d) => d
    }
}

fn item_family(item: ebml::doc) -> char {
    let fam = ebml::get_doc(item, tag_items_data_item_family);
    ebml::doc_as_u8(fam) as char
}

fn item_symbol(item: ebml::doc) -> ~str {
    let sym = ebml::get_doc(item, tag_items_data_item_symbol);
    return str::from_bytes(ebml::doc_data(sym));
}

fn item_parent_item(d: ebml::doc) -> option<ast::def_id> {
    for ebml::tagged_docs(d, tag_items_data_parent_item) |did| {
        return some(ebml::with_doc_data(did, |d| parse_def_id(d)));
    }
    none
}

fn item_def_id(d: ebml::doc, cdata: cmd) -> ast::def_id {
    let tagdoc = ebml::get_doc(d, tag_def_id);
    return translate_def_id(cdata, ebml::with_doc_data(tagdoc,
                                                    |d| parse_def_id(d)));
}

fn field_mutability(d: ebml::doc) -> ast::class_mutability {
    // Use maybe_get_doc in case it's a method
    option::map_default(
        ebml::maybe_get_doc(d, tag_class_mut),
        ast::class_immutable,
        |d| {
            match ebml::doc_as_u8(d) as char {
              'm' => ast::class_mutable,
              _   => ast::class_immutable
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

fn item_impl_traits(item: ebml::doc, tcx: ty::ctxt, cdata: cmd) -> ~[ty::t] {
    let mut results = ~[];
    for ebml::tagged_docs(item, tag_impl_trait) |ity| {
        vec::push(results, doc_type(ity, tcx, cdata));
    };
    results
}

fn item_ty_param_bounds(item: ebml::doc, tcx: ty::ctxt, cdata: cmd)
    -> @~[ty::param_bounds] {
    let mut bounds = ~[];
    for ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds) |p| {
        let bd = parse_bounds_data(p.data, p.start, cdata.cnum, tcx, |did| {
            translate_def_id(cdata, did)
        });
        vec::push(bounds, bd);
    }
    @bounds
}

fn item_ty_region_param(item: ebml::doc) -> bool {
    match ebml::maybe_get_doc(item, tag_region_param) {
      some(_) => true,
      none => false
    }
}

fn item_ty_param_count(item: ebml::doc) -> uint {
    let mut n = 0u;
    ebml::tagged_docs(item, tag_items_data_item_ty_param_bounds,
                      |_p| { n += 1u; true } );
    n
}

fn enum_variant_ids(item: ebml::doc, cdata: cmd) -> ~[ast::def_id] {
    let mut ids: ~[ast::def_id] = ~[];
    let v = tag_items_data_item_variant;
    for ebml::tagged_docs(item, v) |p| {
        let ext = ebml::with_doc_data(p, |d| parse_def_id(d));
        vec::push(ids, {crate: cdata.cnum, node: ext.node});
    };
    return ids;
}

// Given a path and serialized crate metadata, returns the IDs of the
// definitions the path may refer to.
fn resolve_path(path: ~[ast::ident], data: @~[u8]) -> ~[ast::def_id] {
    fn eq_item(data: &[u8], s: ~str) -> bool {
        // XXX: Use string equality.
        let data_len = data.len();
        let s_len = s.len();
        if data_len != s_len {
            return false;
        }
        let mut i = 0;
        while i < data_len {
            if data[i] != s[i] {
                return false;
            }
            i += 1;
        }
        return true;
    }
    let s = ast_util::path_name_i(path);
    let md = ebml::doc(data);
    let paths = ebml::get_doc(md, tag_paths);
    let eqer = |a| eq_item(a, s);
    let mut result: ~[ast::def_id] = ~[];
    debug!{"resolve_path: looking up %s", s};
    for lookup_hash(paths, eqer, hash_path(s)).each |doc| {
        let did_doc = ebml::get_doc(doc, tag_def_id);
        vec::push(result, ebml::with_doc_data(did_doc, |d| parse_def_id(d)));
    }
    return result;
}

fn item_path(item_doc: ebml::doc) -> ast_map::path {
    let path_doc = ebml::get_doc(item_doc, tag_path);

    let len_doc = ebml::get_doc(path_doc, tag_path_len);
    let len = ebml::doc_as_u32(len_doc) as uint;

    let mut result = ~[];
    vec::reserve(result, len);

    for ebml::docs(path_doc) |tag, elt_doc| {
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

    return result;
}

fn item_name(item: ebml::doc) -> ast::ident {
    let name = ebml::get_doc(item, tag_paths_data_name);
    @str::from_bytes(ebml::doc_data(name))
}

fn lookup_item_name(data: @~[u8], id: ast::node_id) -> ast::ident {
    item_name(lookup_item(id, data))
}

fn item_to_def_like(item: ebml::doc, did: ast::def_id, cnum: ast::crate_num)
        -> def_like {
    let fam_ch = item_family(item);
    match fam_ch {
      'c' => dl_def(ast::def_const(did)),
      'C' => dl_def(ast::def_class(did, true)),
      'S' => dl_def(ast::def_class(did, false)),
      'u' => dl_def(ast::def_fn(did, ast::unsafe_fn)),
      'f' => dl_def(ast::def_fn(did, ast::impure_fn)),
      'p' => dl_def(ast::def_fn(did, ast::pure_fn)),
      'e' => dl_def(ast::def_fn(did, ast::extern_fn)),
      'U' => dl_def(ast::def_static_method(did, ast::unsafe_fn)),
      'F' => dl_def(ast::def_static_method(did, ast::impure_fn)),
      'P' => dl_def(ast::def_static_method(did, ast::pure_fn)),
      'y' => dl_def(ast::def_ty(did)),
      't' => dl_def(ast::def_ty(did)),
      'm' => dl_def(ast::def_mod(did)),
      'n' => dl_def(ast::def_foreign_mod(did)),
      'v' => {
        let mut tid = option::get(item_parent_item(item));
        tid = {crate: cnum, node: tid.node};
        dl_def(ast::def_variant(tid, did))
      }
      'I' => dl_def(ast::def_ty(did)),
      'i' => dl_impl(did),
      'g' | 'j' | 'N' => dl_field,
      ch => fail fmt!{"unexpected family code: '%c'", ch}
    }
}

fn lookup_def(cnum: ast::crate_num, data: @~[u8], did_: ast::def_id) ->
   ast::def {
    let item = lookup_item(did_.node, data);
    let did = {crate: cnum, node: did_.node};
    // We treat references to enums as references to types.
    return def_like_to_def(item_to_def_like(item, did, cnum));
}

fn get_type(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> ty::ty_param_bounds_and_ty {

    let item = lookup_item(id, cdata.data);
    let t = item_type({crate: cdata.cnum, node: id}, item, tcx, cdata);
    let tp_bounds = if family_has_type_params(item_family(item)) {
        item_ty_param_bounds(item, tcx, cdata)
    } else { @~[] };
    let rp = item_ty_region_param(item);
    return {bounds: tp_bounds, rp: rp, ty: t};
}

fn get_region_param(cdata: cmd, id: ast::node_id) -> bool {
    let item = lookup_item(id, cdata.data);
    return item_ty_region_param(item);
}

fn get_type_param_count(data: @~[u8], id: ast::node_id) -> uint {
    item_ty_param_count(lookup_item(id, data))
}

fn get_impl_traits(cdata: cmd, id: ast::node_id, tcx: ty::ctxt) -> ~[ty::t] {
    item_impl_traits(lookup_item(id, cdata.data), tcx, cdata)
}

fn get_impl_method(cdata: cmd, id: ast::node_id,
                   name: ast::ident) -> ast::def_id {
    let items = ebml::get_doc(ebml::doc(cdata.data), tag_items);
    let mut found = none;
    for ebml::tagged_docs(find_item(id, items), tag_item_impl_method) |mid| {
        let m_did = ebml::with_doc_data(mid, |d| parse_def_id(d));
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
    let cls_items = match maybe_find_item(id, items) {
      some(it) => it,
      none => fail (fmt!{"get_class_method: class id not found \
                              when looking up method %s", *name})
    };
    for ebml::tagged_docs(cls_items, tag_item_trait_method) |mid| {
        let m_did = item_def_id(mid, cdata);
        if item_name(mid) == name {
            found = some(m_did);
        }
    }
    match found {
      some(found) => found,
      none => fail (fmt!{"get_class_method: no method named %s", *name})
    }
}

fn class_dtor(cdata: cmd, id: ast::node_id) -> option<ast::def_id> {
    let items = ebml::get_doc(ebml::doc(cdata.data), tag_items);
    let mut found = none;
    let cls_items = match maybe_find_item(id, items) {
            some(it) => it,
            none     => fail (fmt!{"class_dtor: class id not found \
              when looking up dtor for %d", id})
    };
    for ebml::tagged_docs(cls_items, tag_item_dtor) |doc| {
         let doc1 = ebml::get_doc(doc, tag_def_id);
         let did = ebml::with_doc_data(doc1, |d| parse_def_id(d));
         found = some(translate_def_id(cdata, did));
    };
    found
}

fn get_symbol(data: @~[u8], id: ast::node_id) -> ~str {
    return item_symbol(lookup_item(id, data));
}

// Something that a name can resolve to.
enum def_like {
    dl_def(ast::def),
    dl_impl(ast::def_id),
    dl_field
}

fn def_like_to_def(def_like: def_like) -> ast::def {
    match def_like {
        dl_def(def) => return def,
        dl_impl(*) => fail ~"found impl in def_like_to_def",
        dl_field => fail ~"found field in def_like_to_def"
    }
}

// A path.
class path_entry {
    // The full path, separated by '::'.
    let path_string: ~str;
    // The definition, implementation, or field that this path corresponds to.
    let def_like: def_like;

    new(path_string: ~str, def_like: def_like) {
        self.path_string = path_string;
        self.def_like = def_like;
    }
}

/// Iterates over all the paths in the given crate.
fn each_path(cdata: cmd, f: fn(path_entry) -> bool) {
    let root = ebml::doc(cdata.data);
    let items = ebml::get_doc(root, tag_items);
    let items_data = ebml::get_doc(items, tag_items_data);

    let mut broken = false;

    // First, go through all the explicit items.
    for ebml::tagged_docs(items_data, tag_items_data_item) |item_doc| {
        if !broken {
            let name = ast_map::path_to_str_with_sep(item_path(item_doc),
                                                     ~"::");
            if name != ~"" {
                // Extract the def ID.
                let def_id = item_def_id(item_doc, cdata);

                // Construct the def for this item.
                debug!{"(each_path) yielding explicit item: %s", name};
                let def_like = item_to_def_like(item_doc, def_id, cdata.cnum);

                // Hand the information off to the iteratee.
                let this_path_entry = path_entry(name, def_like);
                if !f(this_path_entry) {
                    broken = true;      // XXX: This is awful.
                }
            }
        }
    }

    // If broken, stop here.
    if broken {
        return;
    }

    // Next, go through all the paths. We will find items that we didn't know
    // about before (reexports in particular).
    //
    // XXX: This is broken; the paths are actually hierarchical.

    let outer_paths = ebml::get_doc(root, tag_paths);
    let inner_paths = ebml::get_doc(outer_paths, tag_paths);

    fn g(cdata: cmd, items: ebml::doc, path_doc: ebml::doc, &broken: bool,
         f: fn(path_entry) -> bool) {

        if !broken {
            let path = item_name(path_doc);

            // Extract the def ID.
            let def_id = item_def_id(path_doc, cdata);

            // Get the item.
            match maybe_find_item(def_id.node, items) {
                none => {
                    debug!{"(each_path) ignoring implicit item: %s",
                            *path};
                }
                some(item_doc) => {
                    // Construct the def for this item.
                    let def_like = item_to_def_like(item_doc, def_id,
                                                    cdata.cnum);

                    // Hand the information off to the iteratee.
                    debug!{"(each_path) yielding implicit item: %s",
                            *path};
                    let this_path_entry = path_entry(*path, def_like);
                    if (!f(this_path_entry)) {
                        broken = true;      // XXX: This is awful.
                    }
                }
            }
        }
    }

    for ebml::tagged_docs(inner_paths, tag_paths_data_item) |path_doc| {
        g(cdata, items, path_doc, broken, f);
    }

    for ebml::tagged_docs(inner_paths, tag_paths_foreign_path) |path_doc| {
        g(cdata, items, path_doc, broken, f);
    }
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
    debug!{"Looking up item: %d", id};
    let item_doc = lookup_item(id, cdata.data);
    let path = vec::init(item_path(item_doc));
    match decode_inlined_item(cdata, tcx, path, item_doc) {
      some(ii) => csearch::found(ii),
      none => {
        match item_parent_item(item_doc) {
          some(did) => {
            let did = translate_def_id(cdata, did);
            let parent_item = lookup_item(did.node, cdata.data);
            match decode_inlined_item(cdata, tcx, path,
                                               parent_item) {
              some(ii) => csearch::found_parent(did, ii),
              none => csearch::not_found
            }
          }
          none => csearch::not_found
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
        match ty::get(ctor_ty).struct {
          ty::ty_fn(f) => {
            for f.inputs.each |a| { vec::push(arg_tys, a.ty); }
          }
          _ => { /* Nullary enum variant. */ }
        }
        match variant_disr_val(item) {
          some(val) => { disr_val = val; }
          _         => { /* empty */ }
        }
        vec::push(infos, @{args: arg_tys, ctor_ty: ctor_ty, name: name,
                           id: did, disr_val: disr_val});
        disr_val += 1;
    }
    return infos;
}

// NB: These types are duplicated in resolve.rs
type method_info = {
    did: ast::def_id,
    n_tps: uint,
    ident: ast::ident,
    self_type: ast::self_ty_
};

type _impl = {did: ast::def_id, ident: ast::ident, methods: ~[@method_info]};

fn get_self_ty(item: ebml::doc) -> ast::self_ty_ {
    fn get_mutability(ch: u8) -> ast::mutability {
        match ch as char {
            'i' => { ast::m_imm }
            'm' => { ast::m_mutbl }
            'c' => { ast::m_const }
            _ => {
                fail fmt!{"unknown mutability character: `%c`", ch as char}
            }
        }
    }

    let self_type_doc = ebml::get_doc(item, tag_item_trait_method_self_ty);
    let string = ebml::doc_as_str(self_type_doc);

    let self_ty_kind = string[0];
    match self_ty_kind as char {
        's' => { return ast::sty_static; }
        'r' => { return ast::sty_by_ref; }
        'v' => { return ast::sty_value; }
        '@' => { return ast::sty_box(get_mutability(string[1])); }
        '~' => { return ast::sty_uniq(get_mutability(string[1])); }
        '&' => { return ast::sty_region(get_mutability(string[1])); }
        _ => {
            fail fmt!{"unknown self type code: `%c`", self_ty_kind as char};
        }
    }
}

fn item_impl_methods(cdata: cmd, item: ebml::doc, base_tps: uint)
    -> ~[@method_info] {
    let mut rslt = ~[];
    for ebml::tagged_docs(item, tag_item_impl_method) |doc| {
        let m_did = ebml::with_doc_data(doc, |d| parse_def_id(d));
        let mth_item = lookup_item(m_did.node, cdata.data);
        let self_ty = get_self_ty(mth_item);
        vec::push(rslt, @{did: translate_def_id(cdata, m_did),
                    /* FIXME (maybe #2323) tjc: take a look at this. */
                   n_tps: item_ty_param_count(mth_item) - base_tps,
                   ident: item_name(mth_item),
                   self_type: self_ty});
    }
    rslt
}

fn get_impls_for_mod(cdata: cmd,
                     m_id: ast::node_id,
                     name: option<ast::ident>,
                     get_cdata: fn(ast::crate_num) -> cmd)
                  -> @~[@_impl] {

    let data = cdata.data;
    let mod_item = lookup_item(m_id, data);
    let mut result = ~[];
    for ebml::tagged_docs(mod_item, tag_mod_impl) |doc| {
        let did = ebml::with_doc_data(doc, |d| parse_def_id(d));
        let local_did = translate_def_id(cdata, did);
        debug!{"(get impls for mod) getting did %? for '%?'",
               local_did, name};
          // The impl may be defined in a different crate. Ask the caller
          // to give us the metadata
        let impl_cdata = get_cdata(local_did.crate);
        let impl_data = impl_cdata.data;
        let item = lookup_item(local_did.node, impl_data);
        let nm = item_name(item);
        if match name { some(n) => { n == nm } none => { true } } {
           let base_tps = item_ty_param_count(item);
           vec::push(result, @{
                did: local_did, ident: nm,
                methods: item_impl_methods(impl_cdata, item, base_tps)
            });
        };
    }
    @result
}

/* Works for both classes and traits */
fn get_trait_methods(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> @~[ty::method] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    for ebml::tagged_docs(item, tag_item_trait_method) |mth| {
        let bounds = item_ty_param_bounds(mth, tcx, cdata);
        let name = item_name(mth);
        let ty = doc_type(mth, tcx, cdata);
        let fty = match ty::get(ty).struct {
          ty::ty_fn(f) => f,
          _ => {
            tcx.diag.handler().bug(
                ~"get_trait_methods: id has non-function type");
        } };
        let self_ty = get_self_ty(mth);
        vec::push(result, {ident: name, tps: bounds, fty: fty,
                    self_ty: self_ty,
                    purity: match check item_family(mth) {
                      'u' => ast::unsafe_fn,
                      'f' => ast::impure_fn,
                      'p' => ast::pure_fn
                    }, vis: ast::public});
    }
    @result
}

// If the item in question is a trait, returns its set of methods and
// their self types. Otherwise, returns none. This overlaps in an
// annoying way with get_trait_methods.
fn get_method_names_if_trait(cdata: cmd, node_id: ast::node_id)
                          -> option<@DVec<(@~str, ast::self_ty_)>> {

    let item = lookup_item(node_id, cdata.data);
    if item_family(item) != 'I' {
        return none;
    }

    let resulting_methods = @dvec();
    for ebml::tagged_docs(item, tag_item_trait_method) |method| {
        resulting_methods.push(
            (item_name(method), get_self_ty(method)));
    }
    return some(resulting_methods);
}

fn get_item_attrs(cdata: cmd,
                  node_id: ast::node_id,
                  f: fn(~[@ast::meta_item])) {

    let item = lookup_item(node_id, cdata.data);
    for ebml::tagged_docs(item, tag_attributes) |attributes| {
        for ebml::tagged_docs(attributes, tag_attribute) |attribute| {
            f(get_meta_items(attribute));
        }
    }
}

// Helper function that gets either fields or methods
fn get_class_members(cdata: cmd, id: ast::node_id,
                     p: fn(char) -> bool) -> ~[ty::field_ty] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    for ebml::tagged_docs(item, tag_item_field) |an_item| {
       let f = item_family(an_item);
       if p(f) {
          let name = item_name(an_item);
          let did = item_def_id(an_item, cdata);
          let mt = field_mutability(an_item);
          vec::push(result, {ident: name, id: did, vis:
                  family_to_visibility(f), mutability: mt});
       }
    }
    result
}

pure fn family_to_visibility(family: char) -> ast::visibility {
    match family {
      'g' => ast::public,
      'j' => ast::private,
      'N' => ast::inherited,
      _ => fail
    }
}

/* 'g' for public field, 'j' for private field, 'N' for inherited field */
fn get_class_fields(cdata: cmd, id: ast::node_id) -> ~[ty::field_ty] {
    get_class_members(cdata, id, |f| f == 'g' || f == 'j' || f == 'N')
}

fn family_has_type_params(fam_ch: char) -> bool {
    match fam_ch {
      'c' | 'T' | 'm' | 'n' | 'g' | 'h' | 'j' | 'e' | 'N' => false,
      'f' | 'u' | 'p' | 'F' | 'U' | 'P' | 'y' | 't' | 'v' | 'i' | 'I' | 'C'
          | 'a' | 'S'
          => true,
      _ => fail fmt!("'%c' is not a family", fam_ch)
    }
}

fn family_names_type(fam_ch: char) -> bool {
    match fam_ch { 'y' | 't' | 'I' => true, _ => false }
}

fn read_path(d: ebml::doc) -> {path: ~str, pos: uint} {
    let desc = ebml::doc_data(d);
    let pos = io::u64_from_be_bytes(desc, 0u, 4u) as uint;
    let pathbytes = vec::slice::<u8>(desc, 4u, vec::len::<u8>(desc));
    let path = str::from_bytes(pathbytes);
    return {path: path, pos: pos};
}

fn describe_def(items: ebml::doc, id: ast::def_id) -> ~str {
    if id.crate != ast::local_crate { return ~"external"; }
    let it = match maybe_find_item(id.node, items) {
        some(it) => it,
        none => fail (fmt!{"describe_def: item not found %?", id})
    };
    return item_family_to_str(item_family(it));
}

fn item_family_to_str(fam: char) -> ~str {
    match check fam {
      'c' => return ~"const",
      'f' => return ~"fn",
      'u' => return ~"unsafe fn",
      'p' => return ~"pure fn",
      'F' => return ~"static method",
      'U' => return ~"unsafe static method",
      'P' => return ~"pure static method",
      'e' => return ~"foreign fn",
      'y' => return ~"type",
      'T' => return ~"foreign type",
      't' => return ~"type",
      'm' => return ~"mod",
      'n' => return ~"foreign mod",
      'v' => return ~"enum",
      'i' => return ~"impl",
      'I' => return ~"trait",
      'C' => return ~"class",
      'S' => return ~"struct",
      'g' => return ~"public field",
      'j' => return ~"private field",
      'N' => return ~"inherited field"
    }
}

fn get_meta_items(md: ebml::doc) -> ~[@ast::meta_item] {
    let mut items: ~[@ast::meta_item] = ~[];
    for ebml::tagged_docs(md, tag_meta_item_word) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(ebml::doc_data(nd));
        vec::push(items, attr::mk_word_item(@n));
    };
    for ebml::tagged_docs(md, tag_meta_item_name_value) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = ebml::get_doc(meta_item_doc, tag_meta_item_value);
        let n = str::from_bytes(ebml::doc_data(nd));
        let v = str::from_bytes(ebml::doc_data(vd));
        // FIXME (#623): Should be able to decode meta_name_value variants,
        // but currently the encoder just drops them
        vec::push(items, attr::mk_name_value_item_str(@n, v));
    };
    for ebml::tagged_docs(md, tag_meta_item_list) |meta_item_doc| {
        let nd = ebml::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(ebml::doc_data(nd));
        let subitems = get_meta_items(meta_item_doc);
        vec::push(items, attr::mk_list_item(@n, subitems));
    };
    return items;
}

fn get_attributes(md: ebml::doc) -> ~[ast::attribute] {
    let mut attrs: ~[ast::attribute] = ~[];
    match ebml::maybe_get_doc(md, tag_attributes) {
      option::some(attrs_d) => {
        for ebml::tagged_docs(attrs_d, tag_attribute) |attr_doc| {
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert (vec::len(meta_items) == 1u);
            let meta_item = meta_items[0];
            vec::push(attrs,
                      {node: {style: ast::attr_outer, value: *meta_item,
                              is_sugared_doc: false},
                       span: ast_util::dummy_sp()});
        };
      }
      option::none => ()
    }
    return attrs;
}

fn list_meta_items(meta_items: ebml::doc, out: io::Writer) {
    for get_meta_items(meta_items).each |mi| {
        out.write_str(fmt!{"%s\n", pprust::meta_item_to_str(*mi)});
    }
}

fn list_crate_attributes(md: ebml::doc, hash: @~str, out: io::Writer) {
    out.write_str(fmt!{"=Crate Attributes (%s)=\n", *hash});

    for get_attributes(md).each |attr| {
        out.write_str(fmt!{"%s\n", pprust::attribute_to_str(attr)});
    }

    out.write_str(~"\n\n");
}

fn get_crate_attributes(data: @~[u8]) -> ~[ast::attribute] {
    return get_attributes(ebml::doc(data));
}

type crate_dep = {cnum: ast::crate_num, name: ast::ident,
                  vers: @~str, hash: @~str};

fn get_crate_deps(data: @~[u8]) -> ~[crate_dep] {
    let mut deps: ~[crate_dep] = ~[];
    let cratedoc = ebml::doc(data);
    let depsdoc = ebml::get_doc(cratedoc, tag_crate_deps);
    let mut crate_num = 1;
    fn docstr(doc: ebml::doc, tag_: uint) -> ~str {
        str::from_bytes(ebml::doc_data(ebml::get_doc(doc, tag_)))
    }
    for ebml::tagged_docs(depsdoc, tag_crate_dep) |depdoc| {
        vec::push(deps, {cnum: crate_num,
                  name: @docstr(depdoc, tag_crate_dep_name),
                  vers: @docstr(depdoc, tag_crate_dep_vers),
                  hash: @docstr(depdoc, tag_crate_dep_hash)});
        crate_num += 1;
    };
    return deps;
}

fn list_crate_deps(data: @~[u8], out: io::Writer) {
    out.write_str(~"=External Dependencies=\n");

    for get_crate_deps(data).each |dep| {
        out.write_str(fmt!{"%d %s-%s-%s\n",
                           dep.cnum, *dep.name, *dep.hash, *dep.vers});
    }

    out.write_str(~"\n");
}

fn get_crate_hash(data: @~[u8]) -> @~str {
    let cratedoc = ebml::doc(data);
    let hashdoc = ebml::get_doc(cratedoc, tag_crate_hash);
    return @str::from_bytes(ebml::doc_data(hashdoc));
}

fn get_crate_vers(data: @~[u8]) -> @~str {
    let attrs = decoder::get_crate_attributes(data);
    return match attr::last_meta_item_value_str_by_name(
        attr::find_linkage_metas(attrs), ~"vers") {
      some(ver) => ver,
      none => @~"0.0"
    };
}

fn list_crate_items(bytes: @~[u8], md: ebml::doc, out: io::Writer) {
    out.write_str(~"=Items=\n");
    let items = ebml::get_doc(md, tag_items);
    do iter_crate_items(bytes) |tag, path, did| {
      // Don't print out any metadata info about intrinsics
       if tag != tag_paths_foreign_path {
            out.write_str(fmt!{"%s (%s)\n", path,
                               describe_def(items, did)});
       }
    }
    out.write_str(~"\n");
}

fn iter_crate_items(bytes: @~[u8], proc: fn(uint, ~str, ast::def_id)) {
    let md = ebml::doc(bytes);
    let paths = ebml::get_doc(md, tag_paths);
    let index = ebml::get_doc(paths, tag_index);
    let bs = ebml::get_doc(index, tag_index_buckets);
    for ebml::tagged_docs(bs, tag_index_buckets_bucket) |bucket| {
        let et = tag_index_buckets_bucket_elt;
        for ebml::tagged_docs(bucket, et) |elt| {
            let data = read_path(elt);
            let {tag:t, doc:def} = ebml::doc_at(bytes, data.pos);
            let did_doc = ebml::get_doc(def, tag_def_id);
            let did = ebml::with_doc_data(did_doc, |d| parse_def_id(d));
            proc(t, data.path, did);
        };
    };
}

fn get_crate_module_paths(bytes: @~[u8]) -> ~[(ast::def_id, ~str)] {
    fn mod_of_path(p: ~str) -> ~str {
        str::connect(vec::init(str::split_str(p, ~"::")), ~"::")
    }

    // find all module (path, def_ids), which are not
    // fowarded path due to renamed import or reexport
    let mut res = ~[];
    let mods = map::str_hash();
    do iter_crate_items(bytes) |_tag, path, did| {
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
    return do vec::filter(res) |x| {
        let (_, xp) = x;
        mods.contains_key(xp)
    }
}

fn list_crate_metadata(bytes: @~[u8], out: io::Writer) {
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
        return {crate: cdata.cnum, node: did.node};
    }

    match cdata.cnum_map.find(did.crate) {
      option::some(n) => return {crate: n, node: did.node},
      option::none => fail ~"didn't find a crate in the cnum_map"
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
