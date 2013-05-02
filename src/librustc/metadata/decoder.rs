// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Decoding metadata from a single crate's metadata

use metadata::cstore::crate_metadata;
use metadata::common::*;
use metadata::csearch::{ProvidedTraitMethodInfo, StaticMethodInfo};
use metadata::csearch;
use metadata::cstore;
use metadata::decoder;
use metadata::tydecode::{parse_ty_data, parse_def_id,
                         parse_type_param_def_data,
                         parse_bare_fn_ty_data, parse_trait_ref_data};
use middle::{ty, resolve};

use core::hash::HashUtil;
use core::int;
use core::io::WriterUtil;
use core::io;
use core::option;
use core::str;
use core::vec;
use std::ebml::reader;
use std::ebml;
use std::serialize::Decodable;
use syntax::ast_map;
use syntax::attr;
use syntax::diagnostic::span_handler;
use syntax::parse::token::{StringRef, ident_interner, special_idents};
use syntax::print::pprust;
use syntax::{ast, ast_util};
use syntax::codemap;

type cmd = @crate_metadata;

// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.

fn lookup_hash(d: ebml::Doc, eq_fn: &fn(x:&[u8]) -> bool, hash: uint) ->
   Option<ebml::Doc> {
    let index = reader::get_doc(d, tag_index);
    let table = reader::get_doc(index, tag_index_table);
    let hash_pos = table.start + hash % 256u * 4u;
    let pos = io::u64_from_be_bytes(*d.data, hash_pos, 4u) as uint;
    let tagged_doc = reader::doc_at(d.data, pos);

    let belt = tag_index_buckets_bucket_elt;
    for reader::tagged_docs(tagged_doc.doc, belt) |elt| {
        let pos = io::u64_from_be_bytes(*elt.data, elt.start, 4u) as uint;
        if eq_fn(vec::slice(*elt.data, elt.start + 4u, elt.end)) {
            return Some(reader::doc_at(d.data, pos).doc);
        }
    };
    None
}

pub type GetCrateDataCb<'self> = &'self fn(ast::crate_num) -> cmd;

pub fn maybe_find_item(item_id: int, items: ebml::Doc) -> Option<ebml::Doc> {
    fn eq_item(bytes: &[u8], item_id: int) -> bool {
        return io::u64_from_be_bytes(
            vec::slice(bytes, 0u, 4u), 0u, 4u) as int
            == item_id;
    }
    lookup_hash(items,
                |a| eq_item(a, item_id),
                item_id.hash() as uint)
}

fn find_item(item_id: int, items: ebml::Doc) -> ebml::Doc {
    return maybe_find_item(item_id, items).get();
}

// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item(item_id: int, data: @~[u8]) -> ebml::Doc {
    let items = reader::get_doc(reader::Doc(data), tag_items);
    match maybe_find_item(item_id, items) {
       None => fail!("lookup_item: id not found: %d", item_id),
       Some(d) => d
    }
}

#[deriving(Eq)]
enum Family {
    Const,                 // c
    Fn,                    // f
    UnsafeFn,              // u
    PureFn,                // p
    StaticMethod,          // F
    UnsafeStaticMethod,    // U
    PureStaticMethod,      // P
    ForeignFn,             // e
    Type,                  // y
    ForeignType,           // T
    Mod,                   // m
    ForeignMod,            // n
    Enum,                  // t
    Variant,               // v
    Impl,                  // i
    Trait,                 // I
    Struct,                // S
    PublicField,           // g
    PrivateField,          // j
    InheritedField         // N
}

fn item_family(item: ebml::Doc) -> Family {
    let fam = reader::get_doc(item, tag_items_data_item_family);
    match reader::doc_as_u8(fam) as char {
      'c' => Const,
      'f' => Fn,
      'u' => UnsafeFn,
      'p' => PureFn,
      'F' => StaticMethod,
      'U' => UnsafeStaticMethod,
      'P' => PureStaticMethod,
      'e' => ForeignFn,
      'y' => Type,
      'T' => ForeignType,
      'm' => Mod,
      'n' => ForeignMod,
      't' => Enum,
      'v' => Variant,
      'i' => Impl,
      'I' => Trait,
      'S' => Struct,
      'g' => PublicField,
      'j' => PrivateField,
      'N' => InheritedField,
       c => fail!("unexpected family char: %c", c)
    }
}

fn item_visibility(item: ebml::Doc) -> ast::visibility {
    match reader::maybe_get_doc(item, tag_items_data_item_visibility) {
        None => ast::public,
        Some(visibility_doc) => {
            match reader::doc_as_u8(visibility_doc) as char {
                'y' => ast::public,
                'n' => ast::private,
                'i' => ast::inherited,
                _ => fail!("unknown visibility character")
            }
        }
    }
}

fn item_method_sort(item: ebml::Doc) -> char {
    for reader::tagged_docs(item, tag_item_trait_method_sort) |doc| {
        return str::from_bytes(reader::doc_data(doc))[0] as char;
    }
    return 'r';
}

fn item_symbol(item: ebml::Doc) -> ~str {
    let sym = reader::get_doc(item, tag_items_data_item_symbol);
    return str::from_bytes(reader::doc_data(sym));
}

fn item_parent_item(d: ebml::Doc) -> Option<ast::def_id> {
    for reader::tagged_docs(d, tag_items_data_parent_item) |did| {
        return Some(reader::with_doc_data(did, |d| parse_def_id(d)));
    }
    None
}

fn translated_parent_item_opt(cnum: ast::crate_num, d: ebml::Doc) ->
        Option<ast::def_id> {
    let trait_did_opt = item_parent_item(d);
    do trait_did_opt.map |trait_did| {
        ast::def_id { crate: cnum, node: trait_did.node }
    }
}

fn item_reqd_and_translated_parent_item(cnum: ast::crate_num,
                                        d: ebml::Doc) -> ast::def_id {
    let trait_did = item_parent_item(d).expect(~"item without parent");
    ast::def_id { crate: cnum, node: trait_did.node }
}

fn item_def_id(d: ebml::Doc, cdata: cmd) -> ast::def_id {
    let tagdoc = reader::get_doc(d, tag_def_id);
    return translate_def_id(cdata, reader::with_doc_data(tagdoc,
                                                    |d| parse_def_id(d)));
}

#[cfg(stage0)]
fn each_reexport(d: ebml::Doc, f: &fn(ebml::Doc) -> bool) {
    for reader::tagged_docs(d, tag_items_data_item_reexport) |reexport_doc| {
        if !f(reexport_doc) {
            return;
        }
    }
}
#[cfg(not(stage0))]
fn each_reexport(d: ebml::Doc, f: &fn(ebml::Doc) -> bool) -> bool {
    for reader::tagged_docs(d, tag_items_data_item_reexport) |reexport_doc| {
        if !f(reexport_doc) {
            return false;
        }
    }
    return true;
}

fn variant_disr_val(d: ebml::Doc) -> Option<int> {
    do reader::maybe_get_doc(d, tag_disr_val).chain |val_doc| {
        int::parse_bytes(reader::doc_data(val_doc), 10u)
    }
}

fn doc_type(doc: ebml::Doc, tcx: ty::ctxt, cdata: cmd) -> ty::t {
    let tp = reader::get_doc(doc, tag_items_data_item_type);
    parse_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                  |_, did| translate_def_id(cdata, did))
}

fn doc_method_fty(doc: ebml::Doc, tcx: ty::ctxt, cdata: cmd) -> ty::BareFnTy {
    let tp = reader::get_doc(doc, tag_item_method_fty);
    parse_bare_fn_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                          |_, did| translate_def_id(cdata, did))
}

fn doc_transformed_self_ty(doc: ebml::Doc,
                           tcx: ty::ctxt,
                           cdata: cmd) -> Option<ty::t>
{
    do reader::maybe_get_doc(doc, tag_item_method_transformed_self_ty).map |tp| {
        parse_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                      |_, did| translate_def_id(cdata, did))
    }
}

pub fn item_type(_item_id: ast::def_id, item: ebml::Doc,
                 tcx: ty::ctxt, cdata: cmd) -> ty::t {
    doc_type(item, tcx, cdata)
}

fn doc_trait_ref(doc: ebml::Doc, tcx: ty::ctxt, cdata: cmd) -> ty::TraitRef {
    parse_trait_ref_data(doc.data, cdata.cnum, doc.start, tcx,
                         |_, did| translate_def_id(cdata, did))
}

fn item_trait_ref(doc: ebml::Doc, tcx: ty::ctxt, cdata: cmd) -> ty::TraitRef {
    let tp = reader::get_doc(doc, tag_item_trait_ref);
    doc_trait_ref(tp, tcx, cdata)
}

fn item_ty_param_defs(item: ebml::Doc, tcx: ty::ctxt, cdata: cmd,
                      tag: uint)
    -> @~[ty::TypeParameterDef] {
    let mut bounds = ~[];
    for reader::tagged_docs(item, tag) |p| {
        let bd = parse_type_param_def_data(
            p.data, p.start, cdata.cnum, tcx,
            |_, did| translate_def_id(cdata, did));
        bounds.push(bd);
    }
    @bounds
}

fn item_ty_region_param(item: ebml::Doc) -> Option<ty::region_variance> {
    reader::maybe_get_doc(item, tag_region_param).map(|doc| {
        let mut decoder = reader::Decoder(*doc);
        Decodable::decode(&mut decoder)
    })
}

fn item_ty_param_count(item: ebml::Doc) -> uint {
    let mut n = 0u;
    reader::tagged_docs(item, tag_items_data_item_ty_param_bounds,
                      |_p| { n += 1u; true } );
    n
}

fn enum_variant_ids(item: ebml::Doc, cdata: cmd) -> ~[ast::def_id] {
    let mut ids: ~[ast::def_id] = ~[];
    let v = tag_items_data_item_variant;
    for reader::tagged_docs(item, v) |p| {
        let ext = reader::with_doc_data(p, |d| parse_def_id(d));
        ids.push(ast::def_id { crate: cdata.cnum, node: ext.node });
    };
    return ids;
}

fn item_path(intr: @ident_interner, item_doc: ebml::Doc) -> ast_map::path {
    let path_doc = reader::get_doc(item_doc, tag_path);

    let len_doc = reader::get_doc(path_doc, tag_path_len);
    let len = reader::doc_as_u32(len_doc) as uint;

    let mut result = vec::with_capacity(len);
    for reader::docs(path_doc) |tag, elt_doc| {
        if tag == tag_path_elt_mod {
            let str = reader::doc_as_str(elt_doc);
            result.push(ast_map::path_mod(intr.intern(str)));
        } else if tag == tag_path_elt_name {
            let str = reader::doc_as_str(elt_doc);
            result.push(ast_map::path_name(intr.intern(str)));
        } else {
            // ignore tag_path_len element
        }
    }

    return result;
}

fn item_name(intr: @ident_interner, item: ebml::Doc) -> ast::ident {
    let name = reader::get_doc(item, tag_paths_data_name);
    do reader::with_doc_data(name) |data| {
        let string = str::from_bytes_slice(data);
        match intr.find_equiv(&StringRef(string)) {
            None => intr.intern(string),
            Some(val) => val,
        }
    }
}

fn item_to_def_like(item: ebml::Doc, did: ast::def_id, cnum: ast::crate_num)
    -> def_like
{
    let fam = item_family(item);
    match fam {
        Const     => dl_def(ast::def_const(did)),
        Struct    => dl_def(ast::def_struct(did)),
        UnsafeFn  => dl_def(ast::def_fn(did, ast::unsafe_fn)),
        Fn        => dl_def(ast::def_fn(did, ast::impure_fn)),
        PureFn    => dl_def(ast::def_fn(did, ast::pure_fn)),
        ForeignFn => dl_def(ast::def_fn(did, ast::extern_fn)),
        UnsafeStaticMethod => {
            let trait_did_opt = translated_parent_item_opt(cnum, item);
            dl_def(ast::def_static_method(did, trait_did_opt, ast::unsafe_fn))
        }
        StaticMethod => {
            let trait_did_opt = translated_parent_item_opt(cnum, item);
            dl_def(ast::def_static_method(did, trait_did_opt, ast::impure_fn))
        }
        PureStaticMethod => {
            let trait_did_opt = translated_parent_item_opt(cnum, item);
            dl_def(ast::def_static_method(did, trait_did_opt, ast::pure_fn))
        }
        Type | ForeignType => dl_def(ast::def_ty(did)),
        Mod => dl_def(ast::def_mod(did)),
        ForeignMod => dl_def(ast::def_foreign_mod(did)),
        Variant => {
            let enum_did = item_reqd_and_translated_parent_item(cnum, item);
            dl_def(ast::def_variant(enum_did, did))
        }
        Trait => dl_def(ast::def_trait(did)),
        Enum => dl_def(ast::def_ty(did)),
        Impl => dl_impl(did),
        PublicField | PrivateField | InheritedField => dl_field,
    }
}

pub fn lookup_def(cnum: ast::crate_num, data: @~[u8], did_: ast::def_id) ->
   ast::def {
    let item = lookup_item(did_.node, data);
    let did = ast::def_id { crate: cnum, node: did_.node };
    // We treat references to enums as references to types.
    return def_like_to_def(item_to_def_like(item, did, cnum));
}

pub fn get_trait_def(cdata: cmd,
                     item_id: ast::node_id,
                     tcx: ty::ctxt) -> ty::TraitDef
{
    let item_doc = lookup_item(item_id, cdata.data);
    let tp_defs = item_ty_param_defs(item_doc, tcx, cdata,
                                     tag_items_data_item_ty_param_bounds);
    let rp = item_ty_region_param(item_doc);
    ty::TraitDef {
        generics: ty::Generics {type_param_defs: tp_defs,
                                region_param: rp},
        trait_ref: @item_trait_ref(item_doc, tcx, cdata)
    }
}

pub fn get_type(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
    -> ty::ty_param_bounds_and_ty {

    let item = lookup_item(id, cdata.data);
    let t = item_type(ast::def_id { crate: cdata.cnum, node: id }, item, tcx,
                      cdata);
    let tp_defs = if family_has_type_params(item_family(item)) {
        item_ty_param_defs(item, tcx, cdata, tag_items_data_item_ty_param_bounds)
    } else { @~[] };
    let rp = item_ty_region_param(item);
    ty::ty_param_bounds_and_ty {
        generics: ty::Generics {type_param_defs: tp_defs,
                                region_param: rp},
        ty: t
    }
}

pub fn get_region_param(cdata: cmd, id: ast::node_id)
    -> Option<ty::region_variance> {

    let item = lookup_item(id, cdata.data);
    return item_ty_region_param(item);
}

pub fn get_type_param_count(data: @~[u8], id: ast::node_id) -> uint {
    item_ty_param_count(lookup_item(id, data))
}

pub fn get_impl_traits(cdata: cmd,
                       id: ast::node_id,
                       tcx: ty::ctxt) -> ~[@ty::TraitRef]
{
    let item_doc = lookup_item(id, cdata.data);
    let mut results = ~[];
    for reader::tagged_docs(item_doc, tag_item_trait_ref) |tp| {
        let trait_ref =
            @parse_trait_ref_data(tp.data, cdata.cnum, tp.start, tcx,
                                  |_, did| translate_def_id(cdata, did));
        results.push(trait_ref);
    };
    results
}

pub fn get_impl_method(intr: @ident_interner, cdata: cmd, id: ast::node_id,
                       name: ast::ident) -> ast::def_id {
    let items = reader::get_doc(reader::Doc(cdata.data), tag_items);
    let mut found = None;
    for reader::tagged_docs(find_item(id, items), tag_item_impl_method)
        |mid| {
            let m_did = reader::with_doc_data(mid, |d| parse_def_id(d));
            if item_name(intr, find_item(m_did.node, items)) == name {
                found = Some(translate_def_id(cdata, m_did));
            }
        }
    found.get()
}

pub fn get_symbol(data: @~[u8], id: ast::node_id) -> ~str {
    return item_symbol(lookup_item(id, data));
}

// Something that a name can resolve to.
pub enum def_like {
    dl_def(ast::def),
    dl_impl(ast::def_id),
    dl_field
}

fn def_like_to_def(def_like: def_like) -> ast::def {
    match def_like {
        dl_def(def) => return def,
        dl_impl(*) => fail!("found impl in def_like_to_def"),
        dl_field => fail!("found field in def_like_to_def")
    }
}

/// Iterates over the language items in the given crate.
#[cfg(stage0)]
pub fn each_lang_item(cdata: cmd, f: &fn(ast::node_id, uint) -> bool) {
    let root = reader::Doc(cdata.data);
    let lang_items = reader::get_doc(root, tag_lang_items);
    for reader::tagged_docs(lang_items, tag_lang_items_item) |item_doc| {
        let id_doc = reader::get_doc(item_doc, tag_lang_items_item_id);
        let id = reader::doc_as_u32(id_doc) as uint;
        let node_id_doc = reader::get_doc(item_doc,
                                          tag_lang_items_item_node_id);
        let node_id = reader::doc_as_u32(node_id_doc) as ast::node_id;

        if !f(node_id, id) {
            break;
        }
    }
}
/// Iterates over the language items in the given crate.
#[cfg(not(stage0))]
pub fn each_lang_item(cdata: cmd, f: &fn(ast::node_id, uint) -> bool) -> bool {
    let root = reader::Doc(cdata.data);
    let lang_items = reader::get_doc(root, tag_lang_items);
    for reader::tagged_docs(lang_items, tag_lang_items_item) |item_doc| {
        let id_doc = reader::get_doc(item_doc, tag_lang_items_item_id);
        let id = reader::doc_as_u32(id_doc) as uint;
        let node_id_doc = reader::get_doc(item_doc,
                                          tag_lang_items_item_node_id);
        let node_id = reader::doc_as_u32(node_id_doc) as ast::node_id;

        if !f(node_id, id) {
            return false;
        }
    }
    return true;
}

/// Iterates over all the paths in the given crate.
pub fn _each_path(intr: @ident_interner, cdata: cmd,
                  get_crate_data: GetCrateDataCb,
                  f: &fn(&str, def_like) -> bool) -> bool {
    let root = reader::Doc(cdata.data);
    let items = reader::get_doc(root, tag_items);
    let items_data = reader::get_doc(items, tag_items_data);

    let mut broken = false;

    // First, go through all the explicit items.
    for reader::tagged_docs(items_data, tag_items_data_item) |item_doc| {
        if !broken {
            let path = ast_map::path_to_str_with_sep(
                item_path(intr, item_doc), ~"::", intr);
            let path_is_empty = path.is_empty();
            if !path_is_empty {
                // Extract the def ID.
                let def_id = item_def_id(item_doc, cdata);

                // Construct the def for this item.
                debug!("(each_path) yielding explicit item: %s", path);
                let def_like = item_to_def_like(item_doc, def_id, cdata.cnum);

                // Hand the information off to the iteratee.
                if !f(path, def_like) {
                    broken = true;      // FIXME #4572: This is awful.
                }
            }

            // If this is a module, find the reexports.
            for each_reexport(item_doc) |reexport_doc| {
                if !broken {
                    let def_id_doc =
                        reader::get_doc(reexport_doc,
                            tag_items_data_item_reexport_def_id);
                    let def_id =
                        reader::with_doc_data(def_id_doc,
                                              |d| parse_def_id(d));
                    let def_id = translate_def_id(cdata, def_id);

                    let reexport_name_doc =
                        reader::get_doc(reexport_doc,
                                      tag_items_data_item_reexport_name);
                    let reexport_name = reader::doc_as_str(reexport_name_doc);

                    let reexport_path;
                    if path_is_empty {
                        reexport_path = reexport_name;
                    } else {
                        reexport_path = path + ~"::" + reexport_name;
                    }

                    // This reexport may be in yet another crate
                    let other_crates_items = if def_id.crate == cdata.cnum {
                        items
                    } else {
                        let crate_data = get_crate_data(def_id.crate);
                        let root = reader::Doc(crate_data.data);
                        reader::get_doc(root, tag_items)
                    };

                    // Get the item.
                    match maybe_find_item(def_id.node, other_crates_items) {
                        None => {}
                        Some(item_doc) => {
                            // Construct the def for this item.
                            let def_like = item_to_def_like(item_doc,
                                                            def_id,
                                                            cdata.cnum);

                            // Hand the information off to the iteratee.
                            debug!("(each_path) yielding reexported \
                                    item: %s", reexport_path);

                            if (!f(reexport_path, def_like)) {
                                broken = true;  // FIXME #4572: This is awful.
                            }
                        }
                    }
                }
            }
        }
    }

    return broken;
}

#[cfg(stage0)]
pub fn each_path(intr: @ident_interner, cdata: cmd,
                 get_crate_data: GetCrateDataCb,
                 f: &fn(&str, def_like) -> bool) {
    _each_path(intr, cdata, get_crate_data, f);
}
#[cfg(not(stage0))]
pub fn each_path(intr: @ident_interner, cdata: cmd,
                 get_crate_data: GetCrateDataCb,
                 f: &fn(&str, def_like) -> bool) -> bool {
    _each_path(intr, cdata, get_crate_data, f)
}

pub fn get_item_path(intr: @ident_interner, cdata: cmd, id: ast::node_id)
    -> ast_map::path {
    item_path(intr, lookup_item(id, cdata.data))
}

pub type decode_inlined_item<'self> = &'self fn(
    cdata: @cstore::crate_metadata,
    tcx: ty::ctxt,
    path: ast_map::path,
    par_doc: ebml::Doc) -> Option<ast::inlined_item>;

pub fn maybe_get_item_ast(intr: @ident_interner, cdata: cmd, tcx: ty::ctxt,
                          id: ast::node_id,
                          decode_inlined_item: decode_inlined_item)
                       -> csearch::found_ast {
    debug!("Looking up item: %d", id);
    let item_doc = lookup_item(id, cdata.data);
    let path = {
        let item_path = item_path(intr, item_doc);
        vec::to_owned(item_path.init())
    };
    match decode_inlined_item(cdata, tcx, copy path, item_doc) {
      Some(ref ii) => csearch::found((/*bad*/copy *ii)),
      None => {
        match item_parent_item(item_doc) {
          Some(did) => {
            let did = translate_def_id(cdata, did);
            let parent_item = lookup_item(did.node, cdata.data);
            match decode_inlined_item(cdata, tcx, path, parent_item) {
              Some(ref ii) => csearch::found_parent(did, (/*bad*/copy *ii)),
              None => csearch::not_found
            }
          }
          None => csearch::not_found
        }
      }
    }
}

pub fn get_enum_variants(intr: @ident_interner, cdata: cmd, id: ast::node_id,
                     tcx: ty::ctxt) -> ~[ty::VariantInfo] {
    let data = cdata.data;
    let items = reader::get_doc(reader::Doc(data), tag_items);
    let item = find_item(id, items);
    let mut infos: ~[ty::VariantInfo] = ~[];
    let variant_ids = enum_variant_ids(item, cdata);
    let mut disr_val = 0;
    for variant_ids.each |did| {
        let item = find_item(did.node, items);
        let ctor_ty = item_type(ast::def_id { crate: cdata.cnum, node: id},
                                item, tcx, cdata);
        let name = item_name(intr, item);
        let arg_tys = match ty::get(ctor_ty).sty {
          ty::ty_bare_fn(ref f) => copy f.sig.inputs,
          _ => ~[], // Nullary enum variant.
        };
        match variant_disr_val(item) {
          Some(val) => { disr_val = val; }
          _         => { /* empty */ }
        }
        infos.push(@ty::VariantInfo_{args: arg_tys,
                       ctor_ty: ctor_ty, name: name,
                  // I'm not even sure if we encode visibility
                  // for variants -- TEST -- tjc
                  id: *did, disr_val: disr_val, vis: ast::inherited});
        disr_val += 1;
    }
    return infos;
}

fn get_explicit_self(item: ebml::Doc) -> ast::explicit_self_ {
    fn get_mutability(ch: u8) -> ast::mutability {
        match ch as char {
            'i' => { ast::m_imm }
            'm' => { ast::m_mutbl }
            'c' => { ast::m_const }
            _ => {
                fail!("unknown mutability character: `%c`", ch as char)
            }
        }
    }

    let explicit_self_doc = reader::get_doc(item, tag_item_trait_method_explicit_self);
    let string = reader::doc_as_str(explicit_self_doc);

    let explicit_self_kind = string[0];
    match explicit_self_kind as char {
        's' => { return ast::sty_static; }
        'v' => { return ast::sty_value; }
        '@' => { return ast::sty_box(get_mutability(string[1])); }
        '~' => { return ast::sty_uniq(get_mutability(string[1])); }
        '&' => {
            // FIXME(#4846) expl. region
            return ast::sty_region(None, get_mutability(string[1]));
        }
        _ => {
            fail!("unknown self type code: `%c`", explicit_self_kind as char);
        }
    }
}

fn item_impl_methods(intr: @ident_interner, cdata: cmd, item: ebml::Doc,
                     base_tps: uint) -> ~[@resolve::MethodInfo] {
    let mut rslt = ~[];
    for reader::tagged_docs(item, tag_item_impl_method) |doc| {
        let m_did = reader::with_doc_data(doc, |d| parse_def_id(d));
        let mth_item = lookup_item(m_did.node, cdata.data);
        let explicit_self = get_explicit_self(mth_item);
        rslt.push(@resolve::MethodInfo {
                    did: translate_def_id(cdata, m_did),
                    n_tps: item_ty_param_count(mth_item) - base_tps,
                    ident: item_name(intr, mth_item),
                    explicit_self: explicit_self});
    }
    rslt
}

pub fn get_impls_for_mod(intr: @ident_interner,
                         cdata: cmd,
                         m_id: ast::node_id,
                         name: Option<ast::ident>,
                         get_cdata: &fn(ast::crate_num) -> cmd)
                      -> @~[@resolve::Impl] {
    let data = cdata.data;
    let mod_item = lookup_item(m_id, data);
    let mut result = ~[];
    for reader::tagged_docs(mod_item, tag_mod_impl) |doc| {
        let did = reader::with_doc_data(doc, |d| parse_def_id(d));
        let local_did = translate_def_id(cdata, did);
        debug!("(get impls for mod) getting did %? for '%?'",
               local_did, name);
          // The impl may be defined in a different crate. Ask the caller
          // to give us the metadata
        let impl_cdata = get_cdata(local_did.crate);
        let impl_data = impl_cdata.data;
        let item = lookup_item(local_did.node, impl_data);
        let nm = item_name(intr, item);
        if match name { Some(n) => { n == nm } None => { true } } {
           let base_tps = item_ty_param_count(item);
           result.push(@resolve::Impl {
                did: local_did, ident: nm,
                methods: item_impl_methods(intr, impl_cdata, item, base_tps)
            });
        };
    }
    @result
}

pub fn get_method_name_and_explicit_self(
    intr: @ident_interner,
    cdata: cmd,
    id: ast::node_id) -> (ast::ident, ast::explicit_self_)
{
    let method_doc = lookup_item(id, cdata.data);
    let name = item_name(intr, method_doc);
    let explicit_self = get_explicit_self(method_doc);
    (name, explicit_self)
}

pub fn get_method(intr: @ident_interner, cdata: cmd, id: ast::node_id,
                  tcx: ty::ctxt) -> ty::Method
{
    let method_doc = lookup_item(id, cdata.data);
    let def_id = item_def_id(method_doc, cdata);
    let name = item_name(intr, method_doc);
    let type_param_defs = item_ty_param_defs(method_doc, tcx, cdata,
                                             tag_item_method_tps);
    let transformed_self_ty = doc_transformed_self_ty(method_doc, tcx, cdata);
    let fty = doc_method_fty(method_doc, tcx, cdata);
    let vis = item_visibility(method_doc);
    let explicit_self = get_explicit_self(method_doc);

    ty::Method::new(
        name,
        ty::Generics {
            type_param_defs: type_param_defs,
            region_param: None
        },
        transformed_self_ty,
        fty,
        explicit_self,
        vis,
        def_id
    )
}

pub fn get_trait_method_def_ids(cdata: cmd,
                                id: ast::node_id) -> ~[ast::def_id] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    for reader::tagged_docs(item, tag_item_trait_method) |mth| {
        result.push(item_def_id(mth, cdata));
    }
    result
}

pub fn get_provided_trait_methods(intr: @ident_interner, cdata: cmd,
                                  id: ast::node_id, tcx: ty::ctxt) ->
        ~[ProvidedTraitMethodInfo] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];

    for reader::tagged_docs(item, tag_item_trait_method) |mth| {
        if item_method_sort(mth) != 'p' { loop; }

        let did = item_def_id(mth, cdata);

        let type_param_defs =
            item_ty_param_defs(mth, tcx, cdata,
                               tag_items_data_item_ty_param_bounds);
        let name = item_name(intr, mth);
        let ty = doc_type(mth, tcx, cdata);

        let fty = match ty::get(ty).sty {
            ty::ty_bare_fn(ref f) => copy *f,
            _ => {
                tcx.diag.handler().bug(~"get_provided_trait_methods(): id \
                                         has non-function type");
            }
        };

        let transformed_self_ty = doc_transformed_self_ty(mth, tcx, cdata);
        let explicit_self = get_explicit_self(mth);

        let ty_method = ty::Method::new(
            name,
            ty::Generics {
                type_param_defs: type_param_defs,
                region_param: None
            },
            transformed_self_ty,
            fty,
            explicit_self,
            ast::public,
            did
        );
        let provided_trait_method_info = ProvidedTraitMethodInfo {
            ty: ty_method,
            def_id: did
        };

        vec::push(&mut result, provided_trait_method_info);
    }

    return result;
}

/// Returns the supertraits of the given trait.
pub fn get_supertraits(cdata: cmd, id: ast::node_id, tcx: ty::ctxt)
                    -> ~[@ty::TraitRef] {
    let mut results = ~[];
    let item_doc = lookup_item(id, cdata.data);
    for reader::tagged_docs(item_doc, tag_item_super_trait_ref) |trait_doc| {
        results.push(@doc_trait_ref(trait_doc, tcx, cdata));
    }
    return results;
}

pub fn get_type_name_if_impl(intr: @ident_interner,
                             cdata: cmd,
                             node_id: ast::node_id) -> Option<ast::ident> {
    let item = lookup_item(node_id, cdata.data);
    if item_family(item) != Impl {
        return None;
    }

    for reader::tagged_docs(item, tag_item_impl_type_basename) |doc| {
        return Some(intr.intern(str::from_bytes(reader::doc_data(doc))));
    }

    return None;
}

pub fn get_static_methods_if_impl(intr: @ident_interner,
                                  cdata: cmd,
                                  node_id: ast::node_id)
                               -> Option<~[StaticMethodInfo]> {
    let item = lookup_item(node_id, cdata.data);
    if item_family(item) != Impl {
        return None;
    }

    // If this impl implements a trait, don't consider it.
    for reader::tagged_docs(item, tag_item_trait_ref) |_doc| {
        return None;
    }

    let mut impl_method_ids = ~[];
    for reader::tagged_docs(item, tag_item_impl_method) |impl_method_doc| {
        impl_method_ids.push(parse_def_id(reader::doc_data(impl_method_doc)));
    }

    let mut static_impl_methods = ~[];
    for impl_method_ids.each |impl_method_id| {
        let impl_method_doc = lookup_item(impl_method_id.node, cdata.data);
        let family = item_family(impl_method_doc);
        match family {
            StaticMethod | UnsafeStaticMethod | PureStaticMethod => {
                let purity;
                match item_family(impl_method_doc) {
                    StaticMethod => purity = ast::impure_fn,
                    UnsafeStaticMethod => purity = ast::unsafe_fn,
                    PureStaticMethod => purity = ast::pure_fn,
                    _ => fail!()
                }

                static_impl_methods.push(StaticMethodInfo {
                    ident: item_name(intr, impl_method_doc),
                    def_id: item_def_id(impl_method_doc, cdata),
                    purity: purity
                });
            }
            _ => {}
        }
    }

    return Some(static_impl_methods);
}

pub fn get_item_attrs(cdata: cmd,
                      node_id: ast::node_id,
                      f: &fn(~[@ast::meta_item])) {

    let item = lookup_item(node_id, cdata.data);
    for reader::tagged_docs(item, tag_attributes) |attributes| {
        for reader::tagged_docs(attributes, tag_attribute) |attribute| {
            f(get_meta_items(attribute));
        }
    }
}

fn struct_field_family_to_visibility(family: Family) -> ast::visibility {
    match family {
      PublicField => ast::public,
      PrivateField => ast::private,
      InheritedField => ast::inherited,
      _ => fail!()
    }
}

pub fn get_struct_fields(intr: @ident_interner, cdata: cmd, id: ast::node_id)
    -> ~[ty::field_ty] {
    let data = cdata.data;
    let item = lookup_item(id, data);
    let mut result = ~[];
    for reader::tagged_docs(item, tag_item_field) |an_item| {
        let f = item_family(an_item);
        if f == PublicField || f == PrivateField || f == InheritedField {
            let name = item_name(intr, an_item);
            let did = item_def_id(an_item, cdata);
            result.push(ty::field_ty {
                ident: name,
                id: did, vis:
                struct_field_family_to_visibility(f),
            });
        }
    }
    for reader::tagged_docs(item, tag_item_unnamed_field) |an_item| {
        let did = item_def_id(an_item, cdata);
        result.push(ty::field_ty {
            ident: special_idents::unnamed_field,
            id: did,
            vis: ast::inherited,
        });
    }
    result
}

pub fn get_item_visibility(cdata: cmd, id: ast::node_id)
                        -> ast::visibility {
    item_visibility(lookup_item(id, cdata.data))
}

fn family_has_type_params(fam: Family) -> bool {
    match fam {
      Const | ForeignType | Mod | ForeignMod | PublicField | PrivateField
      | ForeignFn => false,
      _           => true
    }
}

fn family_names_type(fam: Family) -> bool {
    match fam { Type | Mod | Trait => true, _ => false }
}

fn read_path(d: ebml::Doc) -> (~str, uint) {
    let desc = reader::doc_data(d);
    let pos = io::u64_from_be_bytes(desc, 0u, 4u) as uint;
    let pathbytes = vec::slice::<u8>(desc, 4u, vec::len::<u8>(desc));
    let path = str::from_bytes(pathbytes);

    (path, pos)
}

fn describe_def(items: ebml::Doc, id: ast::def_id) -> ~str {
    if id.crate != ast::local_crate { return ~"external"; }
    let it = match maybe_find_item(id.node, items) {
        Some(it) => it,
        None => fail!("describe_def: item not found %?", id)
    };
    return item_family_to_str(item_family(it));
}

fn item_family_to_str(fam: Family) -> ~str {
    match fam {
      Const => ~"const",
      Fn => ~"fn",
      UnsafeFn => ~"unsafe fn",
      PureFn => ~"pure fn",
      StaticMethod => ~"static method",
      UnsafeStaticMethod => ~"unsafe static method",
      PureStaticMethod => ~"pure static method",
      ForeignFn => ~"foreign fn",
      Type => ~"type",
      ForeignType => ~"foreign type",
      Mod => ~"mod",
      ForeignMod => ~"foreign mod",
      Enum => ~"enum",
      Variant => ~"variant",
      Impl => ~"impl",
      Trait => ~"trait",
      Struct => ~"struct",
      PublicField => ~"public field",
      PrivateField => ~"private field",
      InheritedField => ~"inherited field",
    }
}

fn get_meta_items(md: ebml::Doc) -> ~[@ast::meta_item] {
    let mut items: ~[@ast::meta_item] = ~[];
    for reader::tagged_docs(md, tag_meta_item_word) |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(reader::doc_data(nd));
        items.push(attr::mk_word_item(@n));
    };
    for reader::tagged_docs(md, tag_meta_item_name_value) |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = reader::get_doc(meta_item_doc, tag_meta_item_value);
        let n = str::from_bytes(reader::doc_data(nd));
        let v = str::from_bytes(reader::doc_data(vd));
        // FIXME (#623): Should be able to decode meta_name_value variants,
        // but currently the encoder just drops them
        items.push(attr::mk_name_value_item_str(@n, @v));
    };
    for reader::tagged_docs(md, tag_meta_item_list) |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let n = str::from_bytes(reader::doc_data(nd));
        let subitems = get_meta_items(meta_item_doc);
        items.push(attr::mk_list_item(@n, subitems));
    };
    return items;
}

fn get_attributes(md: ebml::Doc) -> ~[ast::attribute] {
    let mut attrs: ~[ast::attribute] = ~[];
    match reader::maybe_get_doc(md, tag_attributes) {
      option::Some(attrs_d) => {
        for reader::tagged_docs(attrs_d, tag_attribute) |attr_doc| {
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert!((vec::len(meta_items) == 1u));
            let meta_item = meta_items[0];
            attrs.push(
                codemap::spanned {
                    node: ast::attribute_ {
                        style: ast::attr_outer,
                        value: meta_item,
                        is_sugared_doc: false,
                    },
                    span: codemap::dummy_sp()
                });
        };
      }
      option::None => ()
    }
    return attrs;
}

fn list_meta_items(intr: @ident_interner,
                   meta_items: ebml::Doc,
                   out: @io::Writer) {
    for get_meta_items(meta_items).each |mi| {
        out.write_str(fmt!("%s\n", pprust::meta_item_to_str(*mi, intr)));
    }
}

fn list_crate_attributes(intr: @ident_interner, md: ebml::Doc, hash: &str,
                         out: @io::Writer) {
    out.write_str(fmt!("=Crate Attributes (%s)=\n", hash));

    for get_attributes(md).each |attr| {
        out.write_str(fmt!("%s\n", pprust::attribute_to_str(*attr, intr)));
    }

    out.write_str(~"\n\n");
}

pub fn get_crate_attributes(data: @~[u8]) -> ~[ast::attribute] {
    return get_attributes(reader::Doc(data));
}

pub struct crate_dep {
    cnum: ast::crate_num,
    name: ast::ident,
    vers: @~str,
    hash: @~str
}

pub fn get_crate_deps(intr: @ident_interner, data: @~[u8]) -> ~[crate_dep] {
    let mut deps: ~[crate_dep] = ~[];
    let cratedoc = reader::Doc(data);
    let depsdoc = reader::get_doc(cratedoc, tag_crate_deps);
    let mut crate_num = 1;
    fn docstr(doc: ebml::Doc, tag_: uint) -> ~str {
        str::from_bytes(reader::doc_data(reader::get_doc(doc, tag_)))
    }
    for reader::tagged_docs(depsdoc, tag_crate_dep) |depdoc| {
        deps.push(crate_dep {cnum: crate_num,
                  name: intr.intern(docstr(depdoc, tag_crate_dep_name)),
                  vers: @docstr(depdoc, tag_crate_dep_vers),
                  hash: @docstr(depdoc, tag_crate_dep_hash)});
        crate_num += 1;
    };
    return deps;
}

fn list_crate_deps(intr: @ident_interner, data: @~[u8], out: @io::Writer) {
    out.write_str(~"=External Dependencies=\n");

    for get_crate_deps(intr, data).each |dep| {
        out.write_str(
            fmt!("%d %s-%s-%s\n",
                 dep.cnum, *intr.get(dep.name), *dep.hash, *dep.vers));
    }

    out.write_str(~"\n");
}

pub fn get_crate_hash(data: @~[u8]) -> @~str {
    let cratedoc = reader::Doc(data);
    let hashdoc = reader::get_doc(cratedoc, tag_crate_hash);
    @str::from_bytes(reader::doc_data(hashdoc))
}

pub fn get_crate_vers(data: @~[u8]) -> @~str {
    let attrs = decoder::get_crate_attributes(data);
    let linkage_attrs = attr::find_linkage_metas(attrs);

    match attr::last_meta_item_value_str_by_name(linkage_attrs, ~"vers") {
        Some(ver) => ver,
        None => @~"0.0"
    }
}

fn iter_crate_items(intr: @ident_interner, cdata: cmd,
                    get_crate_data: GetCrateDataCb,
                    proc: &fn(path: &str, ast::def_id)) {
    for each_path(intr, cdata, get_crate_data) |path_string, def_like| {
        match def_like {
            dl_impl(*) | dl_field => {}
            dl_def(def) => {
                proc(path_string,
                     ast_util::def_id_of_def(def))
            }
        }
    }
}

pub fn list_crate_metadata(intr: @ident_interner, bytes: @~[u8],
                           out: @io::Writer) {
    let hash = get_crate_hash(bytes);
    let md = reader::Doc(bytes);
    list_crate_attributes(intr, md, *hash, out);
    list_crate_deps(intr, bytes, out);
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
pub fn translate_def_id(cdata: cmd, did: ast::def_id) -> ast::def_id {
    if did.crate == ast::local_crate {
        return ast::def_id { crate: cdata.cnum, node: did.node };
    }

    match cdata.cnum_map.find(&did.crate) {
      option::Some(&n) => ast::def_id { crate: n, node: did.node },
      option::None => fail!("didn't find a crate in the cnum_map")
    }
}

pub fn get_link_args_for_crate(cdata: cmd) -> ~[~str] {
    let link_args = reader::get_doc(reader::Doc(cdata.data), tag_link_args);
    let mut result = ~[];
    for reader::tagged_docs(link_args, tag_link_args_arg) |arg_doc| {
        result.push(reader::doc_as_str(arg_doc));
    }
    result
}
