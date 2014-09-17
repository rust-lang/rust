// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Decoding metadata from a single crate's metadata

#![allow(non_camel_case_types)]

use back::svh::Svh;
use metadata::cstore::crate_metadata;
use metadata::common::*;
use metadata::csearch::StaticMethodInfo;
use metadata::csearch;
use metadata::cstore;
use metadata::tydecode::{parse_ty_data, parse_region_data, parse_def_id,
                         parse_type_param_def_data, parse_bounds_data,
                         parse_bare_fn_ty_data, parse_trait_ref_data};
use middle::def;
use middle::lang_items;
use middle::resolve::{TraitItemKind, TypeTraitItemKind};
use middle::subst;
use middle::ty::{ImplContainer, TraitContainer};
use middle::ty;
use middle::typeck;
use middle::astencode::vtable_decoder_helpers;

use std::hash::Hash;
use std::hash;
use std::io::extensions::u64_from_be_bytes;
use std::io;
use std::collections::hashmap::HashMap;
use std::rc::Rc;
use std::u64;
use rbml::reader;
use rbml;
use serialize::Decodable;
use syntax::ast_map;
use syntax::attr;
use syntax::parse::token::{IdentInterner, special_idents};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ast;
use syntax::codemap;
use syntax::ptr::P;

pub type Cmd<'a> = &'a crate_metadata;

// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.

fn lookup_hash<'a>(d: rbml::Doc<'a>, eq_fn: |&[u8]| -> bool,
                   hash: u64) -> Option<rbml::Doc<'a>> {
    let index = reader::get_doc(d, tag_index);
    let table = reader::get_doc(index, tag_index_table);
    let hash_pos = table.start + (hash % 256 * 4) as uint;
    let pos = u64_from_be_bytes(d.data, hash_pos, 4) as uint;
    let tagged_doc = reader::doc_at(d.data, pos).unwrap();

    let belt = tag_index_buckets_bucket_elt;

    let mut ret = None;
    reader::tagged_docs(tagged_doc.doc, belt, |elt| {
        let pos = u64_from_be_bytes(elt.data, elt.start, 4) as uint;
        if eq_fn(elt.data.slice(elt.start + 4, elt.end)) {
            ret = Some(reader::doc_at(d.data, pos).unwrap().doc);
            false
        } else {
            true
        }
    });
    ret
}

pub fn maybe_find_item<'a>(item_id: ast::NodeId,
                           items: rbml::Doc<'a>) -> Option<rbml::Doc<'a>> {
    fn eq_item(bytes: &[u8], item_id: ast::NodeId) -> bool {
        return u64_from_be_bytes(
            bytes.slice(0u, 4u), 0u, 4u) as ast::NodeId
            == item_id;
    }
    lookup_hash(items,
                |a| eq_item(a, item_id),
                hash::hash(&(item_id as i64)))
}

fn find_item<'a>(item_id: ast::NodeId, items: rbml::Doc<'a>) -> rbml::Doc<'a> {
    match maybe_find_item(item_id, items) {
       None => fail!("lookup_item: id not found: {}", item_id),
       Some(d) => d
    }
}

// Looks up an item in the given metadata and returns an rbml doc pointing
// to the item data.
fn lookup_item<'a>(item_id: ast::NodeId, data: &'a [u8]) -> rbml::Doc<'a> {
    let items = reader::get_doc(rbml::Doc::new(data), tag_items);
    find_item(item_id, items)
}

#[deriving(PartialEq)]
enum Family {
    ImmStatic,             // c
    MutStatic,             // b
    Fn,                    // f
    UnsafeFn,              // u
    StaticMethod,          // F
    UnsafeStaticMethod,    // U
    Type,                  // y
    ForeignType,           // T
    Mod,                   // m
    ForeignMod,            // n
    Enum,                  // t
    TupleVariant,          // v
    StructVariant,         // V
    Impl,                  // i
    Trait,                 // I
    Struct,                // S
    PublicField,           // g
    InheritedField         // N
}

fn item_family(item: rbml::Doc) -> Family {
    let fam = reader::get_doc(item, tag_items_data_item_family);
    match reader::doc_as_u8(fam) as char {
      'c' => ImmStatic,
      'b' => MutStatic,
      'f' => Fn,
      'u' => UnsafeFn,
      'F' => StaticMethod,
      'U' => UnsafeStaticMethod,
      'y' => Type,
      'T' => ForeignType,
      'm' => Mod,
      'n' => ForeignMod,
      't' => Enum,
      'v' => TupleVariant,
      'V' => StructVariant,
      'i' => Impl,
      'I' => Trait,
      'S' => Struct,
      'g' => PublicField,
      'N' => InheritedField,
       c => fail!("unexpected family char: {}", c)
    }
}

fn item_visibility(item: rbml::Doc) -> ast::Visibility {
    match reader::maybe_get_doc(item, tag_items_data_item_visibility) {
        None => ast::Public,
        Some(visibility_doc) => {
            match reader::doc_as_u8(visibility_doc) as char {
                'y' => ast::Public,
                'i' => ast::Inherited,
                _ => fail!("unknown visibility character")
            }
        }
    }
}

fn item_sort(item: rbml::Doc) -> char {
    // NB(pcwalton): The default of 'r' here is relied upon in
    // `is_associated_type` below.
    let mut ret = 'r';
    reader::tagged_docs(item, tag_item_trait_item_sort, |doc| {
        ret = doc.as_str_slice().as_bytes()[0] as char;
        false
    });
    ret
}

fn item_symbol(item: rbml::Doc) -> String {
    reader::get_doc(item, tag_items_data_item_symbol).as_str().to_string()
}

fn item_parent_item(d: rbml::Doc) -> Option<ast::DefId> {
    let mut ret = None;
    reader::tagged_docs(d, tag_items_data_parent_item, |did| {
        ret = Some(reader::with_doc_data(did, parse_def_id));
        false
    });
    ret
}

fn item_reqd_and_translated_parent_item(cnum: ast::CrateNum,
                                        d: rbml::Doc) -> ast::DefId {
    let trait_did = item_parent_item(d).expect("item without parent");
    ast::DefId { krate: cnum, node: trait_did.node }
}

fn item_def_id(d: rbml::Doc, cdata: Cmd) -> ast::DefId {
    let tagdoc = reader::get_doc(d, tag_def_id);
    return translate_def_id(cdata, reader::with_doc_data(tagdoc, parse_def_id));
}

fn get_provided_source(d: rbml::Doc, cdata: Cmd) -> Option<ast::DefId> {
    reader::maybe_get_doc(d, tag_item_method_provided_source).map(|doc| {
        translate_def_id(cdata, reader::with_doc_data(doc, parse_def_id))
    })
}

fn each_reexport(d: rbml::Doc, f: |rbml::Doc| -> bool) -> bool {
    reader::tagged_docs(d, tag_items_data_item_reexport, f)
}

fn variant_disr_val(d: rbml::Doc) -> Option<ty::Disr> {
    reader::maybe_get_doc(d, tag_disr_val).and_then(|val_doc| {
        reader::with_doc_data(val_doc, |data| u64::parse_bytes(data, 10u))
    })
}

fn doc_type(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::t {
    let tp = reader::get_doc(doc, tag_items_data_item_type);
    parse_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                  |_, did| translate_def_id(cdata, did))
}

fn doc_method_fty(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::BareFnTy {
    let tp = reader::get_doc(doc, tag_item_method_fty);
    parse_bare_fn_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                          |_, did| translate_def_id(cdata, did))
}

pub fn item_type(_item_id: ast::DefId, item: rbml::Doc,
                 tcx: &ty::ctxt, cdata: Cmd) -> ty::t {
    doc_type(item, tcx, cdata)
}

fn doc_trait_ref(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::TraitRef {
    parse_trait_ref_data(doc.data, cdata.cnum, doc.start, tcx,
                         |_, did| translate_def_id(cdata, did))
}

fn item_trait_ref(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::TraitRef {
    let tp = reader::get_doc(doc, tag_item_trait_ref);
    doc_trait_ref(tp, tcx, cdata)
}

fn doc_bounds(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::ParamBounds {
    parse_bounds_data(doc.data, cdata.cnum, doc.start, tcx,
                      |_, did| translate_def_id(cdata, did))
}

fn trait_def_bounds(doc: rbml::Doc, tcx: &ty::ctxt, cdata: Cmd) -> ty::ParamBounds {
    let d = reader::get_doc(doc, tag_trait_def_bounds);
    doc_bounds(d, tcx, cdata)
}

fn enum_variant_ids(item: rbml::Doc, cdata: Cmd) -> Vec<ast::DefId> {
    let mut ids: Vec<ast::DefId> = Vec::new();
    let v = tag_items_data_item_variant;
    reader::tagged_docs(item, v, |p| {
        let ext = reader::with_doc_data(p, parse_def_id);
        ids.push(ast::DefId { krate: cdata.cnum, node: ext.node });
        true
    });
    return ids;
}

fn item_path(item_doc: rbml::Doc) -> Vec<ast_map::PathElem> {
    let path_doc = reader::get_doc(item_doc, tag_path);

    let len_doc = reader::get_doc(path_doc, tag_path_len);
    let len = reader::doc_as_u32(len_doc) as uint;

    let mut result = Vec::with_capacity(len);
    reader::docs(path_doc, |tag, elt_doc| {
        if tag == tag_path_elem_mod {
            let s = elt_doc.as_str_slice();
            result.push(ast_map::PathMod(token::intern(s)));
        } else if tag == tag_path_elem_name {
            let s = elt_doc.as_str_slice();
            result.push(ast_map::PathName(token::intern(s)));
        } else {
            // ignore tag_path_len element
        }
        true
    });

    result
}

fn item_name(intr: &IdentInterner, item: rbml::Doc) -> ast::Ident {
    let name = reader::get_doc(item, tag_paths_data_name);
    let string = name.as_str_slice();
    match intr.find_equiv(&string) {
        None => token::str_to_ident(string),
        Some(val) => ast::Ident::new(val),
    }
}

fn item_to_def_like(item: rbml::Doc, did: ast::DefId, cnum: ast::CrateNum)
    -> DefLike {
    let fam = item_family(item);
    match fam {
        ImmStatic => DlDef(def::DefStatic(did, false)),
        MutStatic => DlDef(def::DefStatic(did, true)),
        Struct    => DlDef(def::DefStruct(did)),
        UnsafeFn  => DlDef(def::DefFn(did, ast::UnsafeFn)),
        Fn        => DlDef(def::DefFn(did, ast::NormalFn)),
        StaticMethod | UnsafeStaticMethod => {
            let fn_style = if fam == UnsafeStaticMethod {
                ast::UnsafeFn
            } else {
                ast::NormalFn
            };
            // def_static_method carries an optional field of its enclosing
            // trait or enclosing impl (if this is an inherent static method).
            // So we need to detect whether this is in a trait or not, which
            // we do through the mildly hacky way of checking whether there is
            // a trait_parent_sort.
            let provenance = if reader::maybe_get_doc(
                  item, tag_item_trait_parent_sort).is_some() {
                def::FromTrait(item_reqd_and_translated_parent_item(cnum,
                                                                    item))
            } else {
                def::FromImpl(item_reqd_and_translated_parent_item(cnum,
                                                                   item))
            };
            DlDef(def::DefStaticMethod(did, provenance, fn_style))
        }
        Type | ForeignType => DlDef(def::DefTy(did, false)),
        Mod => DlDef(def::DefMod(did)),
        ForeignMod => DlDef(def::DefForeignMod(did)),
        StructVariant => {
            let enum_did = item_reqd_and_translated_parent_item(cnum, item);
            DlDef(def::DefVariant(enum_did, did, true))
        }
        TupleVariant => {
            let enum_did = item_reqd_and_translated_parent_item(cnum, item);
            DlDef(def::DefVariant(enum_did, did, false))
        }
        Trait => DlDef(def::DefTrait(did)),
        Enum => DlDef(def::DefTy(did, true)),
        Impl => DlImpl(did),
        PublicField | InheritedField => DlField,
    }
}

pub fn get_trait_def(cdata: Cmd,
                     item_id: ast::NodeId,
                     tcx: &ty::ctxt) -> ty::TraitDef
{
    let item_doc = lookup_item(item_id, cdata.data());
    let generics = doc_generics(item_doc, tcx, cdata, tag_item_generics);
    let bounds = trait_def_bounds(item_doc, tcx, cdata);

    ty::TraitDef {
        generics: generics,
        bounds: bounds,
        trait_ref: Rc::new(item_trait_ref(item_doc, tcx, cdata))
    }
}

pub fn get_type(cdata: Cmd, id: ast::NodeId, tcx: &ty::ctxt)
    -> ty::Polytype {

    let item = lookup_item(id, cdata.data());

    let t = item_type(ast::DefId { krate: cdata.cnum, node: id }, item, tcx,
                      cdata);

    let generics = doc_generics(item, tcx, cdata, tag_item_generics);

    ty::Polytype {
        generics: generics,
        ty: t
    }
}

pub fn get_stability(cdata: Cmd, id: ast::NodeId) -> Option<attr::Stability> {
    let item = lookup_item(id, cdata.data());
    reader::maybe_get_doc(item, tag_items_data_item_stability).map(|doc| {
        let mut decoder = reader::Decoder::new(doc);
        Decodable::decode(&mut decoder).unwrap()
    })
}

pub fn get_impl_trait(cdata: Cmd,
                      id: ast::NodeId,
                      tcx: &ty::ctxt) -> Option<Rc<ty::TraitRef>>
{
    let item_doc = lookup_item(id, cdata.data());
    reader::maybe_get_doc(item_doc, tag_item_trait_ref).map(|tp| {
        Rc::new(doc_trait_ref(tp, tcx, cdata))
    })
}

pub fn get_impl_vtables(cdata: Cmd,
                        id: ast::NodeId,
                        tcx: &ty::ctxt)
                        -> typeck::vtable_res
{
    let item_doc = lookup_item(id, cdata.data());
    let vtables_doc = reader::get_doc(item_doc, tag_item_impl_vtables);
    let mut decoder = reader::Decoder::new(vtables_doc);
    decoder.read_vtable_res(tcx, cdata)
}


pub fn get_symbol(data: &[u8], id: ast::NodeId) -> String {
    return item_symbol(lookup_item(id, data));
}

// Something that a name can resolve to.
#[deriving(Clone)]
pub enum DefLike {
    DlDef(def::Def),
    DlImpl(ast::DefId),
    DlField
}

/// Iterates over the language items in the given crate.
pub fn each_lang_item(cdata: Cmd, f: |ast::NodeId, uint| -> bool) -> bool {
    let root = rbml::Doc::new(cdata.data());
    let lang_items = reader::get_doc(root, tag_lang_items);
    reader::tagged_docs(lang_items, tag_lang_items_item, |item_doc| {
        let id_doc = reader::get_doc(item_doc, tag_lang_items_item_id);
        let id = reader::doc_as_u32(id_doc) as uint;
        let node_id_doc = reader::get_doc(item_doc,
                                          tag_lang_items_item_node_id);
        let node_id = reader::doc_as_u32(node_id_doc) as ast::NodeId;

        f(node_id, id)
    })
}

pub type GetCrateDataCb<'a> = |ast::CrateNum|: 'a -> Rc<crate_metadata>;

fn each_child_of_item_or_crate(intr: Rc<IdentInterner>,
                               cdata: Cmd,
                               item_doc: rbml::Doc,
                               get_crate_data: GetCrateDataCb,
                               callback: |DefLike,
                                          ast::Ident,
                                          ast::Visibility|) {
    // Iterate over all children.
    let _ = reader::tagged_docs(item_doc, tag_mod_child, |child_info_doc| {
        let child_def_id = reader::with_doc_data(child_info_doc,
                                                 parse_def_id);
        let child_def_id = translate_def_id(cdata, child_def_id);

        // This item may be in yet another crate if it was the child of a
        // reexport.
        let crate_data = if child_def_id.krate == cdata.cnum {
            None
        } else {
            Some(get_crate_data(child_def_id.krate))
        };
        let crate_data = match crate_data {
            Some(ref cdata) => &**cdata,
            None => cdata
        };

        let other_crates_items = reader::get_doc(rbml::Doc::new(crate_data.data()), tag_items);

        // Get the item.
        match maybe_find_item(child_def_id.node, other_crates_items) {
            None => {}
            Some(child_item_doc) => {
                // Hand off the item to the callback.
                let child_name = item_name(&*intr, child_item_doc);
                let def_like = item_to_def_like(child_item_doc,
                                                child_def_id,
                                                cdata.cnum);
                let visibility = item_visibility(child_item_doc);
                callback(def_like, child_name, visibility);

            }
        }

        true
    });

    // As a special case, iterate over all static methods of
    // associated implementations too. This is a bit of a botch.
    // --pcwalton
    let _ = reader::tagged_docs(item_doc,
                                tag_items_data_item_inherent_impl,
                                |inherent_impl_def_id_doc| {
        let inherent_impl_def_id = item_def_id(inherent_impl_def_id_doc,
                                               cdata);
        let items = reader::get_doc(rbml::Doc::new(cdata.data()), tag_items);
        match maybe_find_item(inherent_impl_def_id.node, items) {
            None => {}
            Some(inherent_impl_doc) => {
                let _ = reader::tagged_docs(inherent_impl_doc,
                                            tag_item_impl_item,
                                            |impl_item_def_id_doc| {
                    let impl_item_def_id = item_def_id(impl_item_def_id_doc,
                                                       cdata);
                    match maybe_find_item(impl_item_def_id.node, items) {
                        None => {}
                        Some(impl_method_doc) => {
                            match item_family(impl_method_doc) {
                                StaticMethod | UnsafeStaticMethod => {
                                    // Hand off the static method
                                    // to the callback.
                                    let static_method_name =
                                        item_name(&*intr, impl_method_doc);
                                    let static_method_def_like =
                                        item_to_def_like(impl_method_doc,
                                                         impl_item_def_id,
                                                         cdata.cnum);
                                    callback(static_method_def_like,
                                             static_method_name,
                                             item_visibility(impl_method_doc));
                                }
                                _ => {}
                            }
                        }
                    }

                    true
                });
            }
        }

        true
    });

    // Iterate over all reexports.
    let _ = each_reexport(item_doc, |reexport_doc| {
        let def_id_doc = reader::get_doc(reexport_doc,
                                         tag_items_data_item_reexport_def_id);
        let child_def_id = reader::with_doc_data(def_id_doc,
                                                 parse_def_id);
        let child_def_id = translate_def_id(cdata, child_def_id);

        let name_doc = reader::get_doc(reexport_doc,
                                       tag_items_data_item_reexport_name);
        let name = name_doc.as_str_slice();

        // This reexport may be in yet another crate.
        let crate_data = if child_def_id.krate == cdata.cnum {
            None
        } else {
            Some(get_crate_data(child_def_id.krate))
        };
        let crate_data = match crate_data {
            Some(ref cdata) => &**cdata,
            None => cdata
        };

        let other_crates_items = reader::get_doc(rbml::Doc::new(crate_data.data()), tag_items);

        // Get the item.
        match maybe_find_item(child_def_id.node, other_crates_items) {
            None => {}
            Some(child_item_doc) => {
                // Hand off the item to the callback.
                let def_like = item_to_def_like(child_item_doc,
                                                child_def_id,
                                                child_def_id.krate);
                // These items have a public visibility because they're part of
                // a public re-export.
                callback(def_like, token::str_to_ident(name), ast::Public);
            }
        }

        true
    });
}

/// Iterates over each child of the given item.
pub fn each_child_of_item(intr: Rc<IdentInterner>,
                          cdata: Cmd,
                          id: ast::NodeId,
                          get_crate_data: GetCrateDataCb,
                          callback: |DefLike, ast::Ident, ast::Visibility|) {
    // Find the item.
    let root_doc = rbml::Doc::new(cdata.data());
    let items = reader::get_doc(root_doc, tag_items);
    let item_doc = match maybe_find_item(id, items) {
        None => return,
        Some(item_doc) => item_doc,
    };

    each_child_of_item_or_crate(intr,
                                cdata,
                                item_doc,
                                get_crate_data,
                                callback)
}

/// Iterates over all the top-level crate items.
pub fn each_top_level_item_of_crate(intr: Rc<IdentInterner>,
                                    cdata: Cmd,
                                    get_crate_data: GetCrateDataCb,
                                    callback: |DefLike,
                                               ast::Ident,
                                               ast::Visibility|) {
    let root_doc = rbml::Doc::new(cdata.data());
    let misc_info_doc = reader::get_doc(root_doc, tag_misc_info);
    let crate_items_doc = reader::get_doc(misc_info_doc,
                                          tag_misc_info_crate_items);

    each_child_of_item_or_crate(intr,
                                cdata,
                                crate_items_doc,
                                get_crate_data,
                                callback)
}

pub fn get_item_path(cdata: Cmd, id: ast::NodeId) -> Vec<ast_map::PathElem> {
    item_path(lookup_item(id, cdata.data()))
}

pub type DecodeInlinedItem<'a> = <'tcx> |cdata: Cmd,
                                         tcx: &ty::ctxt<'tcx>,
                                         path: Vec<ast_map::PathElem>,
                                         par_doc: rbml::Doc|: 'a
                                         -> Result<&'tcx ast::InlinedItem,
                                                   Vec<ast_map::PathElem>>;

pub fn maybe_get_item_ast<'tcx>(cdata: Cmd, tcx: &ty::ctxt<'tcx>, id: ast::NodeId,
                                decode_inlined_item: DecodeInlinedItem)
                                -> csearch::found_ast<'tcx> {
    debug!("Looking up item: {}", id);
    let item_doc = lookup_item(id, cdata.data());
    let path = Vec::from_slice(item_path(item_doc).init());
    match decode_inlined_item(cdata, tcx, path, item_doc) {
        Ok(ii) => csearch::found(ii),
        Err(path) => {
            match item_parent_item(item_doc) {
                Some(did) => {
                    let did = translate_def_id(cdata, did);
                    let parent_item = lookup_item(did.node, cdata.data());
                    match decode_inlined_item(cdata, tcx, path, parent_item) {
                        Ok(ii) => csearch::found_parent(did, ii),
                        Err(_) => csearch::not_found
                    }
                }
                None => csearch::not_found
            }
        }
    }
}

pub fn get_enum_variants(intr: Rc<IdentInterner>, cdata: Cmd, id: ast::NodeId,
                     tcx: &ty::ctxt) -> Vec<Rc<ty::VariantInfo>> {
    let data = cdata.data();
    let items = reader::get_doc(rbml::Doc::new(data), tag_items);
    let item = find_item(id, items);
    let mut disr_val = 0;
    enum_variant_ids(item, cdata).iter().map(|did| {
        let item = find_item(did.node, items);
        let ctor_ty = item_type(ast::DefId { krate: cdata.cnum, node: id},
                                item, tcx, cdata);
        let name = item_name(&*intr, item);
        let arg_tys = match ty::get(ctor_ty).sty {
            ty::ty_bare_fn(ref f) => f.sig.inputs.clone(),
            _ => Vec::new(), // Nullary enum variant.
        };
        match variant_disr_val(item) {
            Some(val) => { disr_val = val; }
            _         => { /* empty */ }
        }
        let old_disr_val = disr_val;
        disr_val += 1;
        Rc::new(ty::VariantInfo {
            args: arg_tys,
            arg_names: None,
            ctor_ty: ctor_ty,
            name: name,
            // I'm not even sure if we encode visibility
            // for variants -- TEST -- tjc
            id: *did,
            disr_val: old_disr_val,
            vis: ast::Inherited
        })
    }).collect()
}

fn get_explicit_self(item: rbml::Doc) -> ty::ExplicitSelfCategory {
    fn get_mutability(ch: u8) -> ast::Mutability {
        match ch as char {
            'i' => ast::MutImmutable,
            'm' => ast::MutMutable,
            _ => fail!("unknown mutability character: `{}`", ch as char),
        }
    }

    let explicit_self_doc = reader::get_doc(item, tag_item_trait_method_explicit_self);
    let string = explicit_self_doc.as_str_slice();

    let explicit_self_kind = string.as_bytes()[0];
    match explicit_self_kind as char {
        's' => ty::StaticExplicitSelfCategory,
        'v' => ty::ByValueExplicitSelfCategory,
        '~' => ty::ByBoxExplicitSelfCategory,
        // FIXME(#4846) expl. region
        '&' => {
            ty::ByReferenceExplicitSelfCategory(
                ty::ReEmpty,
                get_mutability(string.as_bytes()[1]))
        }
        _ => fail!("unknown self type code: `{}`", explicit_self_kind as char)
    }
}

/// Returns the def IDs of all the items in the given implementation.
pub fn get_impl_items(cdata: Cmd, impl_id: ast::NodeId)
                      -> Vec<ty::ImplOrTraitItemId> {
    let mut impl_items = Vec::new();
    reader::tagged_docs(lookup_item(impl_id, cdata.data()),
                        tag_item_impl_item, |doc| {
        let def_id = item_def_id(doc, cdata);
        match item_sort(doc) {
            'r' | 'p' => impl_items.push(ty::MethodTraitItemId(def_id)),
            't' => impl_items.push(ty::TypeTraitItemId(def_id)),
            _ => fail!("unknown impl item sort"),
        }
        true
    });

    impl_items
}

pub fn get_trait_item_name_and_kind(intr: Rc<IdentInterner>,
                                    cdata: Cmd,
                                    id: ast::NodeId)
                                    -> (ast::Ident, TraitItemKind) {
    let doc = lookup_item(id, cdata.data());
    let name = item_name(&*intr, doc);
    match item_sort(doc) {
        'r' | 'p' => {
            let explicit_self = get_explicit_self(doc);
            (name, TraitItemKind::from_explicit_self_category(explicit_self))
        }
        't' => (name, TypeTraitItemKind),
        c => {
            fail!("get_trait_item_name_and_kind(): unknown trait item kind \
                   in metadata: `{}`", c)
        }
    }
}

pub fn get_impl_or_trait_item(intr: Rc<IdentInterner>,
                              cdata: Cmd,
                              id: ast::NodeId,
                              tcx: &ty::ctxt)
                              -> ty::ImplOrTraitItem {
    let method_doc = lookup_item(id, cdata.data());

    let def_id = item_def_id(method_doc, cdata);

    let container_id = item_reqd_and_translated_parent_item(cdata.cnum,
                                                            method_doc);
    let container_doc = lookup_item(container_id.node, cdata.data());
    let container = match item_family(container_doc) {
        Trait => TraitContainer(container_id),
        _ => ImplContainer(container_id),
    };

    let name = item_name(&*intr, method_doc);
    let vis = item_visibility(method_doc);

    match item_sort(method_doc) {
        'r' | 'p' => {
            let generics = doc_generics(method_doc, tcx, cdata,
                                        tag_method_ty_generics);
            let fty = doc_method_fty(method_doc, tcx, cdata);
            let explicit_self = get_explicit_self(method_doc);
            let provided_source = get_provided_source(method_doc, cdata);

            ty::MethodTraitItem(Rc::new(ty::Method::new(name,
                                                        generics,
                                                        fty,
                                                        explicit_self,
                                                        vis,
                                                        def_id,
                                                        container,
                                                        provided_source)))
        }
        't' => {
            ty::TypeTraitItem(Rc::new(ty::AssociatedType {
                ident: name,
                vis: vis,
                def_id: def_id,
                container: container,
            }))
        }
        _ => fail!("unknown impl/trait item sort"),
    }
}

pub fn get_trait_item_def_ids(cdata: Cmd, id: ast::NodeId)
                              -> Vec<ty::ImplOrTraitItemId> {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = Vec::new();
    reader::tagged_docs(item, tag_item_trait_item, |mth| {
        let def_id = item_def_id(mth, cdata);
        match item_sort(mth) {
            'r' | 'p' => result.push(ty::MethodTraitItemId(def_id)),
            't' => result.push(ty::TypeTraitItemId(def_id)),
            _ => fail!("unknown trait item sort"),
        }
        true
    });
    result
}

pub fn get_item_variances(cdata: Cmd, id: ast::NodeId) -> ty::ItemVariances {
    let data = cdata.data();
    let item_doc = lookup_item(id, data);
    let variance_doc = reader::get_doc(item_doc, tag_item_variances);
    let mut decoder = reader::Decoder::new(variance_doc);
    Decodable::decode(&mut decoder).unwrap()
}

pub fn get_provided_trait_methods(intr: Rc<IdentInterner>,
                                  cdata: Cmd,
                                  id: ast::NodeId,
                                  tcx: &ty::ctxt)
                                  -> Vec<Rc<ty::Method>> {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = Vec::new();

    reader::tagged_docs(item, tag_item_trait_item, |mth_id| {
        let did = item_def_id(mth_id, cdata);
        let mth = lookup_item(did.node, data);

        if item_sort(mth) == 'p' {
            let trait_item = get_impl_or_trait_item(intr.clone(),
                                                    cdata,
                                                    did.node,
                                                    tcx);
            match trait_item {
                ty::MethodTraitItem(ref method) => {
                    result.push((*method).clone())
                }
                ty::TypeTraitItem(_) => {}
            }
        }
        true
    });

    return result;
}

/// Returns the supertraits of the given trait.
pub fn get_supertraits(cdata: Cmd, id: ast::NodeId, tcx: &ty::ctxt)
                    -> Vec<Rc<ty::TraitRef>> {
    let mut results = Vec::new();
    let item_doc = lookup_item(id, cdata.data());
    reader::tagged_docs(item_doc, tag_item_super_trait_ref, |trait_doc| {
        // NB. Only reads the ones that *aren't* builtin-bounds. See also
        // get_trait_def() for collecting the builtin bounds.
        // FIXME(#8559): The builtin bounds shouldn't be encoded in the first place.
        let trait_ref = doc_trait_ref(trait_doc, tcx, cdata);
        if tcx.lang_items.to_builtin_kind(trait_ref.def_id).is_none() {
            results.push(Rc::new(trait_ref));
        }
        true
    });
    return results;
}

pub fn get_type_name_if_impl(cdata: Cmd,
                             node_id: ast::NodeId) -> Option<ast::Ident> {
    let item = lookup_item(node_id, cdata.data());
    if item_family(item) != Impl {
        return None;
    }

    let mut ret = None;
    reader::tagged_docs(item, tag_item_impl_type_basename, |doc| {
        ret = Some(token::str_to_ident(doc.as_str_slice()));
        false
    });

    ret
}

pub fn get_static_methods_if_impl(intr: Rc<IdentInterner>,
                                  cdata: Cmd,
                                  node_id: ast::NodeId)
                               -> Option<Vec<StaticMethodInfo> > {
    let item = lookup_item(node_id, cdata.data());
    if item_family(item) != Impl {
        return None;
    }

    // If this impl implements a trait, don't consider it.
    let ret = reader::tagged_docs(item, tag_item_trait_ref, |_doc| {
        false
    });

    if !ret { return None }

    let mut impl_method_ids = Vec::new();
    reader::tagged_docs(item, tag_item_impl_item, |impl_method_doc| {
        impl_method_ids.push(item_def_id(impl_method_doc, cdata));
        true
    });

    let mut static_impl_methods = Vec::new();
    for impl_method_id in impl_method_ids.iter() {
        let impl_method_doc = lookup_item(impl_method_id.node, cdata.data());
        let family = item_family(impl_method_doc);
        match family {
            StaticMethod | UnsafeStaticMethod => {
                let fn_style;
                match item_family(impl_method_doc) {
                    StaticMethod => fn_style = ast::NormalFn,
                    UnsafeStaticMethod => fn_style = ast::UnsafeFn,
                    _ => fail!()
                }

                static_impl_methods.push(StaticMethodInfo {
                    ident: item_name(&*intr, impl_method_doc),
                    def_id: item_def_id(impl_method_doc, cdata),
                    fn_style: fn_style,
                    vis: item_visibility(impl_method_doc),
                });
            }
            _ => {}
        }
    }

    return Some(static_impl_methods);
}

/// If node_id is the constructor of a tuple struct, retrieve the NodeId of
/// the actual type definition, otherwise, return None
pub fn get_tuple_struct_definition_if_ctor(cdata: Cmd,
                                           node_id: ast::NodeId)
    -> Option<ast::DefId>
{
    let item = lookup_item(node_id, cdata.data());
    let mut ret = None;
    reader::tagged_docs(item, tag_items_data_item_is_tuple_struct_ctor, |_| {
        ret = Some(item_reqd_and_translated_parent_item(cdata.cnum, item));
        false
    });
    ret
}

pub fn get_item_attrs(cdata: Cmd,
                      orig_node_id: ast::NodeId,
                      f: |Vec<ast::Attribute>|) {
    // The attributes for a tuple struct are attached to the definition, not the ctor;
    // we assume that someone passing in a tuple struct ctor is actually wanting to
    // look at the definition
    let node_id = get_tuple_struct_definition_if_ctor(cdata, orig_node_id);
    let node_id = node_id.map(|x| x.node).unwrap_or(orig_node_id);
    let item = lookup_item(node_id, cdata.data());
    f(get_attributes(item));
}

pub fn get_struct_field_attrs(cdata: Cmd) -> HashMap<ast::NodeId, Vec<ast::Attribute>> {
    let data = rbml::Doc::new(cdata.data());
    let fields = reader::get_doc(data, tag_struct_fields);
    let mut map = HashMap::new();
    reader::tagged_docs(fields, tag_struct_field, |field| {
        let id = reader::doc_as_u32(reader::get_doc(field, tag_struct_field_id));
        let attrs = get_attributes(field);
        map.insert(id, attrs);
        true
    });
    map
}

fn struct_field_family_to_visibility(family: Family) -> ast::Visibility {
    match family {
      PublicField => ast::Public,
      InheritedField => ast::Inherited,
      _ => fail!()
    }
}

pub fn get_struct_fields(intr: Rc<IdentInterner>, cdata: Cmd, id: ast::NodeId)
    -> Vec<ty::field_ty> {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = Vec::new();
    reader::tagged_docs(item, tag_item_field, |an_item| {
        let f = item_family(an_item);
        if f == PublicField || f == InheritedField {
            // FIXME #6993: name should be of type Name, not Ident
            let name = item_name(&*intr, an_item);
            let did = item_def_id(an_item, cdata);
            let tagdoc = reader::get_doc(an_item, tag_item_field_origin);
            let origin_id =  translate_def_id(cdata, reader::with_doc_data(tagdoc, parse_def_id));
            result.push(ty::field_ty {
                name: name.name,
                id: did,
                vis: struct_field_family_to_visibility(f),
                origin: origin_id,
            });
        }
        true
    });
    reader::tagged_docs(item, tag_item_unnamed_field, |an_item| {
        let did = item_def_id(an_item, cdata);
        let tagdoc = reader::get_doc(an_item, tag_item_field_origin);
        let f = item_family(an_item);
        let origin_id =  translate_def_id(cdata, reader::with_doc_data(tagdoc, parse_def_id));
        result.push(ty::field_ty {
            name: special_idents::unnamed_field.name,
            id: did,
            vis: struct_field_family_to_visibility(f),
            origin: origin_id,
        });
        true
    });
    result
}

fn get_meta_items(md: rbml::Doc) -> Vec<P<ast::MetaItem>> {
    let mut items: Vec<P<ast::MetaItem>> = Vec::new();
    reader::tagged_docs(md, tag_meta_item_word, |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let n = token::intern_and_get_ident(nd.as_str_slice());
        items.push(attr::mk_word_item(n));
        true
    });
    reader::tagged_docs(md, tag_meta_item_name_value, |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let vd = reader::get_doc(meta_item_doc, tag_meta_item_value);
        let n = token::intern_and_get_ident(nd.as_str_slice());
        let v = token::intern_and_get_ident(vd.as_str_slice());
        // FIXME (#623): Should be able to decode MetaNameValue variants,
        // but currently the encoder just drops them
        items.push(attr::mk_name_value_item_str(n, v));
        true
    });
    reader::tagged_docs(md, tag_meta_item_list, |meta_item_doc| {
        let nd = reader::get_doc(meta_item_doc, tag_meta_item_name);
        let n = token::intern_and_get_ident(nd.as_str_slice());
        let subitems = get_meta_items(meta_item_doc);
        items.push(attr::mk_list_item(n, subitems.into_iter().collect()));
        true
    });
    return items;
}

fn get_attributes(md: rbml::Doc) -> Vec<ast::Attribute> {
    let mut attrs: Vec<ast::Attribute> = Vec::new();
    match reader::maybe_get_doc(md, tag_attributes) {
      Some(attrs_d) => {
        reader::tagged_docs(attrs_d, tag_attribute, |attr_doc| {
            let is_sugared_doc = reader::doc_as_u8(
                reader::get_doc(attr_doc, tag_attribute_is_sugared_doc)
            ) == 1;
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert_eq!(meta_items.len(), 1u);
            let meta_item = meta_items.into_iter().nth(0).unwrap();
            attrs.push(
                codemap::Spanned {
                    node: ast::Attribute_ {
                        id: attr::mk_attr_id(),
                        style: ast::AttrOuter,
                        value: meta_item,
                        is_sugared_doc: is_sugared_doc,
                    },
                    span: codemap::DUMMY_SP
                });
            true
        });
      }
      None => ()
    }
    return attrs;
}

fn list_crate_attributes(md: rbml::Doc, hash: &Svh,
                         out: &mut io::Writer) -> io::IoResult<()> {
    try!(write!(out, "=Crate Attributes ({})=\n", *hash));

    let r = get_attributes(md);
    for attr in r.iter() {
        try!(write!(out, "{}\n", pprust::attribute_to_string(attr)));
    }

    write!(out, "\n\n")
}

pub fn get_crate_attributes(data: &[u8]) -> Vec<ast::Attribute> {
    get_attributes(rbml::Doc::new(data))
}

#[deriving(Clone)]
pub struct CrateDep {
    pub cnum: ast::CrateNum,
    pub name: String,
    pub hash: Svh,
}

pub fn get_crate_deps(data: &[u8]) -> Vec<CrateDep> {
    let mut deps: Vec<CrateDep> = Vec::new();
    let cratedoc = rbml::Doc::new(data);
    let depsdoc = reader::get_doc(cratedoc, tag_crate_deps);
    let mut crate_num = 1;
    fn docstr(doc: rbml::Doc, tag_: uint) -> String {
        let d = reader::get_doc(doc, tag_);
        d.as_str_slice().to_string()
    }
    reader::tagged_docs(depsdoc, tag_crate_dep, |depdoc| {
        let name = docstr(depdoc, tag_crate_dep_crate_name);
        let hash = Svh::new(docstr(depdoc, tag_crate_dep_hash).as_slice());
        deps.push(CrateDep {
            cnum: crate_num,
            name: name,
            hash: hash,
        });
        crate_num += 1;
        true
    });
    return deps;
}

fn list_crate_deps(data: &[u8], out: &mut io::Writer) -> io::IoResult<()> {
    try!(write!(out, "=External Dependencies=\n"));
    for dep in get_crate_deps(data).iter() {
        try!(write!(out, "{} {}-{}\n", dep.cnum, dep.name, dep.hash));
    }
    try!(write!(out, "\n"));
    Ok(())
}

pub fn maybe_get_crate_hash(data: &[u8]) -> Option<Svh> {
    let cratedoc = rbml::Doc::new(data);
    reader::maybe_get_doc(cratedoc, tag_crate_hash).map(|doc| {
        Svh::new(doc.as_str_slice())
    })
}

pub fn get_crate_hash(data: &[u8]) -> Svh {
    let cratedoc = rbml::Doc::new(data);
    let hashdoc = reader::get_doc(cratedoc, tag_crate_hash);
    Svh::new(hashdoc.as_str_slice())
}

pub fn maybe_get_crate_name(data: &[u8]) -> Option<String> {
    let cratedoc = rbml::Doc::new(data);
    reader::maybe_get_doc(cratedoc, tag_crate_crate_name).map(|doc| {
        doc.as_str_slice().to_string()
    })
}

pub fn get_crate_triple(data: &[u8]) -> Option<String> {
    let cratedoc = rbml::Doc::new(data);
    let triple_doc = reader::maybe_get_doc(cratedoc, tag_crate_triple);
    triple_doc.map(|s| s.as_str().to_string())
}

pub fn get_crate_name(data: &[u8]) -> String {
    maybe_get_crate_name(data).expect("no crate name in crate")
}

pub fn list_crate_metadata(bytes: &[u8], out: &mut io::Writer) -> io::IoResult<()> {
    let hash = get_crate_hash(bytes);
    let md = rbml::Doc::new(bytes);
    try!(list_crate_attributes(md, &hash, out));
    list_crate_deps(bytes, out)
}

// Translates a def_id from an external crate to a def_id for the current
// compilation environment. We use this when trying to load types from
// external crates - if those types further refer to types in other crates
// then we must translate the crate number from that encoded in the external
// crate to the correct local crate number.
pub fn translate_def_id(cdata: Cmd, did: ast::DefId) -> ast::DefId {
    if did.krate == ast::LOCAL_CRATE {
        return ast::DefId { krate: cdata.cnum, node: did.node };
    }

    match cdata.cnum_map.find(&did.krate) {
        Some(&n) => {
            ast::DefId {
                krate: n,
                node: did.node,
            }
        }
        None => fail!("didn't find a crate in the cnum_map")
    }
}

pub fn each_impl(cdata: Cmd, callback: |ast::DefId|) {
    let impls_doc = reader::get_doc(rbml::Doc::new(cdata.data()), tag_impls);
    let _ = reader::tagged_docs(impls_doc, tag_impls_impl, |impl_doc| {
        callback(item_def_id(impl_doc, cdata));
        true
    });
}

pub fn each_implementation_for_type(cdata: Cmd,
                                    id: ast::NodeId,
                                    callback: |ast::DefId|) {
    let item_doc = lookup_item(id, cdata.data());
    reader::tagged_docs(item_doc,
                        tag_items_data_item_inherent_impl,
                        |impl_doc| {
        let implementation_def_id = item_def_id(impl_doc, cdata);
        callback(implementation_def_id);
        true
    });
}

pub fn each_implementation_for_trait(cdata: Cmd,
                                     id: ast::NodeId,
                                     callback: |ast::DefId|) {
    let item_doc = lookup_item(id, cdata.data());

    let _ = reader::tagged_docs(item_doc,
                                tag_items_data_item_extension_impl,
                                |impl_doc| {
        let implementation_def_id = item_def_id(impl_doc, cdata);
        callback(implementation_def_id);
        true
    });
}

pub fn get_trait_of_item(cdata: Cmd, id: ast::NodeId, tcx: &ty::ctxt)
                         -> Option<ast::DefId> {
    let item_doc = lookup_item(id, cdata.data());
    let parent_item_id = match item_parent_item(item_doc) {
        None => return None,
        Some(item_id) => item_id,
    };
    let parent_item_id = translate_def_id(cdata, parent_item_id);
    let parent_item_doc = lookup_item(parent_item_id.node, cdata.data());
    match item_family(parent_item_doc) {
        Trait => Some(item_def_id(parent_item_doc, cdata)),
        Impl => {
            reader::maybe_get_doc(parent_item_doc, tag_item_trait_ref)
                .map(|_| item_trait_ref(parent_item_doc, tcx, cdata).def_id)
        }
        _ => None
    }
}


pub fn get_native_libraries(cdata: Cmd)
                            -> Vec<(cstore::NativeLibaryKind, String)> {
    let libraries = reader::get_doc(rbml::Doc::new(cdata.data()),
                                    tag_native_libraries);
    let mut result = Vec::new();
    reader::tagged_docs(libraries, tag_native_libraries_lib, |lib_doc| {
        let kind_doc = reader::get_doc(lib_doc, tag_native_libraries_kind);
        let name_doc = reader::get_doc(lib_doc, tag_native_libraries_name);
        let kind: cstore::NativeLibaryKind =
            FromPrimitive::from_u32(reader::doc_as_u32(kind_doc)).unwrap();
        let name = name_doc.as_str().to_string();
        result.push((kind, name));
        true
    });
    return result;
}

pub fn get_plugin_registrar_fn(data: &[u8]) -> Option<ast::NodeId> {
    reader::maybe_get_doc(rbml::Doc::new(data), tag_plugin_registrar_fn)
        .map(|doc| FromPrimitive::from_u32(reader::doc_as_u32(doc)).unwrap())
}

pub fn get_exported_macros(data: &[u8]) -> Vec<String> {
    let macros = reader::get_doc(rbml::Doc::new(data),
                                 tag_exported_macros);
    let mut result = Vec::new();
    reader::tagged_docs(macros, tag_macro_def, |macro_doc| {
        result.push(macro_doc.as_str().to_string());
        true
    });
    result
}

pub fn get_dylib_dependency_formats(cdata: Cmd)
    -> Vec<(ast::CrateNum, cstore::LinkagePreference)>
{
    let formats = reader::get_doc(rbml::Doc::new(cdata.data()),
                                  tag_dylib_dependency_formats);
    let mut result = Vec::new();

    debug!("found dylib deps: {}", formats.as_str_slice());
    for spec in formats.as_str_slice().split(',') {
        if spec.len() == 0 { continue }
        let cnum = spec.split(':').nth(0).unwrap();
        let link = spec.split(':').nth(1).unwrap();
        let cnum = from_str(cnum).unwrap();
        let cnum = match cdata.cnum_map.find(&cnum) {
            Some(&n) => n,
            None => fail!("didn't find a crate in the cnum_map")
        };
        result.push((cnum, if link == "d" {
            cstore::RequireDynamic
        } else {
            cstore::RequireStatic
        }));
    }
    return result;
}

pub fn get_missing_lang_items(cdata: Cmd)
    -> Vec<lang_items::LangItem>
{
    let items = reader::get_doc(rbml::Doc::new(cdata.data()), tag_lang_items);
    let mut result = Vec::new();
    reader::tagged_docs(items, tag_lang_items_missing, |missing_doc| {
        let item: lang_items::LangItem =
            FromPrimitive::from_u32(reader::doc_as_u32(missing_doc)).unwrap();
        result.push(item);
        true
    });
    return result;
}

pub fn get_method_arg_names(cdata: Cmd, id: ast::NodeId) -> Vec<String> {
    let mut ret = Vec::new();
    let method_doc = lookup_item(id, cdata.data());
    match reader::maybe_get_doc(method_doc, tag_method_argument_names) {
        Some(args_doc) => {
            reader::tagged_docs(args_doc, tag_method_argument_name, |name_doc| {
                ret.push(name_doc.as_str_slice().to_string());
                true
            });
        }
        None => {}
    }
    return ret;
}

pub fn get_reachable_extern_fns(cdata: Cmd) -> Vec<ast::DefId> {
    let mut ret = Vec::new();
    let items = reader::get_doc(rbml::Doc::new(cdata.data()),
                                tag_reachable_extern_fns);
    reader::tagged_docs(items, tag_reachable_extern_fn_id, |doc| {
        ret.push(ast::DefId {
            krate: cdata.cnum,
            node: reader::doc_as_u32(doc),
        });
        true
    });
    return ret;
}

pub fn is_typedef(cdata: Cmd, id: ast::NodeId) -> bool {
    let item_doc = lookup_item(id, cdata.data());
    match item_family(item_doc) {
        Type => true,
        _ => false,
    }
}

fn doc_generics(base_doc: rbml::Doc,
                tcx: &ty::ctxt,
                cdata: Cmd,
                tag: uint)
                -> ty::Generics
{
    let doc = reader::get_doc(base_doc, tag);

    let mut types = subst::VecPerParamSpace::empty();
    reader::tagged_docs(doc, tag_type_param_def, |p| {
        let bd = parse_type_param_def_data(
            p.data, p.start, cdata.cnum, tcx,
            |_, did| translate_def_id(cdata, did));
        types.push(bd.space, bd);
        true
    });

    let mut regions = subst::VecPerParamSpace::empty();
    reader::tagged_docs(doc, tag_region_param_def, |rp_doc| {
        let ident_str_doc = reader::get_doc(rp_doc,
                                            tag_region_param_def_ident);
        let ident = item_name(&*token::get_ident_interner(), ident_str_doc);
        let def_id_doc = reader::get_doc(rp_doc,
                                         tag_region_param_def_def_id);
        let def_id = reader::with_doc_data(def_id_doc, parse_def_id);
        let def_id = translate_def_id(cdata, def_id);

        let doc = reader::get_doc(rp_doc, tag_region_param_def_space);
        let space = subst::ParamSpace::from_uint(reader::doc_as_u64(doc) as uint);

        let doc = reader::get_doc(rp_doc, tag_region_param_def_index);
        let index = reader::doc_as_u64(doc) as uint;

        let mut bounds = Vec::new();
        reader::tagged_docs(rp_doc, tag_items_data_region, |p| {
            bounds.push(
                parse_region_data(
                    p.data, cdata.cnum, p.start, tcx,
                    |_, did| translate_def_id(cdata, did)));
            true
        });

        regions.push(space, ty::RegionParameterDef { name: ident.name,
                                                     def_id: def_id,
                                                     space: space,
                                                     index: index,
                                                     bounds: bounds });

        true
    });

    ty::Generics { types: types, regions: regions }
}

pub fn is_associated_type(cdata: Cmd, id: ast::NodeId) -> bool {
    let items = reader::get_doc(rbml::Doc::new(cdata.data()), tag_items);
    match maybe_find_item(id, items) {
        None => false,
        Some(item) => item_sort(item) == 't',
    }
}

