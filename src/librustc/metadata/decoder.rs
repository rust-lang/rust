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

#[allow(non_camel_case_types)];

use metadata::cstore::crate_metadata;
use metadata::common::*;
use metadata::csearch::StaticMethodInfo;
use metadata::csearch;
use metadata::cstore;
use metadata::decoder;
use metadata::tydecode::{parse_ty_data, parse_def_id,
                         parse_type_param_def_data,
                         parse_bare_fn_ty_data, parse_trait_ref_data};
use middle::ty::{ImplContainer, TraitContainer};
use middle::ty;
use middle::typeck;
use middle::astencode::vtable_decoder_helpers;

use std::u64;
use std::hash_old::Hash;
use std::io;
use std::io::extensions::u64_from_be_bytes;
use std::option;
use std::rc::Rc;
use std::vec;
use serialize::ebml::reader;
use serialize::ebml;
use serialize::Decodable;
use syntax::ast_map;
use syntax::attr;
use syntax::parse::token::{IdentInterner, special_idents};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ast;
use syntax::codemap;

type Cmd = @crate_metadata;

// A function that takes a def_id relative to the crate being searched and
// returns a def_id relative to the compilation environment, i.e. if we hit a
// def_id for an item defined in another crate, somebody needs to figure out
// what crate that's in and give us a def_id that makes sense for the current
// build.

fn lookup_hash<'a>(d: ebml::Doc<'a>, eq_fn: |&[u8]| -> bool,
                   hash: u64) -> Option<ebml::Doc<'a>> {
    let index = reader::get_doc(d, tag_index);
    let table = reader::get_doc(index, tag_index_table);
    let hash_pos = table.start + (hash % 256 * 4) as uint;
    let pos = u64_from_be_bytes(d.data, hash_pos, 4) as uint;
    let tagged_doc = reader::doc_at(d.data, pos);

    let belt = tag_index_buckets_bucket_elt;

    let mut ret = None;
    reader::tagged_docs(tagged_doc.doc, belt, |elt| {
        let pos = u64_from_be_bytes(elt.data, elt.start, 4) as uint;
        if eq_fn(elt.data.slice(elt.start + 4, elt.end)) {
            ret = Some(reader::doc_at(d.data, pos).doc);
            false
        } else {
            true
        }
    });
    ret
}

pub type GetCrateDataCb<'a> = 'a |ast::CrateNum| -> Cmd;

pub fn maybe_find_item<'a>(item_id: ast::NodeId,
                           items: ebml::Doc<'a>) -> Option<ebml::Doc<'a>> {
    fn eq_item(bytes: &[u8], item_id: ast::NodeId) -> bool {
        return u64_from_be_bytes(
            bytes.slice(0u, 4u), 0u, 4u) as ast::NodeId
            == item_id;
    }
    lookup_hash(items,
                |a| eq_item(a, item_id),
                (item_id as i64).hash())
}

fn find_item<'a>(item_id: ast::NodeId, items: ebml::Doc<'a>) -> ebml::Doc<'a> {
    match maybe_find_item(item_id, items) {
       None => fail!("lookup_item: id not found: {}", item_id),
       Some(d) => d
    }
}

// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item<'a>(item_id: ast::NodeId, data: &'a [u8]) -> ebml::Doc<'a> {
    let items = reader::get_doc(reader::Doc(data), tag_items);
    find_item(item_id, items)
}

#[deriving(Eq)]
enum Family {
    ImmStatic,             // c
    MutStatic,             // b
    Fn,                    // f
    UnsafeFn,              // u
    StaticMethod,          // F
    UnsafeStaticMethod,    // U
    ForeignFn,             // e
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
    PrivateField,          // j
    InheritedField         // N
}

fn item_family(item: ebml::Doc) -> Family {
    let fam = reader::get_doc(item, tag_items_data_item_family);
    match reader::doc_as_u8(fam) as char {
      'c' => ImmStatic,
      'b' => MutStatic,
      'f' => Fn,
      'u' => UnsafeFn,
      'F' => StaticMethod,
      'U' => UnsafeStaticMethod,
      'e' => ForeignFn,
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
      'j' => PrivateField,
      'N' => InheritedField,
       c => fail!("unexpected family char: {}", c)
    }
}

fn item_visibility(item: ebml::Doc) -> ast::Visibility {
    match reader::maybe_get_doc(item, tag_items_data_item_visibility) {
        None => ast::Public,
        Some(visibility_doc) => {
            match reader::doc_as_u8(visibility_doc) as char {
                'y' => ast::Public,
                'n' => ast::Private,
                'i' => ast::Inherited,
                _ => fail!("unknown visibility character")
            }
        }
    }
}

fn item_method_sort(item: ebml::Doc) -> char {
    let mut ret = 'r';
    reader::tagged_docs(item, tag_item_trait_method_sort, |doc| {
        ret = doc.as_str_slice()[0] as char;
        false
    });
    ret
}

fn item_symbol(item: ebml::Doc) -> ~str {
    reader::get_doc(item, tag_items_data_item_symbol).as_str()
}

fn item_parent_item(d: ebml::Doc) -> Option<ast::DefId> {
    let mut ret = None;
    reader::tagged_docs(d, tag_items_data_parent_item, |did| {
        ret = Some(reader::with_doc_data(did, parse_def_id));
        false
    });
    ret
}

fn item_reqd_and_translated_parent_item(cnum: ast::CrateNum,
                                        d: ebml::Doc) -> ast::DefId {
    let trait_did = item_parent_item(d).expect("item without parent");
    ast::DefId { krate: cnum, node: trait_did.node }
}

fn item_def_id(d: ebml::Doc, cdata: Cmd) -> ast::DefId {
    let tagdoc = reader::get_doc(d, tag_def_id);
    return translate_def_id(cdata, reader::with_doc_data(tagdoc, parse_def_id));
}

fn get_provided_source(d: ebml::Doc, cdata: Cmd) -> Option<ast::DefId> {
    reader::maybe_get_doc(d, tag_item_method_provided_source).map(|doc| {
        translate_def_id(cdata, reader::with_doc_data(doc, parse_def_id))
    })
}

fn each_reexport(d: ebml::Doc, f: |ebml::Doc| -> bool) -> bool {
    reader::tagged_docs(d, tag_items_data_item_reexport, f)
}

fn variant_disr_val(d: ebml::Doc) -> Option<ty::Disr> {
    reader::maybe_get_doc(d, tag_disr_val).and_then(|val_doc| {
        reader::with_doc_data(val_doc, |data| u64::parse_bytes(data, 10u))
    })
}

fn doc_type(doc: ebml::Doc, tcx: ty::ctxt, cdata: Cmd) -> ty::t {
    let tp = reader::get_doc(doc, tag_items_data_item_type);
    parse_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                  |_, did| translate_def_id(cdata, did))
}

fn doc_method_fty(doc: ebml::Doc, tcx: ty::ctxt, cdata: Cmd) -> ty::BareFnTy {
    let tp = reader::get_doc(doc, tag_item_method_fty);
    parse_bare_fn_ty_data(tp.data, cdata.cnum, tp.start, tcx,
                          |_, did| translate_def_id(cdata, did))
}

pub fn item_type(_item_id: ast::DefId, item: ebml::Doc,
                 tcx: ty::ctxt, cdata: Cmd) -> ty::t {
    doc_type(item, tcx, cdata)
}

fn doc_trait_ref(doc: ebml::Doc, tcx: ty::ctxt, cdata: Cmd) -> ty::TraitRef {
    parse_trait_ref_data(doc.data, cdata.cnum, doc.start, tcx,
                         |_, did| translate_def_id(cdata, did))
}

fn item_trait_ref(doc: ebml::Doc, tcx: ty::ctxt, cdata: Cmd) -> ty::TraitRef {
    let tp = reader::get_doc(doc, tag_item_trait_ref);
    doc_trait_ref(tp, tcx, cdata)
}

fn item_ty_param_defs(item: ebml::Doc,
                      tcx: ty::ctxt,
                      cdata: Cmd,
                      tag: uint)
                      -> Rc<~[ty::TypeParameterDef]> {
    let mut bounds = ~[];
    reader::tagged_docs(item, tag, |p| {
        let bd = parse_type_param_def_data(
            p.data, p.start, cdata.cnum, tcx,
            |_, did| translate_def_id(cdata, did));
        bounds.push(bd);
        true
    });
    Rc::new(bounds)
}

fn item_region_param_defs(item_doc: ebml::Doc, cdata: Cmd)
                          -> Rc<~[ty::RegionParameterDef]> {
    let mut v = ~[];
    reader::tagged_docs(item_doc, tag_region_param_def, |rp_doc| {
            let ident_str_doc = reader::get_doc(rp_doc,
                                                tag_region_param_def_ident);
            let ident = item_name(token::get_ident_interner(), ident_str_doc);
            let def_id_doc = reader::get_doc(rp_doc,
                                             tag_region_param_def_def_id);
            let def_id = reader::with_doc_data(def_id_doc, parse_def_id);
            let def_id = translate_def_id(cdata, def_id);
            v.push(ty::RegionParameterDef { ident: ident.name,
                                            def_id: def_id });
            true
        });
    Rc::new(v)
}

fn item_ty_param_count(item: ebml::Doc) -> uint {
    let mut n = 0u;
    reader::tagged_docs(item, tag_items_data_item_ty_param_bounds,
                      |_p| { n += 1u; true } );
    n
}

fn enum_variant_ids(item: ebml::Doc, cdata: Cmd) -> ~[ast::DefId] {
    let mut ids: ~[ast::DefId] = ~[];
    let v = tag_items_data_item_variant;
    reader::tagged_docs(item, v, |p| {
        let ext = reader::with_doc_data(p, parse_def_id);
        ids.push(ast::DefId { krate: cdata.cnum, node: ext.node });
        true
    });
    return ids;
}

fn item_path(item_doc: ebml::Doc) -> ~[ast_map::PathElem] {
    let path_doc = reader::get_doc(item_doc, tag_path);

    let len_doc = reader::get_doc(path_doc, tag_path_len);
    let len = reader::doc_as_u32(len_doc) as uint;

    let mut result = vec::with_capacity(len);
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

fn item_name(intr: &IdentInterner, item: ebml::Doc) -> ast::Ident {
    let name = reader::get_doc(item, tag_paths_data_name);
    let string = name.as_str_slice();
    match intr.find_equiv(&string) {
        None => token::str_to_ident(string),
        Some(val) => ast::Ident::new(val as ast::Name),
    }
}

fn item_to_def_like(item: ebml::Doc, did: ast::DefId, cnum: ast::CrateNum)
    -> DefLike {
    let fam = item_family(item);
    match fam {
        ImmStatic => DlDef(ast::DefStatic(did, false)),
        MutStatic => DlDef(ast::DefStatic(did, true)),
        Struct    => DlDef(ast::DefStruct(did)),
        UnsafeFn  => DlDef(ast::DefFn(did, ast::UnsafeFn)),
        Fn        => DlDef(ast::DefFn(did, ast::ImpureFn)),
        ForeignFn => DlDef(ast::DefFn(did, ast::ExternFn)),
        StaticMethod | UnsafeStaticMethod => {
            let purity = if fam == UnsafeStaticMethod { ast::UnsafeFn } else
                { ast::ImpureFn };
            // def_static_method carries an optional field of its enclosing
            // trait or enclosing impl (if this is an inherent static method).
            // So we need to detect whether this is in a trait or not, which
            // we do through the mildly hacky way of checking whether there is
            // a trait_method_sort.
            let provenance = if reader::maybe_get_doc(
                  item, tag_item_trait_method_sort).is_some() {
                ast::FromTrait(item_reqd_and_translated_parent_item(cnum,
                                                                    item))
            } else {
                ast::FromImpl(item_reqd_and_translated_parent_item(cnum,
                                                                   item))
            };
            DlDef(ast::DefStaticMethod(did, provenance, purity))
        }
        Type | ForeignType => DlDef(ast::DefTy(did)),
        Mod => DlDef(ast::DefMod(did)),
        ForeignMod => DlDef(ast::DefForeignMod(did)),
        StructVariant => {
            let enum_did = item_reqd_and_translated_parent_item(cnum, item);
            DlDef(ast::DefVariant(enum_did, did, true))
        }
        TupleVariant => {
            let enum_did = item_reqd_and_translated_parent_item(cnum, item);
            DlDef(ast::DefVariant(enum_did, did, false))
        }
        Trait => DlDef(ast::DefTrait(did)),
        Enum => DlDef(ast::DefTy(did)),
        Impl => DlImpl(did),
        PublicField | PrivateField | InheritedField => DlField,
    }
}

pub fn get_trait_def(cdata: Cmd,
                     item_id: ast::NodeId,
                     tcx: ty::ctxt) -> ty::TraitDef
{
    let item_doc = lookup_item(item_id, cdata.data());
    let tp_defs = item_ty_param_defs(item_doc, tcx, cdata,
                                     tag_items_data_item_ty_param_bounds);
    let rp_defs = item_region_param_defs(item_doc, cdata);
    let mut bounds = ty::EmptyBuiltinBounds();
    // Collect the builtin bounds from the encoded supertraits.
    // FIXME(#8559): They should be encoded directly.
    reader::tagged_docs(item_doc, tag_item_super_trait_ref, |trait_doc| {
        // NB. Bypasses real supertraits. See get_supertraits() if you wanted them.
        let trait_ref = doc_trait_ref(trait_doc, tcx, cdata);
        tcx.lang_items.to_builtin_kind(trait_ref.def_id).map(|bound| {
            bounds.add(bound);
        });
        true
    });
    ty::TraitDef {
        generics: ty::Generics {type_param_defs: tp_defs,
                                region_param_defs: rp_defs},
        bounds: bounds,
        trait_ref: @item_trait_ref(item_doc, tcx, cdata)
    }
}

pub fn get_type(cdata: Cmd, id: ast::NodeId, tcx: ty::ctxt)
    -> ty::ty_param_bounds_and_ty {

    let item = lookup_item(id, cdata.data());

    let t = item_type(ast::DefId { krate: cdata.cnum, node: id }, item, tcx,
                      cdata);

    let tp_defs = item_ty_param_defs(item, tcx, cdata, tag_items_data_item_ty_param_bounds);
    let rp_defs = item_region_param_defs(item, cdata);

    ty::ty_param_bounds_and_ty {
        generics: ty::Generics {type_param_defs: tp_defs,
                                region_param_defs: rp_defs},
        ty: t
    }
}

pub fn get_type_param_count(data: &[u8], id: ast::NodeId) -> uint {
    item_ty_param_count(lookup_item(id, data))
}

pub fn get_impl_trait(cdata: Cmd,
                      id: ast::NodeId,
                      tcx: ty::ctxt) -> Option<@ty::TraitRef>
{
    let item_doc = lookup_item(id, cdata.data());
    reader::maybe_get_doc(item_doc, tag_item_trait_ref).map(|tp| {
        @doc_trait_ref(tp, tcx, cdata)
    })
}

pub fn get_impl_vtables(cdata: Cmd,
                        id: ast::NodeId,
                        tcx: ty::ctxt) -> typeck::impl_res
{
    let item_doc = lookup_item(id, cdata.data());
    let vtables_doc = reader::get_doc(item_doc, tag_item_impl_vtables);
    let mut decoder = reader::Decoder(vtables_doc);

    typeck::impl_res {
        trait_vtables: decoder.read_vtable_res(tcx, cdata),
        self_vtables: decoder.read_vtable_param_res(tcx, cdata)
    }
}


pub fn get_impl_method(intr: @IdentInterner, cdata: Cmd, id: ast::NodeId,
                       name: ast::Ident) -> Option<ast::DefId> {
    let items = reader::get_doc(reader::Doc(cdata.data()), tag_items);
    let mut found = None;
    reader::tagged_docs(find_item(id, items), tag_item_impl_method, |mid| {
        let m_did = reader::with_doc_data(mid, parse_def_id);
        if item_name(intr, find_item(m_did.node, items)) == name {
            found = Some(translate_def_id(cdata, m_did));
        }
        true
    });
    found
}

pub fn get_symbol(data: &[u8], id: ast::NodeId) -> ~str {
    return item_symbol(lookup_item(id, data));
}

// Something that a name can resolve to.
#[deriving(Clone)]
pub enum DefLike {
    DlDef(ast::Def),
    DlImpl(ast::DefId),
    DlField
}

pub fn def_like_to_def(def_like: DefLike) -> ast::Def {
    match def_like {
        DlDef(def) => return def,
        DlImpl(..) => fail!("found impl in def_like_to_def"),
        DlField => fail!("found field in def_like_to_def")
    }
}

/// Iterates over the language items in the given crate.
pub fn each_lang_item(cdata: Cmd, f: |ast::NodeId, uint| -> bool) -> bool {
    let root = reader::Doc(cdata.data());
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

fn each_child_of_item_or_crate(intr: @IdentInterner,
                               cdata: Cmd,
                               item_doc: ebml::Doc,
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
        let other_crates_items = if child_def_id.krate == cdata.cnum {
            reader::get_doc(reader::Doc(cdata.data()), tag_items)
        } else {
            let crate_data = get_crate_data(child_def_id.krate);
            reader::get_doc(reader::Doc(crate_data.data()), tag_items)
        };

        // Get the item.
        match maybe_find_item(child_def_id.node, other_crates_items) {
            None => {}
            Some(child_item_doc) => {
                // Hand off the item to the callback.
                let child_name = item_name(intr, child_item_doc);
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
        let items = reader::get_doc(reader::Doc(cdata.data()), tag_items);
        match maybe_find_item(inherent_impl_def_id.node, items) {
            None => {}
            Some(inherent_impl_doc) => {
                let _ = reader::tagged_docs(inherent_impl_doc,
                                            tag_item_impl_method,
                                            |impl_method_def_id_doc| {
                    let impl_method_def_id =
                        reader::with_doc_data(impl_method_def_id_doc,
                                              parse_def_id);
                    let impl_method_def_id =
                        translate_def_id(cdata, impl_method_def_id);
                    match maybe_find_item(impl_method_def_id.node, items) {
                        None => {}
                        Some(impl_method_doc) => {
                            match item_family(impl_method_doc) {
                                StaticMethod | UnsafeStaticMethod => {
                                    // Hand off the static method
                                    // to the callback.
                                    let static_method_name =
                                        item_name(intr, impl_method_doc);
                                    let static_method_def_like =
                                        item_to_def_like(impl_method_doc,
                                                         impl_method_def_id,
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
        let other_crates_items = if child_def_id.krate == cdata.cnum {
            reader::get_doc(reader::Doc(cdata.data()), tag_items)
        } else {
            let crate_data = get_crate_data(child_def_id.krate);
            reader::get_doc(reader::Doc(crate_data.data()), tag_items)
        };

        // Get the item.
        match maybe_find_item(child_def_id.node, other_crates_items) {
            None => {}
            Some(child_item_doc) => {
                // Hand off the item to the callback.
                let def_like = item_to_def_like(child_item_doc,
                                                child_def_id,
                                                cdata.cnum);
                // These items have a public visibility because they're part of
                // a public re-export.
                callback(def_like, token::str_to_ident(name), ast::Public);
            }
        }

        true
    });
}

/// Iterates over each child of the given item.
pub fn each_child_of_item(intr: @IdentInterner,
                          cdata: Cmd,
                          id: ast::NodeId,
                          get_crate_data: GetCrateDataCb,
                          callback: |DefLike, ast::Ident, ast::Visibility|) {
    // Find the item.
    let root_doc = reader::Doc(cdata.data());
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
pub fn each_top_level_item_of_crate(intr: @IdentInterner,
                                    cdata: Cmd,
                                    get_crate_data: GetCrateDataCb,
                                    callback: |DefLike,
                                               ast::Ident,
                                               ast::Visibility|) {
    let root_doc = reader::Doc(cdata.data());
    let misc_info_doc = reader::get_doc(root_doc, tag_misc_info);
    let crate_items_doc = reader::get_doc(misc_info_doc,
                                          tag_misc_info_crate_items);

    each_child_of_item_or_crate(intr,
                                cdata,
                                crate_items_doc,
                                get_crate_data,
                                callback)
}

pub fn get_item_path(cdata: Cmd, id: ast::NodeId) -> ~[ast_map::PathElem] {
    item_path(lookup_item(id, cdata.data()))
}

pub type DecodeInlinedItem<'a> = 'a |cdata: @cstore::crate_metadata,
                                     tcx: ty::ctxt,
                                     path: ~[ast_map::PathElem],
                                     par_doc: ebml::Doc|
                                     -> Result<ast::InlinedItem, ~[ast_map::PathElem]>;

pub fn maybe_get_item_ast(cdata: Cmd, tcx: ty::ctxt, id: ast::NodeId,
                          decode_inlined_item: DecodeInlinedItem)
                          -> csearch::found_ast {
    debug!("Looking up item: {}", id);
    let item_doc = lookup_item(id, cdata.data());
    let path = item_path(item_doc).init().to_owned();
    match decode_inlined_item(cdata, tcx, path, item_doc) {
        Ok(ref ii) => csearch::found(*ii),
        Err(path) => {
            match item_parent_item(item_doc) {
                Some(did) => {
                    let did = translate_def_id(cdata, did);
                    let parent_item = lookup_item(did.node, cdata.data());
                    match decode_inlined_item(cdata, tcx, path, parent_item) {
                        Ok(ref ii) => csearch::found_parent(did, *ii),
                        Err(_) => csearch::not_found
                    }
                }
                None => csearch::not_found
            }
        }
    }
}

pub fn get_enum_variants(intr: @IdentInterner, cdata: Cmd, id: ast::NodeId,
                     tcx: ty::ctxt) -> ~[@ty::VariantInfo] {
    let data = cdata.data();
    let items = reader::get_doc(reader::Doc(data), tag_items);
    let item = find_item(id, items);
    let mut infos: ~[@ty::VariantInfo] = ~[];
    let variant_ids = enum_variant_ids(item, cdata);
    let mut disr_val = 0;
    for did in variant_ids.iter() {
        let item = find_item(did.node, items);
        let ctor_ty = item_type(ast::DefId { krate: cdata.cnum, node: id},
                                item, tcx, cdata);
        let name = item_name(intr, item);
        let arg_tys = match ty::get(ctor_ty).sty {
          ty::ty_bare_fn(ref f) => f.sig.inputs.clone(),
          _ => ~[], // Nullary enum variant.
        };
        match variant_disr_val(item) {
          Some(val) => { disr_val = val; }
          _         => { /* empty */ }
        }
        infos.push(@ty::VariantInfo{
            args: arg_tys,
            arg_names: None,
            ctor_ty: ctor_ty,
            name: name,
            // I'm not even sure if we encode visibility
            // for variants -- TEST -- tjc
            id: *did,
            disr_val: disr_val,
            vis: ast::Inherited});
        disr_val += 1;
    }
    return infos;
}

fn get_explicit_self(item: ebml::Doc) -> ast::ExplicitSelf_ {
    fn get_mutability(ch: u8) -> ast::Mutability {
        match ch as char {
            'i' => ast::MutImmutable,
            'm' => ast::MutMutable,
            _ => fail!("unknown mutability character: `{}`", ch as char),
        }
    }

    let explicit_self_doc = reader::get_doc(item, tag_item_trait_method_explicit_self);
    let string = explicit_self_doc.as_str_slice();

    let explicit_self_kind = string[0];
    match explicit_self_kind as char {
        's' => ast::SelfStatic,
        'v' => ast::SelfValue,
        '~' => ast::SelfUniq,
        // FIXME(#4846) expl. region
        '&' => ast::SelfRegion(None, get_mutability(string[1])),
        _ => fail!("unknown self type code: `{}`", explicit_self_kind as char)
    }
}

fn item_impl_methods(intr: @IdentInterner, cdata: Cmd, item: ebml::Doc,
                     tcx: ty::ctxt) -> ~[@ty::Method] {
    let mut rslt = ~[];
    reader::tagged_docs(item, tag_item_impl_method, |doc| {
        let m_did = reader::with_doc_data(doc, parse_def_id);
        rslt.push(@get_method(intr, cdata, m_did.node, tcx));
        true
    });

    rslt
}

/// Returns information about the given implementation.
pub fn get_impl(intr: @IdentInterner, cdata: Cmd, impl_id: ast::NodeId,
               tcx: ty::ctxt)
                -> ty::Impl {
    let data = cdata.data();
    let impl_item = lookup_item(impl_id, data);
    ty::Impl {
        did: ast::DefId {
            krate: cdata.cnum,
            node: impl_id,
        },
        ident: item_name(intr, impl_item),
        methods: item_impl_methods(intr, cdata, impl_item, tcx),
    }
}

pub fn get_method_name_and_explicit_self(
    intr: @IdentInterner,
    cdata: Cmd,
    id: ast::NodeId) -> (ast::Ident, ast::ExplicitSelf_)
{
    let method_doc = lookup_item(id, cdata.data());
    let name = item_name(intr, method_doc);
    let explicit_self = get_explicit_self(method_doc);
    (name, explicit_self)
}

pub fn get_method(intr: @IdentInterner, cdata: Cmd, id: ast::NodeId,
                  tcx: ty::ctxt) -> ty::Method
{
    let method_doc = lookup_item(id, cdata.data());
    let def_id = item_def_id(method_doc, cdata);

    let container_id = item_reqd_and_translated_parent_item(cdata.cnum,
                                                            method_doc);
    let container_doc = lookup_item(container_id.node, cdata.data());
    let container = match item_family(container_doc) {
        Trait => TraitContainer(container_id),
        _ => ImplContainer(container_id),
    };

    let name = item_name(intr, method_doc);
    let type_param_defs = item_ty_param_defs(method_doc, tcx, cdata,
                                             tag_item_method_tps);
    let rp_defs = item_region_param_defs(method_doc, cdata);
    let fty = doc_method_fty(method_doc, tcx, cdata);
    let vis = item_visibility(method_doc);
    let explicit_self = get_explicit_self(method_doc);
    let provided_source = get_provided_source(method_doc, cdata);

    ty::Method::new(
        name,
        ty::Generics {
            type_param_defs: type_param_defs,
            region_param_defs: rp_defs,
        },
        fty,
        explicit_self,
        vis,
        def_id,
        container,
        provided_source
    )
}

pub fn get_trait_method_def_ids(cdata: Cmd,
                                id: ast::NodeId) -> ~[ast::DefId] {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = ~[];
    reader::tagged_docs(item, tag_item_trait_method, |mth| {
        result.push(item_def_id(mth, cdata));
        true
    });
    result
}

pub fn get_item_variances(cdata: Cmd, id: ast::NodeId) -> ty::ItemVariances {
    let data = cdata.data();
    let item_doc = lookup_item(id, data);
    let variance_doc = reader::get_doc(item_doc, tag_item_variances);
    let mut decoder = reader::Decoder(variance_doc);
    Decodable::decode(&mut decoder)
}

pub fn get_provided_trait_methods(intr: @IdentInterner, cdata: Cmd,
                                  id: ast::NodeId, tcx: ty::ctxt) ->
        ~[@ty::Method] {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = ~[];

    reader::tagged_docs(item, tag_item_trait_method, |mth_id| {
        let did = item_def_id(mth_id, cdata);
        let mth = lookup_item(did.node, data);

        if item_method_sort(mth) == 'p' {
            result.push(@get_method(intr, cdata, did.node, tcx));
        }
        true
    });

    return result;
}

/// Returns the supertraits of the given trait.
pub fn get_supertraits(cdata: Cmd, id: ast::NodeId, tcx: ty::ctxt)
                    -> ~[@ty::TraitRef] {
    let mut results = ~[];
    let item_doc = lookup_item(id, cdata.data());
    reader::tagged_docs(item_doc, tag_item_super_trait_ref, |trait_doc| {
        // NB. Only reads the ones that *aren't* builtin-bounds. See also
        // get_trait_def() for collecting the builtin bounds.
        // FIXME(#8559): The builtin bounds shouldn't be encoded in the first place.
        let trait_ref = doc_trait_ref(trait_doc, tcx, cdata);
        if tcx.lang_items.to_builtin_kind(trait_ref.def_id).is_none() {
            results.push(@trait_ref);
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

pub fn get_static_methods_if_impl(intr: @IdentInterner,
                                  cdata: Cmd,
                                  node_id: ast::NodeId)
                               -> Option<~[StaticMethodInfo]> {
    let item = lookup_item(node_id, cdata.data());
    if item_family(item) != Impl {
        return None;
    }

    // If this impl implements a trait, don't consider it.
    let ret = reader::tagged_docs(item, tag_item_trait_ref, |_doc| {
        false
    });

    if !ret { return None }

    let mut impl_method_ids = ~[];
    reader::tagged_docs(item, tag_item_impl_method, |impl_method_doc| {
        impl_method_ids.push(reader::with_doc_data(impl_method_doc, parse_def_id));
        true
    });

    let mut static_impl_methods = ~[];
    for impl_method_id in impl_method_ids.iter() {
        let impl_method_doc = lookup_item(impl_method_id.node, cdata.data());
        let family = item_family(impl_method_doc);
        match family {
            StaticMethod | UnsafeStaticMethod => {
                let purity;
                match item_family(impl_method_doc) {
                    StaticMethod => purity = ast::ImpureFn,
                    UnsafeStaticMethod => purity = ast::UnsafeFn,
                    _ => fail!()
                }

                static_impl_methods.push(StaticMethodInfo {
                    ident: item_name(intr, impl_method_doc),
                    def_id: item_def_id(impl_method_doc, cdata),
                    purity: purity,
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
                                           node_id: ast::NodeId) -> Option<ast::NodeId> {
    let item = lookup_item(node_id, cdata.data());
    let mut ret = None;
    reader::tagged_docs(item, tag_items_data_item_is_tuple_struct_ctor, |_| {
        ret = Some(item_reqd_and_translated_parent_item(cdata.cnum, item));
        false
    });
    ret.map(|x| x.node)
}

pub fn get_item_attrs(cdata: Cmd,
                      node_id: ast::NodeId,
                      f: |~[@ast::MetaItem]|) {
    // The attributes for a tuple struct are attached to the definition, not the ctor;
    // we assume that someone passing in a tuple struct ctor is actually wanting to
    // look at the definition
    let node_id = get_tuple_struct_definition_if_ctor(cdata, node_id).unwrap_or(node_id);
    let item = lookup_item(node_id, cdata.data());
    reader::tagged_docs(item, tag_attributes, |attributes| {
        reader::tagged_docs(attributes, tag_attribute, |attribute| {
            f(get_meta_items(attribute));
            true
        });
        true
    });
}

fn struct_field_family_to_visibility(family: Family) -> ast::Visibility {
    match family {
      PublicField => ast::Public,
      PrivateField => ast::Private,
      InheritedField => ast::Inherited,
      _ => fail!()
    }
}

pub fn get_struct_fields(intr: @IdentInterner, cdata: Cmd, id: ast::NodeId)
    -> ~[ty::field_ty] {
    let data = cdata.data();
    let item = lookup_item(id, data);
    let mut result = ~[];
    reader::tagged_docs(item, tag_item_field, |an_item| {
        let f = item_family(an_item);
        if f == PublicField || f == PrivateField || f == InheritedField {
            // FIXME #6993: name should be of type Name, not Ident
            let name = item_name(intr, an_item);
            let did = item_def_id(an_item, cdata);
            result.push(ty::field_ty {
                name: name.name,
                id: did, vis:
                struct_field_family_to_visibility(f),
            });
        }
        true
    });
    reader::tagged_docs(item, tag_item_unnamed_field, |an_item| {
        let did = item_def_id(an_item, cdata);
        result.push(ty::field_ty {
            name: special_idents::unnamed_field.name,
            id: did,
            vis: ast::Inherited,
        });
        true
    });
    result
}

pub fn get_item_visibility(cdata: Cmd, id: ast::NodeId)
                        -> ast::Visibility {
    item_visibility(lookup_item(id, cdata.data()))
}

fn get_meta_items(md: ebml::Doc) -> ~[@ast::MetaItem] {
    let mut items: ~[@ast::MetaItem] = ~[];
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
        items.push(attr::mk_list_item(n, subitems));
        true
    });
    return items;
}

fn get_attributes(md: ebml::Doc) -> ~[ast::Attribute] {
    let mut attrs: ~[ast::Attribute] = ~[];
    match reader::maybe_get_doc(md, tag_attributes) {
      option::Some(attrs_d) => {
        reader::tagged_docs(attrs_d, tag_attribute, |attr_doc| {
            let meta_items = get_meta_items(attr_doc);
            // Currently it's only possible to have a single meta item on
            // an attribute
            assert_eq!(meta_items.len(), 1u);
            let meta_item = meta_items[0];
            attrs.push(
                codemap::Spanned {
                    node: ast::Attribute_ {
                        style: ast::AttrOuter,
                        value: meta_item,
                        is_sugared_doc: false,
                    },
                    span: codemap::DUMMY_SP
                });
            true
        });
      }
      option::None => ()
    }
    return attrs;
}

fn list_crate_attributes(md: ebml::Doc, hash: &str,
                         out: &mut io::Writer) -> io::IoResult<()> {
    try!(write!(out, "=Crate Attributes ({})=\n", hash));

    let r = get_attributes(md);
    for attr in r.iter() {
        try!(write!(out, "{}\n", pprust::attribute_to_str(attr)));
    }

    write!(out, "\n\n")
}

pub fn get_crate_attributes(data: &[u8]) -> ~[ast::Attribute] {
    return get_attributes(reader::Doc(data));
}

#[deriving(Clone)]
pub struct CrateDep {
    cnum: ast::CrateNum,
    name: ast::Ident,
    vers: ~str,
    hash: ~str
}

pub fn get_crate_deps(data: &[u8]) -> ~[CrateDep] {
    let mut deps: ~[CrateDep] = ~[];
    let cratedoc = reader::Doc(data);
    let depsdoc = reader::get_doc(cratedoc, tag_crate_deps);
    let mut crate_num = 1;
    fn docstr(doc: ebml::Doc, tag_: uint) -> ~str {
        let d = reader::get_doc(doc, tag_);
        d.as_str_slice().to_str()
    }
    reader::tagged_docs(depsdoc, tag_crate_dep, |depdoc| {
        deps.push(CrateDep {cnum: crate_num,
                  name: token::str_to_ident(docstr(depdoc, tag_crate_dep_name)),
                  vers: docstr(depdoc, tag_crate_dep_vers),
                  hash: docstr(depdoc, tag_crate_dep_hash)});
        crate_num += 1;
        true
    });
    return deps;
}

fn list_crate_deps(data: &[u8], out: &mut io::Writer) -> io::IoResult<()> {
    try!(write!(out, "=External Dependencies=\n"));

    let r = get_crate_deps(data);
    for dep in r.iter() {
        try!(write!(out,
                      "{} {}-{}-{}\n",
                      dep.cnum,
                      token::get_ident(dep.name),
                      dep.hash,
                      dep.vers));
    }

    try!(write!(out, "\n"));
    Ok(())
}

pub fn get_crate_hash(data: &[u8]) -> ~str {
    let cratedoc = reader::Doc(data);
    let hashdoc = reader::get_doc(cratedoc, tag_crate_hash);
    hashdoc.as_str_slice().to_str()
}

pub fn get_crate_vers(data: &[u8]) -> ~str {
    let attrs = decoder::get_crate_attributes(data);
    match attr::find_crateid(attrs) {
        None => ~"0.0",
        Some(crateid) => crateid.version_or_default().to_str(),
    }
}

pub fn list_crate_metadata(bytes: &[u8], out: &mut io::Writer) -> io::IoResult<()> {
    let hash = get_crate_hash(bytes);
    let md = reader::Doc(bytes);
    try!(list_crate_attributes(md, hash, out));
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

    let cnum_map = cdata.cnum_map.borrow();
    match cnum_map.get().find(&did.krate) {
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
    let impls_doc = reader::get_doc(reader::Doc(cdata.data()), tag_impls);
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

pub fn get_trait_of_method(cdata: Cmd, id: ast::NodeId, tcx: ty::ctxt)
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


pub fn get_native_libraries(cdata: Cmd) -> ~[(cstore::NativeLibaryKind, ~str)] {
    let libraries = reader::get_doc(reader::Doc(cdata.data()),
                                    tag_native_libraries);
    let mut result = ~[];
    reader::tagged_docs(libraries, tag_native_libraries_lib, |lib_doc| {
        let kind_doc = reader::get_doc(lib_doc, tag_native_libraries_kind);
        let name_doc = reader::get_doc(lib_doc, tag_native_libraries_name);
        let kind: cstore::NativeLibaryKind =
            FromPrimitive::from_u32(reader::doc_as_u32(kind_doc)).unwrap();
        let name = name_doc.as_str();
        result.push((kind, name));
        true
    });
    return result;
}

pub fn get_macro_registrar_fn(cdata: Cmd) -> Option<ast::DefId> {
    reader::maybe_get_doc(reader::Doc(cdata.data()), tag_macro_registrar_fn)
        .map(|doc| item_def_id(doc, cdata))
}

pub fn get_exported_macros(cdata: Cmd) -> ~[~str] {
    let macros = reader::get_doc(reader::Doc(cdata.data()),
                                 tag_exported_macros);
    let mut result = ~[];
    reader::tagged_docs(macros, tag_macro_def, |macro_doc| {
        result.push(macro_doc.as_str());
        true
    });
    result
}
