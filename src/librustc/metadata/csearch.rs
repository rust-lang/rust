// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Searching for information from the cstore


use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use middle::ty;
use middle::typeck;

use std::vec;
use std::rc::Rc;
use reader = extra::ebml::reader;
use syntax::ast;
use syntax::ast_map;
use syntax::diagnostic::expect;

pub struct StaticMethodInfo {
    ident: ast::Ident,
    def_id: ast::DefId,
    purity: ast::Purity,
    vis: ast::Visibility,
}

pub fn get_symbol(cstore: @cstore::CStore, def: ast::DefId) -> ~str {
    let cdata = cstore.get_crate_data(def.crate).data();
    return decoder::get_symbol(cdata, def.node);
}

pub fn get_type_param_count(cstore: @cstore::CStore, def: ast::DefId)
                         -> uint {
    let cdata = cstore.get_crate_data(def.crate).data();
    return decoder::get_type_param_count(cdata, def.node);
}

/// Iterates over all the language items in the given crate.
pub fn each_lang_item(cstore: @cstore::CStore,
                      cnum: ast::CrateNum,
                      f: |ast::NodeId, uint| -> bool)
                      -> bool {
    let crate_data = cstore.get_crate_data(cnum);
    decoder::each_lang_item(crate_data, f)
}

/// Iterates over each child of the given item.
pub fn each_child_of_item(cstore: @cstore::CStore,
                          def_id: ast::DefId,
                          callback: |decoder::DefLike,
                                     ast::Ident,
                                     ast::Visibility|) {
    let crate_data = cstore.get_crate_data(def_id.crate);
    let get_crate_data: decoder::GetCrateDataCb = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_child_of_item(cstore.intr,
                                crate_data,
                                def_id.node,
                                get_crate_data,
                                callback)
}

/// Iterates over each top-level crate item.
pub fn each_top_level_item_of_crate(cstore: @cstore::CStore,
                                    cnum: ast::CrateNum,
                                    callback: |decoder::DefLike,
                                               ast::Ident,
                                               ast::Visibility|) {
    let crate_data = cstore.get_crate_data(cnum);
    let get_crate_data: decoder::GetCrateDataCb = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_top_level_item_of_crate(cstore.intr,
                                          crate_data,
                                          get_crate_data,
                                          callback)
}

pub fn get_item_path(tcx: ty::ctxt, def: ast::DefId) -> ast_map::Path {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    let path = decoder::get_item_path(cdata, def.node);

    // FIXME #1920: This path is not always correct if the crate is not linked
    // into the root namespace.
    vec::append(~[ast_map::PathMod(tcx.sess.ident_of(
        cdata.name))], path)
}

pub enum found_ast {
    found(ast::InlinedItem),
    found_parent(ast::DefId, ast::InlinedItem),
    not_found,
}

// Finds the AST for this item in the crate metadata, if any.  If the item was
// not marked for inlining, then the AST will not be present and hence none
// will be returned.
pub fn maybe_get_item_ast(tcx: ty::ctxt, def: ast::DefId,
                          decode_inlined_item: decoder::decode_inlined_item)
                       -> found_ast {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::maybe_get_item_ast(cdata, tcx, def.node,
                                decode_inlined_item)
}

pub fn get_enum_variants(tcx: ty::ctxt, def: ast::DefId)
                      -> ~[@ty::VariantInfo] {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    return decoder::get_enum_variants(cstore.intr, cdata, def.node, tcx)
}

/// Returns information about the given implementation.
pub fn get_impl(tcx: ty::ctxt, impl_def_id: ast::DefId)
                -> ty::Impl {
    let cdata = tcx.cstore.get_crate_data(impl_def_id.crate);
    decoder::get_impl(tcx.cstore.intr, cdata, impl_def_id.node, tcx)
}

pub fn get_method(tcx: ty::ctxt, def: ast::DefId) -> ty::Method {
    let cdata = tcx.cstore.get_crate_data(def.crate);
    decoder::get_method(tcx.cstore.intr, cdata, def.node, tcx)
}

pub fn get_method_name_and_explicit_self(cstore: @cstore::CStore,
                                         def: ast::DefId)
                                     -> (ast::Ident, ast::ExplicitSelf_)
{
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_method_name_and_explicit_self(cstore.intr, cdata, def.node)
}

pub fn get_trait_method_def_ids(cstore: @cstore::CStore,
                                def: ast::DefId) -> ~[ast::DefId] {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_trait_method_def_ids(cdata, def.node)
}

pub fn get_item_variances(cstore: @cstore::CStore,
                          def: ast::DefId) -> ty::ItemVariances {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_item_variances(cdata, def.node)
}

pub fn get_provided_trait_methods(tcx: ty::ctxt,
                                  def: ast::DefId)
                               -> ~[@ty::Method] {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_provided_trait_methods(cstore.intr, cdata, def.node, tcx)
}

pub fn get_supertraits(tcx: ty::ctxt, def: ast::DefId) -> ~[@ty::TraitRef] {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_supertraits(cdata, def.node, tcx)
}

pub fn get_type_name_if_impl(cstore: @cstore::CStore, def: ast::DefId)
                          -> Option<ast::Ident> {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_type_name_if_impl(cdata, def.node)
}

pub fn get_static_methods_if_impl(cstore: @cstore::CStore,
                                  def: ast::DefId)
                               -> Option<~[StaticMethodInfo]> {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_static_methods_if_impl(cstore.intr, cdata, def.node)
}

pub fn get_item_attrs(cstore: @cstore::CStore,
                      def_id: ast::DefId,
                      f: |~[@ast::MetaItem]|) {
    let cdata = cstore.get_crate_data(def_id.crate);
    decoder::get_item_attrs(cdata, def_id.node, f)
}

pub fn get_struct_fields(cstore: @cstore::CStore,
                         def: ast::DefId)
                      -> ~[ty::field_ty] {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_struct_fields(cstore.intr, cdata, def.node)
}

pub fn get_type(tcx: ty::ctxt,
                def: ast::DefId)
             -> ty::ty_param_bounds_and_ty {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_type(cdata, def.node, tcx)
}

pub fn get_trait_def(tcx: ty::ctxt, def: ast::DefId) -> ty::TraitDef {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_trait_def(cdata, def.node, tcx)
}

pub fn get_field_type(tcx: ty::ctxt, class_id: ast::DefId,
                      def: ast::DefId) -> ty::ty_param_bounds_and_ty {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(class_id.crate);
    let all_items = reader::get_doc(reader::Doc(cdata.data()), tag_items);
    let class_doc = expect(tcx.diag,
                           decoder::maybe_find_item(class_id.node, all_items),
                           || format!("get_field_type: class ID {:?} not found",
                                   class_id) );
    let the_field = expect(tcx.diag,
        decoder::maybe_find_item(def.node, class_doc),
        || format!("get_field_type: in class {:?}, field ID {:?} not found",
                 class_id, def) );
    let ty = decoder::item_type(def, the_field, tcx, cdata);
    ty::ty_param_bounds_and_ty {
        generics: ty::Generics {type_param_defs: Rc::new(~[]),
                                region_param_defs: Rc::new(~[])},
        ty: ty
    }
}

// Given a def_id for an impl, return the trait it implements,
// if there is one.
pub fn get_impl_trait(tcx: ty::ctxt,
                      def: ast::DefId) -> Option<@ty::TraitRef> {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_impl_trait(cdata, def.node, tcx)
}

// Given a def_id for an impl, return information about its vtables
pub fn get_impl_vtables(tcx: ty::ctxt,
                        def: ast::DefId) -> typeck::impl_res {
    let cstore = tcx.cstore;
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_impl_vtables(cdata, def.node, tcx)
}

pub fn get_impl_method(cstore: @cstore::CStore,
                       def: ast::DefId,
                       mname: ast::Ident)
                    -> Option<ast::DefId> {
    let cdata = cstore.get_crate_data(def.crate);
    decoder::get_impl_method(cstore.intr, cdata, def.node, mname)
}

pub fn get_item_visibility(cstore: @cstore::CStore,
                           def_id: ast::DefId)
                        -> ast::Visibility {
    let cdata = cstore.get_crate_data(def_id.crate);
    decoder::get_item_visibility(cdata, def_id.node)
}

pub fn get_native_libraries(cstore: @cstore::CStore,
                            crate_num: ast::CrateNum)
                                -> ~[(cstore::NativeLibaryKind, ~str)] {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::get_native_libraries(cdata)
}

pub fn each_impl(cstore: @cstore::CStore,
                 crate_num: ast::CrateNum,
                 callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::each_impl(cdata, callback)
}

pub fn each_implementation_for_type(cstore: @cstore::CStore,
                                    def_id: ast::DefId,
                                    callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(def_id.crate);
    decoder::each_implementation_for_type(cdata, def_id.node, callback)
}

pub fn each_implementation_for_trait(cstore: @cstore::CStore,
                                     def_id: ast::DefId,
                                     callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(def_id.crate);
    decoder::each_implementation_for_trait(cdata, def_id.node, callback)
}

/// If the given def ID describes a method belonging to a trait (either a
/// default method or an implementation of a trait method), returns the ID of
/// the trait that the method belongs to. Otherwise, returns `None`.
pub fn get_trait_of_method(cstore: @cstore::CStore,
                           def_id: ast::DefId,
                           tcx: ty::ctxt)
                           -> Option<ast::DefId> {
    let cdata = cstore.get_crate_data(def_id.crate);
    decoder::get_trait_of_method(cdata, def_id.node, tcx)
}

pub fn get_macro_registrar_fn(cstore: @cstore::CStore,
                              crate_num: ast::CrateNum)
                              -> Option<ast::DefId> {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::get_macro_registrar_fn(cdata)
}

pub fn get_exported_macros(cstore: @cstore::CStore,
                           crate_num: ast::CrateNum)
                           -> ~[~str] {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::get_exported_macros(cdata)
}
