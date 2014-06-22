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

#![allow(non_camel_case_types)]

use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use middle::lang_items;
use middle::ty;
use middle::typeck;
use middle::subst::VecPerParamSpace;

use serialize::ebml;
use serialize::ebml::reader;
use std::rc::Rc;
use syntax::ast;
use syntax::ast_map;
use syntax::attr;
use syntax::diagnostic::expect;
use syntax::parse::token;

pub struct StaticMethodInfo {
    pub ident: ast::Ident,
    pub def_id: ast::DefId,
    pub fn_style: ast::FnStyle,
    pub vis: ast::Visibility,
}

pub fn get_symbol(cstore: &cstore::CStore, def: ast::DefId) -> String {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_symbol(cdata.data(), def.node)
}

/// Iterates over all the language items in the given crate.
pub fn each_lang_item(cstore: &cstore::CStore,
                      cnum: ast::CrateNum,
                      f: |ast::NodeId, uint| -> bool)
                      -> bool {
    let crate_data = cstore.get_crate_data(cnum);
    decoder::each_lang_item(&*crate_data, f)
}

/// Iterates over each child of the given item.
pub fn each_child_of_item(cstore: &cstore::CStore,
                          def_id: ast::DefId,
                          callback: |decoder::DefLike,
                                     ast::Ident,
                                     ast::Visibility|) {
    let crate_data = cstore.get_crate_data(def_id.krate);
    let get_crate_data: decoder::GetCrateDataCb = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_child_of_item(cstore.intr.clone(),
                                &*crate_data,
                                def_id.node,
                                get_crate_data,
                                callback)
}

/// Iterates over each top-level crate item.
pub fn each_top_level_item_of_crate(cstore: &cstore::CStore,
                                    cnum: ast::CrateNum,
                                    callback: |decoder::DefLike,
                                               ast::Ident,
                                               ast::Visibility|) {
    let crate_data = cstore.get_crate_data(cnum);
    let get_crate_data: decoder::GetCrateDataCb = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_top_level_item_of_crate(cstore.intr.clone(),
                                          &*crate_data,
                                          get_crate_data,
                                          callback)
}

pub fn get_item_path(tcx: &ty::ctxt, def: ast::DefId) -> Vec<ast_map::PathElem> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    let path = decoder::get_item_path(&*cdata, def.node);

    // FIXME #1920: This path is not always correct if the crate is not linked
    // into the root namespace.
    (vec!(ast_map::PathMod(token::intern(cdata.name.as_slice())))).append(
        path.as_slice())
}

pub enum found_ast {
    found(ast::InlinedItem),
    found_parent(ast::DefId, ast::InlinedItem),
    not_found,
}

// Finds the AST for this item in the crate metadata, if any.  If the item was
// not marked for inlining, then the AST will not be present and hence none
// will be returned.
pub fn maybe_get_item_ast(tcx: &ty::ctxt, def: ast::DefId,
                          decode_inlined_item: decoder::DecodeInlinedItem)
                       -> found_ast {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::maybe_get_item_ast(&*cdata, tcx, def.node, decode_inlined_item)
}

pub fn get_enum_variants(tcx: &ty::ctxt, def: ast::DefId)
                      -> Vec<Rc<ty::VariantInfo>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_enum_variants(cstore.intr.clone(), &*cdata, def.node, tcx)
}

/// Returns information about the given implementation.
pub fn get_impl_methods(cstore: &cstore::CStore, impl_def_id: ast::DefId)
                        -> Vec<ast::DefId> {
    let cdata = cstore.get_crate_data(impl_def_id.krate);
    decoder::get_impl_methods(&*cdata, impl_def_id.node)
}

pub fn get_method(tcx: &ty::ctxt, def: ast::DefId) -> ty::Method {
    let cdata = tcx.sess.cstore.get_crate_data(def.krate);
    decoder::get_method(tcx.sess.cstore.intr.clone(), &*cdata, def.node, tcx)
}

pub fn get_method_name_and_explicit_self(cstore: &cstore::CStore,
                                         def: ast::DefId)
                                     -> (ast::Ident, ast::ExplicitSelf_)
{
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_method_name_and_explicit_self(cstore.intr.clone(), &*cdata, def.node)
}

pub fn get_trait_method_def_ids(cstore: &cstore::CStore,
                                def: ast::DefId) -> Vec<ast::DefId> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_trait_method_def_ids(&*cdata, def.node)
}

pub fn get_item_variances(cstore: &cstore::CStore,
                          def: ast::DefId) -> ty::ItemVariances {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_item_variances(&*cdata, def.node)
}

pub fn get_provided_trait_methods(tcx: &ty::ctxt,
                                  def: ast::DefId)
                               -> Vec<Rc<ty::Method>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_provided_trait_methods(cstore.intr.clone(), &*cdata, def.node, tcx)
}

pub fn get_supertraits(tcx: &ty::ctxt, def: ast::DefId) -> Vec<Rc<ty::TraitRef>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_supertraits(&*cdata, def.node, tcx)
}

pub fn get_type_name_if_impl(cstore: &cstore::CStore, def: ast::DefId)
                          -> Option<ast::Ident> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_type_name_if_impl(&*cdata, def.node)
}

pub fn get_static_methods_if_impl(cstore: &cstore::CStore,
                                  def: ast::DefId)
                               -> Option<Vec<StaticMethodInfo> > {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_static_methods_if_impl(cstore.intr.clone(), &*cdata, def.node)
}

pub fn get_item_attrs(cstore: &cstore::CStore,
                      def_id: ast::DefId,
                      f: |Vec<ast::Attribute>|) {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_item_attrs(&*cdata, def_id.node, f)
}

pub fn get_struct_fields(cstore: &cstore::CStore,
                         def: ast::DefId)
                      -> Vec<ty::field_ty> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_struct_fields(cstore.intr.clone(), &*cdata, def.node)
}

pub fn get_type(tcx: &ty::ctxt,
                def: ast::DefId)
             -> ty::Polytype {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_type(&*cdata, def.node, tcx)
}

pub fn get_trait_def(tcx: &ty::ctxt, def: ast::DefId) -> ty::TraitDef {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_trait_def(&*cdata, def.node, tcx)
}

pub fn get_field_type(tcx: &ty::ctxt, class_id: ast::DefId,
                      def: ast::DefId) -> ty::Polytype {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(class_id.krate);
    let all_items = reader::get_doc(ebml::Doc::new(cdata.data()), tag_items);
    let class_doc = expect(tcx.sess.diagnostic(),
                           decoder::maybe_find_item(class_id.node, all_items),
                           || {
        (format!("get_field_type: class ID {:?} not found",
                 class_id)).to_string()
    });
    let the_field = expect(tcx.sess.diagnostic(),
        decoder::maybe_find_item(def.node, class_doc),
        || {
            (format!("get_field_type: in class {:?}, field ID {:?} not found",
                    class_id,
                    def)).to_string()
        });
    let ty = decoder::item_type(def, the_field, tcx, &*cdata);
    ty::Polytype {
        generics: ty::Generics {types: VecPerParamSpace::empty(),
                                regions: VecPerParamSpace::empty()},
        ty: ty
    }
}

// Given a def_id for an impl, return the trait it implements,
// if there is one.
pub fn get_impl_trait(tcx: &ty::ctxt,
                      def: ast::DefId) -> Option<Rc<ty::TraitRef>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_impl_trait(&*cdata, def.node, tcx)
}

// Given a def_id for an impl, return information about its vtables
pub fn get_impl_vtables(tcx: &ty::ctxt,
                        def: ast::DefId)
                        -> typeck::vtable_res {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_impl_vtables(&*cdata, def.node, tcx)
}

pub fn get_native_libraries(cstore: &cstore::CStore,
                            crate_num: ast::CrateNum)
                                -> Vec<(cstore::NativeLibaryKind, String)> {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::get_native_libraries(&*cdata)
}

pub fn each_impl(cstore: &cstore::CStore,
                 crate_num: ast::CrateNum,
                 callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::each_impl(&*cdata, callback)
}

pub fn each_implementation_for_type(cstore: &cstore::CStore,
                                    def_id: ast::DefId,
                                    callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::each_implementation_for_type(&*cdata, def_id.node, callback)
}

pub fn each_implementation_for_trait(cstore: &cstore::CStore,
                                     def_id: ast::DefId,
                                     callback: |ast::DefId|) {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::each_implementation_for_trait(&*cdata, def_id.node, callback)
}

/// If the given def ID describes a method belonging to a trait (either a
/// default method or an implementation of a trait method), returns the ID of
/// the trait that the method belongs to. Otherwise, returns `None`.
pub fn get_trait_of_method(cstore: &cstore::CStore,
                           def_id: ast::DefId,
                           tcx: &ty::ctxt)
                           -> Option<ast::DefId> {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_trait_of_method(&*cdata, def_id.node, tcx)
}

pub fn get_tuple_struct_definition_if_ctor(cstore: &cstore::CStore,
                                           def_id: ast::DefId)
    -> Option<ast::DefId>
{
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_tuple_struct_definition_if_ctor(&*cdata, def_id.node)
}

pub fn get_dylib_dependency_formats(cstore: &cstore::CStore,
                                    cnum: ast::CrateNum)
    -> Vec<(ast::CrateNum, cstore::LinkagePreference)>
{
    let cdata = cstore.get_crate_data(cnum);
    decoder::get_dylib_dependency_formats(&*cdata)
}

pub fn get_missing_lang_items(cstore: &cstore::CStore, cnum: ast::CrateNum)
    -> Vec<lang_items::LangItem>
{
    let cdata = cstore.get_crate_data(cnum);
    decoder::get_missing_lang_items(&*cdata)
}

pub fn get_method_arg_names(cstore: &cstore::CStore, did: ast::DefId)
    -> Vec<String>
{
    let cdata = cstore.get_crate_data(did.krate);
    decoder::get_method_arg_names(&*cdata, did.node)
}

pub fn get_reachable_extern_fns(cstore: &cstore::CStore, cnum: ast::CrateNum)
    -> Vec<ast::DefId>
{
    let cdata = cstore.get_crate_data(cnum);
    decoder::get_reachable_extern_fns(&*cdata)
}

pub fn is_typedef(cstore: &cstore::CStore, did: ast::DefId) -> bool {
    let cdata = cstore.get_crate_data(did.krate);
    decoder::is_typedef(&*cdata, did.node)
}

pub fn get_stability(cstore: &cstore::CStore,
                     def: ast::DefId)
                     -> Option<attr::Stability> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_stability(&*cdata, def.node)
}
