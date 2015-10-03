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

use front::map as ast_map;
use metadata::cstore;
use metadata::decoder;
use metadata::inline::InlinedItem;
use middle::def_id::{DefId, DefIndex};
use middle::lang_items;
use middle::ty;
use util::nodemap::FnvHashMap;

use std::rc::Rc;
use syntax::ast;
use syntax::attr;
use rustc_front::hir;

#[derive(Copy, Clone)]
pub struct MethodInfo {
    pub name: ast::Name,
    pub def_id: DefId,
    pub vis: hir::Visibility,
}

pub fn get_symbol(cstore: &cstore::CStore, def: DefId) -> String {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_symbol(&cdata, def.index)
}

/// Iterates over all the language items in the given crate.
pub fn each_lang_item<F>(cstore: &cstore::CStore,
                         cnum: ast::CrateNum,
                         f: F)
                         -> bool where
    F: FnMut(DefIndex, usize) -> bool,
{
    let crate_data = cstore.get_crate_data(cnum);
    decoder::each_lang_item(&*crate_data, f)
}

/// Iterates over each child of the given item.
pub fn each_child_of_item<F>(cstore: &cstore::CStore,
                             def_id: DefId,
                             callback: F) where
    F: FnMut(decoder::DefLike, ast::Name, hir::Visibility),
{
    let crate_data = cstore.get_crate_data(def_id.krate);
    let get_crate_data = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_child_of_item(cstore.intr.clone(),
                                &*crate_data,
                                def_id.index,
                                get_crate_data,
                                callback)
}

/// Iterates over each top-level crate item.
pub fn each_top_level_item_of_crate<F>(cstore: &cstore::CStore,
                                       cnum: ast::CrateNum,
                                       callback: F) where
    F: FnMut(decoder::DefLike, ast::Name, hir::Visibility),
{
    let crate_data = cstore.get_crate_data(cnum);
    let get_crate_data = |cnum| {
        cstore.get_crate_data(cnum)
    };
    decoder::each_top_level_item_of_crate(cstore.intr.clone(),
                                          &*crate_data,
                                          get_crate_data,
                                          callback)
}

pub fn get_item_path(tcx: &ty::ctxt, def: DefId) -> Vec<ast_map::PathElem> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    let path = decoder::get_item_path(&*cdata, def.index);

    cdata.with_local_path(|cpath| {
        let mut r = Vec::with_capacity(cpath.len() + path.len());
        r.push_all(cpath);
        r.push_all(&path);
        r
    })
}

pub fn get_item_name(tcx: &ty::ctxt, def: DefId) -> ast::Name {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_item_name(&cstore.intr, &cdata, def.index)
}

pub enum FoundAst<'ast> {
    Found(&'ast InlinedItem),
    FoundParent(DefId, &'ast InlinedItem),
    NotFound,
}

// Finds the AST for this item in the crate metadata, if any.  If the item was
// not marked for inlining, then the AST will not be present and hence none
// will be returned.
pub fn maybe_get_item_ast<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId,
                                decode_inlined_item: decoder::DecodeInlinedItem)
                                -> FoundAst<'tcx> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::maybe_get_item_ast(&*cdata, tcx, def.index, decode_inlined_item)
}

/// Returns information about the given implementation.
pub fn get_impl_items(cstore: &cstore::CStore, impl_def_id: DefId)
                      -> Vec<ty::ImplOrTraitItemId> {
    let cdata = cstore.get_crate_data(impl_def_id.krate);
    decoder::get_impl_items(&*cdata, impl_def_id.index)
}

pub fn get_impl_or_trait_item<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId)
                                    -> ty::ImplOrTraitItem<'tcx> {
    let cdata = tcx.sess.cstore.get_crate_data(def.krate);
    decoder::get_impl_or_trait_item(tcx.sess.cstore.intr.clone(),
                                    &*cdata,
                                    def.index,
                                    tcx)
}

pub fn get_trait_name(cstore: &cstore::CStore, def: DefId) -> ast::Name {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_trait_name(cstore.intr.clone(),
                            &*cdata,
                            def.index)
}

pub fn is_static_method(cstore: &cstore::CStore, def: DefId) -> bool {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::is_static_method(&*cdata, def.index)
}

pub fn get_trait_item_def_ids(cstore: &cstore::CStore, def: DefId)
                              -> Vec<ty::ImplOrTraitItemId> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_trait_item_def_ids(&*cdata, def.index)
}

pub fn get_item_variances(cstore: &cstore::CStore,
                          def: DefId) -> ty::ItemVariances {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_item_variances(&*cdata, def.index)
}

pub fn get_provided_trait_methods<'tcx>(tcx: &ty::ctxt<'tcx>,
                                        def: DefId)
                                        -> Vec<Rc<ty::Method<'tcx>>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_provided_trait_methods(cstore.intr.clone(), &*cdata, def.index, tcx)
}

pub fn get_associated_consts<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId)
                                   -> Vec<Rc<ty::AssociatedConst<'tcx>>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_associated_consts(cstore.intr.clone(), &*cdata, def.index, tcx)
}

pub fn get_methods_if_impl(cstore: &cstore::CStore,
                                  def: DefId)
                               -> Option<Vec<MethodInfo> > {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_methods_if_impl(cstore.intr.clone(), &*cdata, def.index)
}

pub fn get_item_attrs(cstore: &cstore::CStore,
                      def_id: DefId)
                      -> Vec<ast::Attribute> {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_item_attrs(&*cdata, def_id.index)
}

pub fn get_struct_field_names(cstore: &cstore::CStore, def: DefId) -> Vec<ast::Name> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_struct_field_names(&cstore.intr, &*cdata, def.index)
}

pub fn get_struct_field_attrs(cstore: &cstore::CStore, def: DefId)
                              -> FnvHashMap<DefId, Vec<ast::Attribute>> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_struct_field_attrs(&*cdata)
}

pub fn get_type<'tcx>(tcx: &ty::ctxt<'tcx>,
                      def: DefId)
                      -> ty::TypeScheme<'tcx> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_type(&*cdata, def.index, tcx)
}

pub fn get_trait_def<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::TraitDef<'tcx> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_trait_def(&*cdata, def.index, tcx)
}

pub fn get_adt_def<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_adt_def(&cstore.intr, &*cdata, def.index, tcx)
}

pub fn get_predicates<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId)
                            -> ty::GenericPredicates<'tcx>
{
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_predicates(&*cdata, def.index, tcx)
}

pub fn get_super_predicates<'tcx>(tcx: &ty::ctxt<'tcx>, def: DefId)
                                  -> ty::GenericPredicates<'tcx>
{
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_super_predicates(&*cdata, def.index, tcx)
}

pub fn get_impl_polarity<'tcx>(tcx: &ty::ctxt<'tcx>,
                               def: DefId)
                               -> Option<hir::ImplPolarity>
{
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_impl_polarity(&*cdata, def.index)
}

pub fn get_custom_coerce_unsized_kind<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    def: DefId)
    -> Option<ty::adjustment::CustomCoerceUnsized>
{
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_custom_coerce_unsized_kind(&*cdata, def.index)
}

// Given a def_id for an impl, return the trait it implements,
// if there is one.
pub fn get_impl_trait<'tcx>(tcx: &ty::ctxt<'tcx>,
                            def: DefId)
                            -> Option<ty::TraitRef<'tcx>> {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_impl_trait(&*cdata, def.index, tcx)
}

pub fn get_native_libraries(cstore: &cstore::CStore, crate_num: ast::CrateNum)
                            -> Vec<(cstore::NativeLibraryKind, String)> {
    let cdata = cstore.get_crate_data(crate_num);
    decoder::get_native_libraries(&*cdata)
}

pub fn each_inherent_implementation_for_type<F>(cstore: &cstore::CStore,
                                                def_id: DefId,
                                                callback: F) where
    F: FnMut(DefId),
{
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::each_inherent_implementation_for_type(&*cdata, def_id.index, callback)
}

pub fn each_implementation_for_trait<F>(cstore: &cstore::CStore,
                                        def_id: DefId,
                                        mut callback: F) where
    F: FnMut(DefId),
{
    cstore.iter_crate_data(|_, cdata| {
        decoder::each_implementation_for_trait(cdata, def_id, &mut callback)
    })
}

/// If the given def ID describes an item belonging to a trait (either a
/// default method or an implementation of a trait method), returns the ID of
/// the trait that the method belongs to. Otherwise, returns `None`.
pub fn get_trait_of_item(cstore: &cstore::CStore,
                         def_id: DefId,
                         tcx: &ty::ctxt)
                         -> Option<DefId> {
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_trait_of_item(&*cdata, def_id.index, tcx)
}

pub fn get_tuple_struct_definition_if_ctor(cstore: &cstore::CStore,
                                           def_id: DefId)
    -> Option<DefId>
{
    let cdata = cstore.get_crate_data(def_id.krate);
    decoder::get_tuple_struct_definition_if_ctor(&*cdata, def_id.index)
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

pub fn get_method_arg_names(cstore: &cstore::CStore, did: DefId)
    -> Vec<String>
{
    let cdata = cstore.get_crate_data(did.krate);
    decoder::get_method_arg_names(&*cdata, did.index)
}

pub fn get_reachable_ids(cstore: &cstore::CStore, cnum: ast::CrateNum)
    -> Vec<DefId>
{
    let cdata = cstore.get_crate_data(cnum);
    decoder::get_reachable_ids(&*cdata)
}

pub fn is_typedef(cstore: &cstore::CStore, did: DefId) -> bool {
    let cdata = cstore.get_crate_data(did.krate);
    decoder::is_typedef(&*cdata, did.index)
}

pub fn is_const_fn(cstore: &cstore::CStore, did: DefId) -> bool {
    let cdata = cstore.get_crate_data(did.krate);
    decoder::is_const_fn(&*cdata, did.index)
}

pub fn is_impl(cstore: &cstore::CStore, did: DefId) -> bool {
    let cdata = cstore.get_crate_data(did.krate);
    decoder::is_impl(&*cdata, did.index)
}

pub fn get_stability(cstore: &cstore::CStore,
                     def: DefId)
                     -> Option<attr::Stability> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_stability(&*cdata, def.index)
}

pub fn is_staged_api(cstore: &cstore::CStore, krate: ast::CrateNum) -> bool {
    cstore.get_crate_data(krate).staged_api
}

pub fn get_repr_attrs(cstore: &cstore::CStore, def: DefId)
                      -> Vec<attr::ReprAttr> {
    let cdata = cstore.get_crate_data(def.krate);
    decoder::get_repr_attrs(&*cdata, def.index)
}

pub fn is_defaulted_trait(cstore: &cstore::CStore, trait_def_id: DefId) -> bool {
    let cdata = cstore.get_crate_data(trait_def_id.krate);
    decoder::is_defaulted_trait(&*cdata, trait_def_id.index)
}

pub fn is_default_impl(cstore: &cstore::CStore, impl_did: DefId) -> bool {
    let cdata = cstore.get_crate_data(impl_did.krate);
    decoder::is_default_impl(&*cdata, impl_did.index)
}

pub fn is_extern_fn(cstore: &cstore::CStore, did: DefId,
                    tcx: &ty::ctxt) -> bool {
    let cdata = cstore.get_crate_data(did.krate);
    decoder::is_extern_fn(&*cdata, did.index, tcx)
}

pub fn closure_kind<'tcx>(tcx: &ty::ctxt<'tcx>, def_id: DefId) -> ty::ClosureKind {
    assert!(!def_id.is_local());
    let cdata = tcx.sess.cstore.get_crate_data(def_id.krate);
    decoder::closure_kind(&*cdata, def_id.index)
}

pub fn closure_ty<'tcx>(tcx: &ty::ctxt<'tcx>, def_id: DefId) -> ty::ClosureTy<'tcx> {
    assert!(!def_id.is_local());
    let cdata = tcx.sess.cstore.get_crate_data(def_id.krate);
    decoder::closure_ty(&*cdata, def_id.index, tcx)
}

pub fn def_path(tcx: &ty::ctxt, def: DefId) -> ast_map::DefPath {
    let cstore = &tcx.sess.cstore;
    let cdata = cstore.get_crate_data(def.krate);
    let path = decoder::def_path(&*cdata, def.index);
    let local_path = cdata.local_def_path();
    local_path.into_iter().chain(path).collect()
}

