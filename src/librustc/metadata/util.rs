// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use front::map as ast_map;
use metadata::cstore;
use metadata::decoder;
use middle::astencode;
use middle::lang_items;
use middle::ty;
use middle::def_id::{DefId, DefIndex};

use std::any::Any;
use std::rc::Rc;
use syntax::ast;
use syntax::attr;
use rustc_front::hir;

pub use metadata::csearch::FoundAst;
pub use metadata::cstore::LinkagePreference;
pub use metadata::decoder::DecodeInlinedItem;
pub use metadata::decoder::DefLike;
pub use metadata::inline::InlinedItem;

pub use self::DefLike::{DlDef, DlField, DlImpl};

pub struct ChildItem {
    pub def: DefLike,
    pub name: ast::Name,
    pub vis: hir::Visibility
}

pub trait CrateStore<'tcx> : Any {
    // item info
    fn stability(&self, def: DefId) -> Option<attr::Stability>;
    fn closure_kind(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                    -> ty::ClosureKind;
    fn closure_ty(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                  -> ty::ClosureTy<'tcx>;
    fn item_variances(&self, def: DefId) -> ty::ItemVariances;
    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr>;
    fn item_type(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx>;
    fn item_path(&self, def: DefId) -> Vec<ast_map::PathElem>;
    fn item_name(&self, def: DefId) -> ast::Name;
    fn item_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx>;
    fn item_super_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx>;
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>;
    fn trait_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId)-> ty::TraitDef<'tcx>;
    fn adt_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>;
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>;

    // trait info
    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId>;
    fn provided_trait_methods(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>>;
    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId>;

    // impl info
    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>;
    fn impl_trait_ref(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>>;
    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity>;
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>;
    fn associated_consts(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>>;

    // trait/impl-item info
    fn trait_of_item(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                     -> Option<DefId>;
    fn impl_or_trait_item(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> ty::ImplOrTraitItem<'tcx>;

    // flags
    fn is_const_fn(&self, did: DefId) -> bool;
    fn is_defaulted_trait(&self, did: DefId) -> bool;
    fn is_impl(&self, did: DefId) -> bool;
    fn is_static_method(&self, did: DefId) -> bool;

    // metadata
    fn dylib_dependency_formats(&self, cnum: ast::CrateNum)
                                    -> Vec<(ast::CrateNum, LinkagePreference)>;
    fn lang_items(&self, cnum: ast::CrateNum) -> Vec<(DefIndex, usize)>;
    fn missing_lang_items(&self, cnum: ast::CrateNum) -> Vec<lang_items::LangItem>;
    fn is_staged_api(&self, cnum: ast::CrateNum) -> bool;
    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>;

    // resolve
    fn def_path(&self, def: DefId) -> ast_map::DefPath;
    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>;
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>;
    fn item_children(&self, did: DefId) -> Vec<ChildItem>;
    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>;

    // misc.
    fn maybe_get_item_ast(&'tcx self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx>;
}

impl<'tcx> CrateStore<'tcx> for cstore::CStore {
    fn stability(&self, def: DefId) -> Option<attr::Stability>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_stability(&*cdata, def.index)
    }

    fn closure_kind(&self, _tcx: &ty::ctxt<'tcx>, def_id: DefId) -> ty::ClosureKind
    {
        assert!(!def_id.is_local());
        let cdata = self.get_crate_data(def_id.krate);
        decoder::closure_kind(&*cdata, def_id.index)
    }

    fn closure_ty(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId) -> ty::ClosureTy<'tcx>
    {
        assert!(!def_id.is_local());
        let cdata = self.get_crate_data(def_id.krate);
        decoder::closure_ty(&*cdata, def_id.index, tcx)
    }

    fn item_variances(&self, def: DefId) -> ty::ItemVariances {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_item_variances(&*cdata, def.index)
    }

    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr> {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_repr_attrs(&*cdata, def.index)
    }

    fn item_type(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_type(&*cdata, def.index, tcx)
    }

    fn item_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_predicates(&*cdata, def.index, tcx)
    }

    fn item_super_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_super_predicates(&*cdata, def.index, tcx)
    }

    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>
    {
        let cdata = self.get_crate_data(def_id.krate);
        decoder::get_item_attrs(&*cdata, def_id.index)
    }

    fn trait_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::TraitDef<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_trait_def(&*cdata, def.index, tcx)
    }

    fn adt_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_adt_def(&self.intr, &*cdata, def.index, tcx)
    }

    fn item_path(&self, def: DefId) -> Vec<ast_map::PathElem> {
        let cdata = self.get_crate_data(def.krate);
        let path = decoder::get_item_path(&*cdata, def.index);

        cdata.with_local_path(|cpath| {
            let mut r = Vec::with_capacity(cpath.len() + path.len());
            r.push_all(cpath);
            r.push_all(&path);
            r
        })
    }

    fn item_name(&self, def: DefId) -> ast::Name {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_item_name(&self.intr, &cdata, def.index)
    }


    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>
    {
        let mut result = vec![];
        let cdata = self.get_crate_data(def_id.krate);
        decoder::each_inherent_implementation_for_type(&*cdata, def_id.index,
                                                       |iid| result.push(iid));
        result
    }

    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId>
    {
        let mut result = vec![];
        self.iter_crate_data(|_, cdata| {
            decoder::each_implementation_for_trait(cdata, def_id, &mut |iid| {
                result.push(iid)
            })
        });
        result
    }

    fn provided_trait_methods(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_provided_trait_methods(self.intr.clone(), &*cdata, def.index, tcx)
    }

    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_trait_item_def_ids(&*cdata, def.index)
    }

    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>
    {
        let cdata = self.get_crate_data(impl_def_id.krate);
        decoder::get_impl_items(&*cdata, impl_def_id.index)
    }

    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_polarity(&*cdata, def.index)
    }

    fn impl_trait_ref(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_trait(&*cdata, def.index, tcx)
    }

    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_custom_coerce_unsized_kind(&*cdata, def.index)
    }

    // FIXME: killme
    fn associated_consts(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>> {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_associated_consts(self.intr.clone(), &*cdata, def.index, tcx)
    }

    fn trait_of_item(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId) -> Option<DefId>
    {
        let cdata = self.get_crate_data(def_id.krate);
        decoder::get_trait_of_item(&*cdata, def_id.index, tcx)
    }

    fn impl_or_trait_item(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> ty::ImplOrTraitItem<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_or_trait_item(
            self.intr.clone(),
            &*cdata,
            def.index,
            tcx)
    }

    fn is_const_fn(&self, did: DefId) -> bool
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_const_fn(&cdata, did.index)
    }

    fn is_defaulted_trait(&self, trait_def_id: DefId) -> bool {
        let cdata = self.get_crate_data(trait_def_id.krate);
        decoder::is_defaulted_trait(&*cdata, trait_def_id.index)
    }

    fn is_impl(&self, did: DefId) -> bool {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_impl(&*cdata, did.index)
    }

    fn is_static_method(&self, def: DefId) -> bool {
        let cdata = self.get_crate_data(def.krate);
        decoder::is_static_method(&*cdata, def.index)
    }

    fn dylib_dependency_formats(&self, cnum: ast::CrateNum)
                                -> Vec<(ast::CrateNum, LinkagePreference)>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_dylib_dependency_formats(&cdata)
    }

    fn lang_items(&self, cnum: ast::CrateNum) -> Vec<(DefIndex, usize)>
    {
        let mut result = vec![];
        let crate_data = self.get_crate_data(cnum);
        decoder::each_lang_item(&*crate_data, |did, lid| {
            result.push((did, lid)); true
        });
        result
    }

    fn missing_lang_items(&self, cnum: ast::CrateNum)
                          -> Vec<lang_items::LangItem>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_missing_lang_items(&*cdata)
    }

    fn is_staged_api(&self, cnum: ast::CrateNum) -> bool
    {
        self.get_crate_data(cnum).staged_api
    }

    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_plugin_registrar_fn(cdata.data()).map(|index| DefId {
            krate: cnum,
            index: index
        })
    }

    fn def_path(&self, def: DefId) -> ast_map::DefPath
    {
        let cdata = self.get_crate_data(def.krate);
        let path = decoder::def_path(&*cdata, def.index);
        let local_path = cdata.local_def_path();
        local_path.into_iter().chain(path).collect()
    }

    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::get_tuple_struct_definition_if_ctor(&*cdata, did.index)
    }

    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_struct_field_names(&self.intr, &*cdata, def.index)
    }

    fn item_children(&self, def_id: DefId) -> Vec<ChildItem>
    {
        let mut result = vec![];
        let crate_data = self.get_crate_data(def_id.krate);
        let get_crate_data = |cnum| self.get_crate_data(cnum);
        decoder::each_child_of_item(
            self.intr.clone(), &*crate_data,
            def_id.index, get_crate_data,
            |def, name, vis| result.push(ChildItem {
                def: def,
                name: name,
                vis: vis
            }));
        result
    }

    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>
    {
        let mut result = vec![];
        let crate_data = self.get_crate_data(cnum);
        let get_crate_data = |cnum| self.get_crate_data(cnum);
        decoder::each_top_level_item_of_crate(
            self.intr.clone(), &*crate_data, get_crate_data,
            |def, name, vis| result.push(ChildItem {
                def: def,
                name: name,
                vis: vis
            }));
        result
    }

    fn maybe_get_item_ast(&'tcx self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        let decode_inlined_item = Box::new(astencode::decode_inlined_item);
        decoder::maybe_get_item_ast(&*cdata, tcx, def.index, decode_inlined_item)
    }
}
