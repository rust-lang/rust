// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cstore;
use decoder;
use encoder;
use loader;

use middle::cstore::{CrateStore, CrateSource, ChildItem, ExternCrate, FoundAst, DefLike};
use middle::cstore::{NativeLibraryKind, LinkMeta, LinkagePreference};
use rustc::hir::def;
use middle::lang_items;
use rustc::ty::{self, Ty, TyCtxt, VariantKind};
use rustc::hir::def_id::{DefId, DefIndex, CRATE_DEF_INDEX};

use rustc::hir::map as hir_map;
use rustc::mir::repr::Mir;
use rustc::mir::mir_map::MirMap;
use rustc::util::nodemap::{FnvHashMap, NodeSet, DefIdMap};
use rustc::session::config::PanicStrategy;

use std::cell::RefCell;
use std::rc::Rc;
use std::path::PathBuf;
use syntax::ast;
use syntax::attr;
use syntax::parse::token;
use rustc::hir::svh::Svh;
use rustc_back::target::Target;
use rustc::hir;

impl<'tcx> CrateStore<'tcx> for cstore::CStore {
    fn stability(&self, def: DefId) -> Option<attr::Stability>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_stability(&cdata, def.index)
    }

    fn deprecation(&self, def: DefId) -> Option<attr::Deprecation>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_deprecation(&cdata, def.index)
    }

    fn visibility(&self, def: DefId) -> ty::Visibility {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_visibility(&cdata, def.index)
    }

    fn closure_kind(&self, def_id: DefId) -> ty::ClosureKind
    {
        assert!(!def_id.is_local());
        let cdata = self.get_crate_data(def_id.krate);
        decoder::closure_kind(&cdata, def_id.index)
    }

    fn closure_ty<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> ty::ClosureTy<'tcx>
    {
        assert!(!def_id.is_local());
        let cdata = self.get_crate_data(def_id.krate);
        decoder::closure_ty(&cdata, def_id.index, tcx)
    }

    fn item_variances(&self, def: DefId) -> ty::ItemVariances {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_item_variances(&cdata, def.index)
    }

    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr> {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_repr_attrs(&cdata, def.index)
    }

    fn item_type<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                     -> ty::TypeScheme<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_type(&cdata, def.index, tcx)
    }

    fn item_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                           -> ty::GenericPredicates<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_predicates(&cdata, def.index, tcx)
    }

    fn item_super_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                                 -> ty::GenericPredicates<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_super_predicates(&cdata, def.index, tcx)
    }

    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>
    {
        let cdata = self.get_crate_data(def_id.krate);
        decoder::get_item_attrs(&cdata, def_id.index)
    }

    fn trait_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId) -> ty::TraitDef<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_trait_def(&cdata, def.index, tcx)
    }

    fn adt_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_adt_def(&self.intr, &cdata, def.index, tcx)
    }

    fn method_arg_names(&self, did: DefId) -> Vec<String>
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::get_method_arg_names(&cdata, did.index)
    }

    fn item_name(&self, def: DefId) -> ast::Name {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_item_name(&self.intr, &cdata, def.index)
    }


    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>
    {
        let mut result = vec![];
        let cdata = self.get_crate_data(def_id.krate);
        decoder::each_inherent_implementation_for_type(&cdata, def_id.index,
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

    fn provided_trait_methods<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                                  -> Vec<Rc<ty::Method<'tcx>>>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_provided_trait_methods(self.intr.clone(), &cdata, def.index, tcx)
    }

    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_trait_item_def_ids(&cdata, def.index)
    }

    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>
    {
        let cdata = self.get_crate_data(impl_def_id.krate);
        decoder::get_impl_items(&cdata, impl_def_id.index)
    }

    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_polarity(&cdata, def.index)
    }

    fn impl_trait_ref<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                          -> Option<ty::TraitRef<'tcx>>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_trait(&cdata, def.index, tcx)
    }

    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_custom_coerce_unsized_kind(&cdata, def.index)
    }

    // FIXME: killme
    fn associated_consts<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                             -> Vec<Rc<ty::AssociatedConst<'tcx>>> {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_associated_consts(self.intr.clone(), &cdata, def.index, tcx)
    }

    fn impl_parent(&self, impl_def: DefId) -> Option<DefId> {
        let cdata = self.get_crate_data(impl_def.krate);
        decoder::get_parent_impl(&*cdata, impl_def.index)
    }

    fn trait_of_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> Option<DefId>
    {
        let cdata = self.get_crate_data(def_id.krate);
        decoder::get_trait_of_item(&cdata, def_id.index, tcx)
    }

    fn impl_or_trait_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<ty::ImplOrTraitItem<'tcx>>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_impl_or_trait_item(
            self.intr.clone(),
            &cdata,
            def.index,
            tcx)
    }

    fn is_const_fn(&self, did: DefId) -> bool
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_const_fn(&cdata, did.index)
    }

    fn is_defaulted_trait(&self, trait_def_id: DefId) -> bool
    {
        let cdata = self.get_crate_data(trait_def_id.krate);
        decoder::is_defaulted_trait(&cdata, trait_def_id.index)
    }

    fn is_impl(&self, did: DefId) -> bool
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_impl(&cdata, did.index)
    }

    fn is_default_impl(&self, impl_did: DefId) -> bool {
        let cdata = self.get_crate_data(impl_did.krate);
        decoder::is_default_impl(&cdata, impl_did.index)
    }

    fn is_extern_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, did: DefId) -> bool {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_extern_item(&cdata, did.index, tcx)
    }

    fn is_foreign_item(&self, did: DefId) -> bool {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_foreign_item(&cdata, did.index)
    }

    fn is_static_method(&self, def: DefId) -> bool
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::is_static_method(&cdata, def.index)
    }

    fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool
    {
        self.do_is_statically_included_foreign_item(id)
    }

    fn is_typedef(&self, did: DefId) -> bool {
        let cdata = self.get_crate_data(did.krate);
        decoder::is_typedef(&cdata, did.index)
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
        decoder::each_lang_item(&crate_data, |did, lid| {
            result.push((did, lid)); true
        });
        result
    }

    fn missing_lang_items(&self, cnum: ast::CrateNum)
                          -> Vec<lang_items::LangItem>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_missing_lang_items(&cdata)
    }

    fn is_staged_api(&self, cnum: ast::CrateNum) -> bool
    {
        self.get_crate_data(cnum).staged_api
    }

    fn is_explicitly_linked(&self, cnum: ast::CrateNum) -> bool
    {
        self.get_crate_data(cnum).explicitly_linked.get()
    }

    fn is_allocator(&self, cnum: ast::CrateNum) -> bool
    {
        self.get_crate_data(cnum).is_allocator()
    }

    fn is_panic_runtime(&self, cnum: ast::CrateNum) -> bool
    {
        self.get_crate_data(cnum).is_panic_runtime()
    }

    fn panic_strategy(&self, cnum: ast::CrateNum) -> PanicStrategy {
        self.get_crate_data(cnum).panic_strategy()
    }

    fn crate_attrs(&self, cnum: ast::CrateNum) -> Vec<ast::Attribute>
    {
        decoder::get_crate_attributes(self.get_crate_data(cnum).data())
    }

    fn crate_name(&self, cnum: ast::CrateNum) -> token::InternedString
    {
        token::intern_and_get_ident(&self.get_crate_data(cnum).name[..])
    }

    fn original_crate_name(&self, cnum: ast::CrateNum) -> token::InternedString
    {
        token::intern_and_get_ident(&self.get_crate_data(cnum).name())
    }

    fn extern_crate(&self, cnum: ast::CrateNum) -> Option<ExternCrate>
    {
        self.get_crate_data(cnum).extern_crate.get()
    }

    fn crate_hash(&self, cnum: ast::CrateNum) -> Svh
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_crate_hash(cdata.data())
    }

    fn crate_disambiguator(&self, cnum: ast::CrateNum) -> token::InternedString
    {
        let cdata = self.get_crate_data(cnum);
        token::intern_and_get_ident(decoder::get_crate_disambiguator(cdata.data()))
    }

    fn crate_struct_field_attrs(&self, cnum: ast::CrateNum)
                                -> FnvHashMap<DefId, Vec<ast::Attribute>>
    {
        decoder::get_struct_field_attrs(&self.get_crate_data(cnum))
    }

    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_plugin_registrar_fn(cdata.data()).map(|index| DefId {
            krate: cnum,
            index: index
        })
    }

    fn native_libraries(&self, cnum: ast::CrateNum) -> Vec<(NativeLibraryKind, String)>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_native_libraries(&cdata)
    }

    fn reachable_ids(&self, cnum: ast::CrateNum) -> Vec<DefId>
    {
        let cdata = self.get_crate_data(cnum);
        decoder::get_reachable_ids(&cdata)
    }

    /// Returns the `DefKey` for a given `DefId`. This indicates the
    /// parent `DefId` as well as some idea of what kind of data the
    /// `DefId` refers to.
    fn def_key(&self, def: DefId) -> hir_map::DefKey {
        let cdata = self.get_crate_data(def.krate);
        decoder::def_key(&cdata, def.index)
    }

    fn relative_def_path(&self, def: DefId) -> hir_map::DefPath {
        let cdata = self.get_crate_data(def.krate);
        decoder::def_path(&cdata, def.index)
    }

    fn variant_kind(&self, def_id: DefId) -> Option<VariantKind> {
        let cdata = self.get_crate_data(def_id.krate);
        decoder::get_variant_kind(&cdata, def_id.index)
    }

    fn struct_ctor_def_id(&self, struct_def_id: DefId) -> Option<DefId>
    {
        let cdata = self.get_crate_data(struct_def_id.krate);
        decoder::get_struct_ctor_def_id(&cdata, struct_def_id.index)
    }

    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>
    {
        let cdata = self.get_crate_data(did.krate);
        decoder::get_tuple_struct_definition_if_ctor(&cdata, did.index)
    }

    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::get_struct_field_names(&self.intr, &cdata, def.index)
    }

    fn item_children(&self, def_id: DefId) -> Vec<ChildItem>
    {
        let mut result = vec![];
        let crate_data = self.get_crate_data(def_id.krate);
        let get_crate_data = |cnum| self.get_crate_data(cnum);
        decoder::each_child_of_item(
            self.intr.clone(), &crate_data,
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
            self.intr.clone(), &crate_data, get_crate_data,
            |def, name, vis| result.push(ChildItem {
                def: def,
                name: name,
                vis: vis
            }));
        result
    }

    fn maybe_get_item_ast<'a>(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> FoundAst<'tcx>
    {
        let cdata = self.get_crate_data(def.krate);
        decoder::maybe_get_item_ast(&cdata, tcx, def.index)
    }

    fn maybe_get_item_mir<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<Mir<'tcx>> {
        let cdata = self.get_crate_data(def.krate);
        decoder::maybe_get_item_mir(&cdata, tcx, def.index)
    }

    fn is_item_mir_available(&self, def: DefId) -> bool {
        let cdata = self.get_crate_data(def.krate);
        decoder::is_item_mir_available(&cdata, def.index)
    }

    fn crates(&self) -> Vec<ast::CrateNum>
    {
        let mut result = vec![];
        self.iter_crate_data(|cnum, _| result.push(cnum));
        result
    }

    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)>
    {
        self.get_used_libraries().borrow().clone()
    }

    fn used_link_args(&self) -> Vec<String>
    {
        self.get_used_link_args().borrow().clone()
    }

    fn metadata_filename(&self) -> &str
    {
        loader::METADATA_FILENAME
    }

    fn metadata_section_name(&self, target: &Target) -> &str
    {
        loader::meta_section_name(target)
    }
    fn encode_type<'a>(&self,
                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       ty: Ty<'tcx>,
                       def_id_to_string: for<'b> fn(TyCtxt<'b, 'tcx, 'tcx>, DefId) -> String)
                       -> Vec<u8>
    {
        encoder::encoded_ty(tcx, ty, def_id_to_string)
    }

    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(ast::CrateNum, Option<PathBuf>)>
    {
        self.do_get_used_crates(prefer)
    }

    fn used_crate_source(&self, cnum: ast::CrateNum) -> CrateSource
    {
        self.opt_used_crate_source(cnum).unwrap()
    }

    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<ast::CrateNum>
    {
        self.do_extern_mod_stmt_cnum(emod_id)
    }

    fn encode_metadata<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           reexports: &def::ExportMap,
                           link_meta: &LinkMeta,
                           reachable: &NodeSet,
                           mir_map: &MirMap<'tcx>,
                           krate: &hir::Crate) -> Vec<u8>
    {
        let ecx = encoder::EncodeContext {
            diag: tcx.sess.diagnostic(),
            tcx: tcx,
            reexports: reexports,
            link_meta: link_meta,
            cstore: self,
            reachable: reachable,
            mir_map: mir_map,
            type_abbrevs: RefCell::new(FnvHashMap()),
        };
        encoder::encode_metadata(ecx, krate)

    }

    fn metadata_encoding_version(&self) -> &[u8]
    {
        encoder::metadata_encoding_version
    }

    /// Returns a map from a sufficiently visible external item (i.e. an external item that is
    /// visible from at least one local module) to a sufficiently visible parent (considering
    /// modules that re-export the external item to be parents).
    fn visible_parent_map<'a>(&'a self) -> ::std::cell::RefMut<'a, DefIdMap<DefId>> {
        let mut visible_parent_map = self.visible_parent_map.borrow_mut();
        if !visible_parent_map.is_empty() { return visible_parent_map; }

        use rustc::middle::cstore::ChildItem;
        use std::collections::vec_deque::VecDeque;
        use std::collections::hash_map::Entry;
        for cnum in 1 .. self.next_crate_num() {
            let cdata = self.get_crate_data(cnum);

            match cdata.extern_crate.get() {
                // Ignore crates without a corresponding local `extern crate` item.
                Some(extern_crate) if !extern_crate.direct => continue,
                _ => {},
            }

            let mut bfs_queue = &mut VecDeque::new();
            let mut add_child = |bfs_queue: &mut VecDeque<_>, child: ChildItem, parent: DefId| {
                let child = match child.def {
                    DefLike::DlDef(def) if child.vis == ty::Visibility::Public => def.def_id(),
                    _ => return,
                };

                match visible_parent_map.entry(child) {
                    Entry::Occupied(mut entry) => {
                        // If `child` is defined in crate `cnum`, ensure
                        // that it is mapped to a parent in `cnum`.
                        if child.krate == cnum && entry.get().krate != cnum {
                            entry.insert(parent);
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(parent);
                        bfs_queue.push_back(child);
                    }
                }
            };

            let croot = DefId { krate: cnum, index: CRATE_DEF_INDEX };
            for child in self.crate_top_level_items(cnum) {
                add_child(bfs_queue, child, croot);
            }
            while let Some(def) = bfs_queue.pop_front() {
                for child in self.item_children(def) {
                    add_child(bfs_queue, child, def);
                }
            }
        }

        visible_parent_map
    }
}
