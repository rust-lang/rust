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
use encoder;
use schema;

use rustc::dep_graph::DepTrackingMapConfig;
use rustc::middle::cstore::{CrateStore, CrateSource, LibSource, DepKind,
                            NativeLibrary, MetadataLoader, LinkMeta,
                            LinkagePreference, LoadedMacro, EncodedMetadata};
use rustc::hir::def;
use rustc::middle::lang_items;
use rustc::session::Session;
use rustc::ty::{self, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::hir::map::{DefKey, DefPath, DisambiguatedDefPathData, DefPathHash};
use rustc::hir::map::blocks::FnLikeNode;
use rustc::hir::map::definitions::{DefPathTable, GlobalMetaDataKind};
use rustc::util::nodemap::{NodeSet, DefIdMap};
use rustc_back::PanicStrategy;

use std::any::Any;
use std::rc::Rc;

use syntax::ast;
use syntax::attr;
use syntax::parse::filemap_to_stream;
use syntax::symbol::Symbol;
use syntax_pos::{Span, NO_EXPANSION};
use rustc::hir::svh::Svh;
use rustc::hir;

macro_rules! provide {
    (<$lt:tt> $tcx:ident, $def_id:ident, $cdata:ident, $($name:ident => $compute:block)*) => {
        pub fn provide<$lt>(providers: &mut Providers<$lt>) {
            $(fn $name<'a, $lt:$lt>($tcx: TyCtxt<'a, $lt, $lt>, $def_id: DefId)
                                    -> <ty::queries::$name<$lt> as
                                        DepTrackingMapConfig>::Value {
                assert!(!$def_id.is_local());

                let def_path_hash = $tcx.def_path_hash($def_id);
                let dep_node = def_path_hash.to_dep_node(::rustc::dep_graph::DepKind::MetaData);

                $tcx.dep_graph.read(dep_node);

                let $cdata = $tcx.sess.cstore.crate_data_as_rc_any($def_id.krate);
                let $cdata = $cdata.downcast_ref::<cstore::CrateMetadata>()
                    .expect("CrateStore crated ata is not a CrateMetadata");
                $compute
            })*

            *providers = Providers {
                $($name,)*
                ..*providers
            };
        }
    }
}

provide! { <'tcx> tcx, def_id, cdata,
    type_of => { cdata.get_type(def_id.index, tcx) }
    generics_of => { tcx.alloc_generics(cdata.get_generics(def_id.index)) }
    predicates_of => { cdata.get_predicates(def_id.index, tcx) }
    super_predicates_of => { cdata.get_super_predicates(def_id.index, tcx) }
    trait_def => {
        tcx.alloc_trait_def(cdata.get_trait_def(def_id.index))
    }
    adt_def => { cdata.get_adt_def(def_id.index, tcx) }
    adt_destructor => {
        let _ = cdata;
        tcx.calculate_dtor(def_id, &mut |_,_| Ok(()))
    }
    variances_of => { Rc::new(cdata.get_item_variances(def_id.index)) }
    associated_item_def_ids => {
        let mut result = vec![];
        cdata.each_child_of_item(def_id.index,
          |child| result.push(child.def.def_id()), tcx.sess);
        Rc::new(result)
    }
    associated_item => { cdata.get_associated_item(def_id.index) }
    impl_trait_ref => { cdata.get_impl_trait(def_id.index, tcx) }
    impl_polarity => { cdata.get_impl_polarity(def_id.index) }
    coerce_unsized_info => {
        cdata.get_coerce_unsized_info(def_id.index).unwrap_or_else(|| {
            bug!("coerce_unsized_info: `{:?}` is missing its info", def_id);
        })
    }
    optimized_mir => {
        let mir = cdata.maybe_get_optimized_mir(tcx, def_id.index).unwrap_or_else(|| {
            bug!("get_optimized_mir: missing MIR for `{:?}`", def_id)
        });

        let mir = tcx.alloc_mir(mir);

        mir
    }
    mir_const_qualif => { cdata.mir_const_qualif(def_id.index) }
    typeck_tables_of => { cdata.item_body_tables(def_id.index, tcx) }
    closure_kind => { cdata.closure_kind(def_id.index) }
    closure_type => { cdata.closure_ty(def_id.index, tcx) }
    inherent_impls => { Rc::new(cdata.get_inherent_implementations_for_type(def_id.index)) }
    is_const_fn => { cdata.is_const_fn(def_id.index) }
    is_foreign_item => { cdata.is_foreign_item(def_id.index) }
    is_default_impl => { cdata.is_default_impl(def_id.index) }
    describe_def => { cdata.get_def(def_id.index) }
    def_span => { cdata.get_span(def_id.index, &tcx.sess) }
    stability => { cdata.get_stability(def_id.index) }
    deprecation => { cdata.get_deprecation(def_id.index) }
    item_attrs => { cdata.get_item_attrs(def_id.index, &tcx.dep_graph) }
    // FIXME(#38501) We've skipped a `read` on the `HirBody` of
    // a `fn` when encoding, so the dep-tracking wouldn't work.
    // This is only used by rustdoc anyway, which shouldn't have
    // incremental recompilation ever enabled.
    fn_arg_names => { cdata.get_fn_arg_names(def_id.index) }
    impl_parent => { cdata.get_parent_impl(def_id.index) }
    trait_of_item => { cdata.get_trait_of_item(def_id.index) }
    is_exported_symbol => {
        let dep_node = cdata.metadata_dep_node(GlobalMetaDataKind::ExportedSymbols);
        cdata.exported_symbols.get(&tcx.dep_graph, dep_node).contains(&def_id.index)
    }
    item_body_nested_bodies => { Rc::new(cdata.item_body_nested_bodies(def_id.index)) }
    const_is_rvalue_promotable_to_static => {
        cdata.const_is_rvalue_promotable_to_static(def_id.index)
    }
    is_mir_available => { cdata.is_item_mir_available(def_id.index) }

    dylib_dependency_formats => { Rc::new(cdata.get_dylib_dependency_formats(&tcx.dep_graph)) }
    is_allocator => { cdata.is_allocator(&tcx.dep_graph) }
    is_panic_runtime => { cdata.is_panic_runtime(&tcx.dep_graph) }
    extern_crate => { Rc::new(cdata.extern_crate.get()) }
}

pub fn provide_local<'tcx>(providers: &mut Providers<'tcx>) {
    fn is_const_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> bool {
        let node_id = tcx.hir.as_local_node_id(def_id)
                             .expect("Non-local call to local provider is_const_fn");

        if let Some(fn_like) = FnLikeNode::from_node(tcx.hir.get(node_id)) {
            fn_like.constness() == hir::Constness::Const
        } else {
            false
        }
    }

    *providers = Providers {
        is_const_fn,
        ..*providers
    };
}

impl CrateStore for cstore::CStore {
    fn crate_data_as_rc_any(&self, krate: CrateNum) -> Rc<Any> {
        self.get_crate_data(krate)
    }

    fn metadata_loader(&self) -> &MetadataLoader {
        &*self.metadata_loader
    }

    fn visibility(&self, def: DefId) -> ty::Visibility {
        self.read_dep_node(def);
        self.get_crate_data(def.krate).get_visibility(def.index)
    }

    fn item_generics_cloned(&self, def: DefId) -> ty::Generics {
        self.read_dep_node(def);
        self.get_crate_data(def.krate).get_generics(def.index)
    }

    fn implementations_of_trait(&self, filter: Option<DefId>) -> Vec<DefId>
    {
        let mut result = vec![];

        self.iter_crate_data(|_, cdata| {
            cdata.get_implementations_for_trait(filter, &self.dep_graph, &mut result)
        });
        result
    }

    fn impl_defaultness(&self, def: DefId) -> hir::Defaultness
    {
        self.read_dep_node(def);
        self.get_crate_data(def.krate).get_impl_defaultness(def.index)
    }

    fn associated_item_cloned(&self, def: DefId) -> ty::AssociatedItem
    {
        self.read_dep_node(def);
        self.get_crate_data(def.krate).get_associated_item(def.index)
    }

    fn is_statically_included_foreign_item(&self, def_id: DefId) -> bool
    {
        self.do_is_statically_included_foreign_item(def_id)
    }

    fn is_dllimport_foreign_item(&self, def_id: DefId) -> bool {
        if def_id.krate == LOCAL_CRATE {
            self.dllimport_foreign_items.borrow().contains(&def_id.index)
        } else {
            self.get_crate_data(def_id.krate)
                .is_dllimport_foreign_item(def_id.index, &self.dep_graph)
        }
    }

    fn dep_kind(&self, cnum: CrateNum) -> DepKind
    {
        let data = self.get_crate_data(cnum);
        let dep_node = data.metadata_dep_node(GlobalMetaDataKind::CrateDeps);
        self.dep_graph.read(dep_node);
        data.dep_kind.get()
    }

    fn export_macros(&self, cnum: CrateNum) {
        let data = self.get_crate_data(cnum);
        let dep_node = data.metadata_dep_node(GlobalMetaDataKind::CrateDeps);

        self.dep_graph.read(dep_node);
        if data.dep_kind.get() == DepKind::UnexportedMacrosOnly {
            data.dep_kind.set(DepKind::MacrosOnly)
        }
    }

    fn lang_items(&self, cnum: CrateNum) -> Vec<(DefIndex, usize)>
    {
        self.get_crate_data(cnum).get_lang_items(&self.dep_graph)
    }

    fn missing_lang_items(&self, cnum: CrateNum)
                          -> Vec<lang_items::LangItem>
    {
        self.get_crate_data(cnum).get_missing_lang_items(&self.dep_graph)
    }

    fn is_compiler_builtins(&self, cnum: CrateNum) -> bool {
        self.get_crate_data(cnum).is_compiler_builtins(&self.dep_graph)
    }

    fn is_sanitizer_runtime(&self, cnum: CrateNum) -> bool {
        self.get_crate_data(cnum).is_sanitizer_runtime(&self.dep_graph)
    }

    fn is_profiler_runtime(&self, cnum: CrateNum) -> bool {
        self.get_crate_data(cnum).is_profiler_runtime(&self.dep_graph)
    }

    fn panic_strategy(&self, cnum: CrateNum) -> PanicStrategy {
        self.get_crate_data(cnum).panic_strategy(&self.dep_graph)
    }

    fn crate_name(&self, cnum: CrateNum) -> Symbol
    {
        self.get_crate_data(cnum).name
    }

    fn original_crate_name(&self, cnum: CrateNum) -> Symbol
    {
        self.get_crate_data(cnum).name()
    }

    fn crate_hash(&self, cnum: CrateNum) -> Svh
    {
        self.get_crate_hash(cnum)
    }

    fn crate_disambiguator(&self, cnum: CrateNum) -> Symbol
    {
        self.get_crate_data(cnum).disambiguator()
    }

    fn plugin_registrar_fn(&self, cnum: CrateNum) -> Option<DefId>
    {
        self.get_crate_data(cnum).root.plugin_registrar_fn.map(|index| DefId {
            krate: cnum,
            index: index
        })
    }

    fn derive_registrar_fn(&self, cnum: CrateNum) -> Option<DefId>
    {
        self.get_crate_data(cnum).root.macro_derive_registrar.map(|index| DefId {
            krate: cnum,
            index: index
        })
    }

    fn native_libraries(&self, cnum: CrateNum) -> Vec<NativeLibrary>
    {
        self.get_crate_data(cnum).get_native_libraries(&self.dep_graph)
    }

    fn exported_symbols(&self, cnum: CrateNum) -> Vec<DefId>
    {
        self.get_crate_data(cnum).get_exported_symbols(&self.dep_graph)
    }

    fn is_no_builtins(&self, cnum: CrateNum) -> bool {
        self.get_crate_data(cnum).is_no_builtins(&self.dep_graph)
    }

    fn retrace_path(&self,
                    cnum: CrateNum,
                    path: &[DisambiguatedDefPathData])
                    -> Option<DefId> {
        let cdata = self.get_crate_data(cnum);
        cdata.def_path_table
             .retrace_path(&path)
             .map(|index| DefId { krate: cnum, index: index })
    }

    /// Returns the `DefKey` for a given `DefId`. This indicates the
    /// parent `DefId` as well as some idea of what kind of data the
    /// `DefId` refers to.
    fn def_key(&self, def: DefId) -> DefKey {
        // Note: loading the def-key (or def-path) for a def-id is not
        // a *read* of its metadata. This is because the def-id is
        // really just an interned shorthand for a def-path, which is the
        // canonical name for an item.
        //
        // self.dep_graph.read(DepNode::MetaData(def));
        self.get_crate_data(def.krate).def_key(def.index)
    }

    fn def_path(&self, def: DefId) -> DefPath {
        // See `Note` above in `def_key()` for why this read is
        // commented out:
        //
        // self.dep_graph.read(DepNode::MetaData(def));
        self.get_crate_data(def.krate).def_path(def.index)
    }

    fn def_path_hash(&self, def: DefId) -> DefPathHash {
        self.get_crate_data(def.krate).def_path_hash(def.index)
    }

    fn def_path_table(&self, cnum: CrateNum) -> Rc<DefPathTable> {
        self.get_crate_data(cnum).def_path_table.clone()
    }

    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>
    {
        self.read_dep_node(def);
        self.get_crate_data(def.krate).get_struct_field_names(def.index)
    }

    fn item_children(&self, def_id: DefId, sess: &Session) -> Vec<def::Export>
    {
        self.read_dep_node(def_id);
        let mut result = vec![];
        self.get_crate_data(def_id.krate)
            .each_child_of_item(def_id.index, |child| result.push(child), sess);
        result
    }

    fn load_macro(&self, id: DefId, sess: &Session) -> LoadedMacro {
        let data = self.get_crate_data(id.krate);
        if let Some(ref proc_macros) = data.proc_macros {
            return LoadedMacro::ProcMacro(proc_macros[id.index.as_usize() - 1].1.clone());
        }

        let (name, def) = data.get_macro(id.index);
        let source_name = format!("<{} macros>", name);

        let filemap = sess.parse_sess.codemap().new_filemap(source_name, def.body);
        let local_span = Span { lo: filemap.start_pos, hi: filemap.end_pos, ctxt: NO_EXPANSION };
        let body = filemap_to_stream(&sess.parse_sess, filemap, None);

        // Mark the attrs as used
        let attrs = data.get_item_attrs(id.index, &self.dep_graph);
        for attr in attrs.iter() {
            attr::mark_used(attr);
        }

        let name = data.def_key(id.index).disambiguated_data.data
            .get_opt_name().expect("no name in load_macro");
        sess.imported_macro_spans.borrow_mut()
            .insert(local_span, (name.to_string(), data.get_span(id.index, sess)));

        LoadedMacro::MacroDef(ast::Item {
            ident: ast::Ident::with_empty_ctxt(name),
            id: ast::DUMMY_NODE_ID,
            span: local_span,
            attrs: attrs.iter().cloned().collect(),
            node: ast::ItemKind::MacroDef(ast::MacroDef {
                tokens: body.into(),
                legacy: def.legacy,
            }),
            vis: ast::Visibility::Inherited,
        })
    }

    fn item_body<'a, 'tcx>(&self,
                           tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           def_id: DefId)
                           -> &'tcx hir::Body {
        self.read_dep_node(def_id);

        if let Some(cached) = tcx.hir.get_inlined_body_untracked(def_id) {
            return cached;
        }

        debug!("item_body({:?}): inlining item", def_id);

        self.get_crate_data(def_id.krate).item_body(tcx, def_id.index)
    }

    fn crates(&self) -> Vec<CrateNum>
    {
        let mut result = vec![];
        self.iter_crate_data(|cnum, _| result.push(cnum));
        result
    }

    fn used_libraries(&self) -> Vec<NativeLibrary>
    {
        self.get_used_libraries().borrow().clone()
    }

    fn used_link_args(&self) -> Vec<String>
    {
        self.get_used_link_args().borrow().clone()
    }
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(CrateNum, LibSource)>
    {
        self.do_get_used_crates(prefer)
    }

    fn used_crate_source(&self, cnum: CrateNum) -> CrateSource
    {
        self.get_crate_data(cnum).source.clone()
    }

    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum>
    {
        self.do_extern_mod_stmt_cnum(emod_id)
    }

    fn encode_metadata<'a, 'tcx>(&self,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 link_meta: &LinkMeta,
                                 reachable: &NodeSet)
                                 -> EncodedMetadata
    {
        encoder::encode_metadata(tcx, link_meta, reachable)
    }

    fn metadata_encoding_version(&self) -> &[u8]
    {
        schema::METADATA_HEADER
    }

    /// Returns a map from a sufficiently visible external item (i.e. an external item that is
    /// visible from at least one local module) to a sufficiently visible parent (considering
    /// modules that re-export the external item to be parents).
    fn visible_parent_map<'a>(&'a self, sess: &Session) -> ::std::cell::Ref<'a, DefIdMap<DefId>> {
        {
            let visible_parent_map = self.visible_parent_map.borrow();
            if !visible_parent_map.is_empty() {
                return visible_parent_map;
            }
        }

        use std::collections::vec_deque::VecDeque;
        use std::collections::hash_map::Entry;

        let mut visible_parent_map = self.visible_parent_map.borrow_mut();

        for cnum in (1 .. self.next_crate_num().as_usize()).map(CrateNum::new) {
            let cdata = self.get_crate_data(cnum);

            match cdata.extern_crate.get() {
                // Ignore crates without a corresponding local `extern crate` item.
                Some(extern_crate) if !extern_crate.direct => continue,
                _ => {},
            }

            let mut bfs_queue = &mut VecDeque::new();
            let mut add_child = |bfs_queue: &mut VecDeque<_>, child: def::Export, parent: DefId| {
                let child = child.def.def_id();

                if self.visibility(child) != ty::Visibility::Public {
                    return;
                }

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

            bfs_queue.push_back(DefId {
                krate: cnum,
                index: CRATE_DEF_INDEX
            });
            while let Some(def) = bfs_queue.pop_front() {
                for child in self.item_children(def, sess) {
                    add_child(bfs_queue, child, def);
                }
            }
        }

        drop(visible_parent_map);
        self.visible_parent_map.borrow()
    }
}
