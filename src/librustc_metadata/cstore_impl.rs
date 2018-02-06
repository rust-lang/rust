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
use link_args;
use native_libs;
use schema;

use rustc::ty::maps::QueryConfig;
use rustc::middle::cstore::{CrateStore, DepKind,
                            MetadataLoader, LinkMeta,
                            LoadedMacro, EncodedMetadata, NativeLibraryKind};
use rustc::middle::stability::DeprecationEntry;
use rustc::hir::def;
use rustc::session::{CrateDisambiguator, Session};
use rustc::ty::{self, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE, CRATE_DEF_INDEX};
use rustc::hir::map::{DefKey, DefPath, DefPathHash};
use rustc::hir::map::blocks::FnLikeNode;
use rustc::hir::map::definitions::DefPathTable;
use rustc::util::nodemap::{NodeSet, DefIdMap};

use std::any::Any;
use std::rc::Rc;

use syntax::ast;
use syntax::attr;
use syntax::ext::base::SyntaxExtension;
use syntax::parse::filemap_to_stream;
use syntax::symbol::Symbol;
use syntax_pos::{Span, NO_EXPANSION, FileName};
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc::hir;

macro_rules! provide {
    (<$lt:tt> $tcx:ident, $def_id:ident, $other:ident, $cdata:ident,
      $($name:ident => $compute:block)*) => {
        pub fn provide_extern<$lt>(providers: &mut Providers<$lt>) {
            $(fn $name<'a, $lt:$lt, T>($tcx: TyCtxt<'a, $lt, $lt>, def_id_arg: T)
                                    -> <ty::queries::$name<$lt> as
                                        QueryConfig>::Value
                where T: IntoArgs,
            {
                #[allow(unused_variables)]
                let ($def_id, $other) = def_id_arg.into_args();
                assert!(!$def_id.is_local());

                let def_path_hash = $tcx.def_path_hash(DefId {
                    krate: $def_id.krate,
                    index: CRATE_DEF_INDEX
                });
                let dep_node = def_path_hash
                    .to_dep_node(::rustc::dep_graph::DepKind::CrateMetadata);
                // The DepNodeIndex of the DepNode::CrateMetadata should be
                // cached somewhere, so that we can use read_index().
                $tcx.dep_graph.read(dep_node);

                let $cdata = $tcx.crate_data_as_rc_any($def_id.krate);
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

// small trait to work around different signature queries all being defined via
// the macro above.
trait IntoArgs {
    fn into_args(self) -> (DefId, DefId);
}

impl IntoArgs for DefId {
    fn into_args(self) -> (DefId, DefId) { (self, self) }
}

impl IntoArgs for CrateNum {
    fn into_args(self) -> (DefId, DefId) { (self.as_def_id(), self.as_def_id()) }
}

impl IntoArgs for (CrateNum, DefId) {
    fn into_args(self) -> (DefId, DefId) { (self.0.as_def_id(), self.1) }
}

provide! { <'tcx> tcx, def_id, other, cdata,
    type_of => { cdata.get_type(def_id.index, tcx) }
    generics_of => {
        tcx.alloc_generics(cdata.get_generics(def_id.index, tcx.sess))
    }
    predicates_of => { cdata.get_predicates(def_id.index, tcx) }
    super_predicates_of => { cdata.get_super_predicates(def_id.index, tcx) }
    trait_def => {
        tcx.alloc_trait_def(cdata.get_trait_def(def_id.index, tcx.sess))
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
    mir_const_qualif => {
        (cdata.mir_const_qualif(def_id.index), Rc::new(IdxSetBuf::new_empty(0)))
    }
    typeck_tables_of => { cdata.item_body_tables(def_id.index, tcx) }
    fn_sig => { cdata.fn_sig(def_id.index, tcx) }
    inherent_impls => { Rc::new(cdata.get_inherent_implementations_for_type(def_id.index)) }
    is_const_fn => { cdata.is_const_fn(def_id.index) }
    is_foreign_item => { cdata.is_foreign_item(def_id.index) }
    describe_def => { cdata.get_def(def_id.index) }
    def_span => { cdata.get_span(def_id.index, &tcx.sess) }
    lookup_stability => {
        cdata.get_stability(def_id.index).map(|s| tcx.intern_stability(s))
    }
    lookup_deprecation_entry => {
        cdata.get_deprecation(def_id.index).map(DeprecationEntry::external)
    }
    item_attrs => { cdata.get_item_attrs(def_id.index, tcx.sess) }
    // FIXME(#38501) We've skipped a `read` on the `HirBody` of
    // a `fn` when encoding, so the dep-tracking wouldn't work.
    // This is only used by rustdoc anyway, which shouldn't have
    // incremental recompilation ever enabled.
    fn_arg_names => { cdata.get_fn_arg_names(def_id.index) }
    impl_parent => { cdata.get_parent_impl(def_id.index) }
    trait_of_item => { cdata.get_trait_of_item(def_id.index) }
    is_exported_symbol => {
        cdata.exported_symbols.contains(&def_id.index)
    }
    item_body_nested_bodies => { cdata.item_body_nested_bodies(def_id.index) }
    const_is_rvalue_promotable_to_static => {
        cdata.const_is_rvalue_promotable_to_static(def_id.index)
    }
    is_mir_available => { cdata.is_item_mir_available(def_id.index) }

    dylib_dependency_formats => { Rc::new(cdata.get_dylib_dependency_formats()) }
    is_panic_runtime => { cdata.is_panic_runtime(tcx.sess) }
    is_compiler_builtins => { cdata.is_compiler_builtins(tcx.sess) }
    has_global_allocator => { cdata.has_global_allocator() }
    is_sanitizer_runtime => { cdata.is_sanitizer_runtime(tcx.sess) }
    is_profiler_runtime => { cdata.is_profiler_runtime(tcx.sess) }
    panic_strategy => { cdata.panic_strategy() }
    extern_crate => { Rc::new(cdata.extern_crate.get()) }
    is_no_builtins => { cdata.is_no_builtins(tcx.sess) }
    impl_defaultness => { cdata.get_impl_defaultness(def_id.index) }
    exported_symbol_ids => { Rc::new(cdata.get_exported_symbols()) }
    native_libraries => { Rc::new(cdata.get_native_libraries(tcx.sess)) }
    plugin_registrar_fn => {
        cdata.root.plugin_registrar_fn.map(|index| {
            DefId { krate: def_id.krate, index }
        })
    }
    derive_registrar_fn => {
        cdata.root.macro_derive_registrar.map(|index| {
            DefId { krate: def_id.krate, index }
        })
    }
    crate_disambiguator => { cdata.disambiguator() }
    crate_hash => { cdata.hash() }
    original_crate_name => { cdata.name() }

    implementations_of_trait => {
        let mut result = vec![];
        let filter = Some(other);
        cdata.get_implementations_for_trait(filter, &mut result);
        Rc::new(result)
    }

    all_trait_implementations => {
        let mut result = vec![];
        cdata.get_implementations_for_trait(None, &mut result);
        Rc::new(result)
    }

    is_dllimport_foreign_item => {
        cdata.is_dllimport_foreign_item(def_id.index)
    }
    visibility => { cdata.get_visibility(def_id.index) }
    dep_kind => { cdata.dep_kind.get() }
    crate_name => { cdata.name }
    item_children => {
        let mut result = vec![];
        cdata.each_child_of_item(def_id.index, |child| result.push(child), tcx.sess);
        Rc::new(result)
    }
    defined_lang_items => { Rc::new(cdata.get_lang_items()) }
    missing_lang_items => { Rc::new(cdata.get_missing_lang_items()) }

    extern_const_body => {
        debug!("item_body({:?}): inlining item", def_id);
        cdata.extern_const_body(tcx, def_id.index)
    }

    missing_extern_crate_item => {
        match cdata.extern_crate.get() {
            Some(extern_crate) if !extern_crate.direct => true,
            _ => false,
        }
    }

    used_crate_source => { Rc::new(cdata.source.clone()) }

    has_copy_closures => { cdata.has_copy_closures(tcx.sess) }
    has_clone_closures => { cdata.has_clone_closures(tcx.sess) }
}

pub fn provide<'tcx>(providers: &mut Providers<'tcx>) {
    fn is_const_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> bool {
        let node_id = tcx.hir.as_local_node_id(def_id)
                             .expect("Non-local call to local provider is_const_fn");

        if let Some(fn_like) = FnLikeNode::from_node(tcx.hir.get(node_id)) {
            fn_like.constness() == hir::Constness::Const
        } else {
            false
        }
    }

    // FIXME(#44234) - almost all of these queries have no sub-queries and
    // therefore no actual inputs, they're just reading tables calculated in
    // resolve! Does this work? Unsure! That's what the issue is about
    *providers = Providers {
        is_const_fn,
        is_dllimport_foreign_item: |tcx, id| {
            tcx.native_library_kind(id) == Some(NativeLibraryKind::NativeUnknown)
        },
        is_statically_included_foreign_item: |tcx, id| {
            match tcx.native_library_kind(id) {
                Some(NativeLibraryKind::NativeStatic) |
                Some(NativeLibraryKind::NativeStaticNobundle) => true,
                _ => false,
            }
        },
        native_library_kind: |tcx, id| {
            tcx.native_libraries(id.krate)
                .iter()
                .filter(|lib| native_libs::relevant_lib(&tcx.sess, lib))
                .find(|l| l.foreign_items.contains(&id))
                .map(|l| l.kind)
        },
        native_libraries: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            Rc::new(native_libs::collect(tcx))
        },
        link_args: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            Rc::new(link_args::collect(tcx))
        },

        // Returns a map from a sufficiently visible external item (i.e. an
        // external item that is visible from at least one local module) to a
        // sufficiently visible parent (considering modules that re-export the
        // external item to be parents).
        visible_parent_map: |tcx, cnum| {
            use std::collections::vec_deque::VecDeque;
            use std::collections::hash_map::Entry;

            assert_eq!(cnum, LOCAL_CRATE);
            let mut visible_parent_map: DefIdMap<DefId> = DefIdMap();

            // Issue 46112: We want the map to prefer the shortest
            // paths when reporting the path to an item. Therefore we
            // build up the map via a breadth-first search (BFS),
            // which naturally yields minimal-length paths.
            //
            // Note that it needs to be a BFS over the whole forest of
            // crates, not just each individual crate; otherwise you
            // only get paths that are locally minimal with respect to
            // whatever crate we happened to encounter first in this
            // traversal, but not globally minimal across all crates.
            let bfs_queue = &mut VecDeque::new();

            // Preferring shortest paths alone does not guarantee a
            // deterministic result; so sort by crate num to avoid
            // hashtable iteration non-determinism. This only makes
            // things as deterministic as crate-nums assignment is,
            // which is to say, its not deterministic in general. But
            // we believe that libstd is consistently assigned crate
            // num 1, so it should be enough to resolve #46112.
            let mut crates: Vec<CrateNum> = (*tcx.crates()).clone();
            crates.sort();

            for &cnum in crates.iter() {
                // Ignore crates without a corresponding local `extern crate` item.
                if tcx.missing_extern_crate_item(cnum) {
                    continue
                }

                bfs_queue.push_back(DefId {
                    krate: cnum,
                    index: CRATE_DEF_INDEX
                });
            }

            // (restrict scope of mutable-borrow of `visible_parent_map`)
            {
                let visible_parent_map = &mut visible_parent_map;
                let mut add_child = |bfs_queue: &mut VecDeque<_>,
                                     child: &def::Export,
                                     parent: DefId| {
                    if child.vis != ty::Visibility::Public {
                        return;
                    }

                    let child = child.def.def_id();

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

                while let Some(def) = bfs_queue.pop_front() {
                    for child in tcx.item_children(def).iter() {
                        add_child(bfs_queue, child, def);
                    }
                }
            }

            Rc::new(visible_parent_map)
        },

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

    fn visibility_untracked(&self, def: DefId) -> ty::Visibility {
        self.get_crate_data(def.krate).get_visibility(def.index)
    }

    fn item_generics_cloned_untracked(&self, def: DefId, sess: &Session) -> ty::Generics {
        self.get_crate_data(def.krate).get_generics(def.index, sess)
    }

    fn associated_item_cloned_untracked(&self, def: DefId) -> ty::AssociatedItem
    {
        self.get_crate_data(def.krate).get_associated_item(def.index)
    }

    fn dep_kind_untracked(&self, cnum: CrateNum) -> DepKind
    {
        self.get_crate_data(cnum).dep_kind.get()
    }

    fn export_macros_untracked(&self, cnum: CrateNum) {
        let data = self.get_crate_data(cnum);
        if data.dep_kind.get() == DepKind::UnexportedMacrosOnly {
            data.dep_kind.set(DepKind::MacrosOnly)
        }
    }

    fn crate_name_untracked(&self, cnum: CrateNum) -> Symbol
    {
        self.get_crate_data(cnum).name
    }

    fn crate_disambiguator_untracked(&self, cnum: CrateNum) -> CrateDisambiguator
    {
        self.get_crate_data(cnum).disambiguator()
    }

    fn crate_hash_untracked(&self, cnum: CrateNum) -> hir::svh::Svh
    {
        self.get_crate_data(cnum).hash()
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

    fn struct_field_names_untracked(&self, def: DefId) -> Vec<ast::Name>
    {
        self.get_crate_data(def.krate).get_struct_field_names(def.index)
    }

    fn item_children_untracked(&self, def_id: DefId, sess: &Session) -> Vec<def::Export>
    {
        let mut result = vec![];
        self.get_crate_data(def_id.krate)
            .each_child_of_item(def_id.index, |child| result.push(child), sess);
        result
    }

    fn load_macro_untracked(&self, id: DefId, sess: &Session) -> LoadedMacro {
        let data = self.get_crate_data(id.krate);
        if let Some(ref proc_macros) = data.proc_macros {
            return LoadedMacro::ProcMacro(proc_macros[id.index.to_proc_macro_index()].1.clone());
        } else if data.name == "proc_macro" &&
                  self.get_crate_data(id.krate).item_name(id.index) == "quote" {
            let ext = SyntaxExtension::ProcMacro(Box::new(::proc_macro::__internal::Quoter));
            return LoadedMacro::ProcMacro(Rc::new(ext));
        }

        let (name, def) = data.get_macro(id.index);
        let source_name = FileName::Macros(name.to_string());

        let filemap = sess.parse_sess.codemap().new_filemap(source_name, def.body);
        let local_span = Span::new(filemap.start_pos, filemap.end_pos, NO_EXPANSION);
        let body = filemap_to_stream(&sess.parse_sess, filemap, None);

        // Mark the attrs as used
        let attrs = data.get_item_attrs(id.index, sess);
        for attr in attrs.iter() {
            attr::mark_used(attr);
        }

        let name = data.def_key(id.index).disambiguated_data.data
            .get_opt_name().expect("no name in load_macro");
        sess.imported_macro_spans.borrow_mut()
            .insert(local_span, (name.to_string(), data.get_span(id.index, sess)));

        LoadedMacro::MacroDef(ast::Item {
            ident: ast::Ident::from_str(&name),
            id: ast::DUMMY_NODE_ID,
            span: local_span,
            attrs: attrs.iter().cloned().collect(),
            node: ast::ItemKind::MacroDef(ast::MacroDef {
                tokens: body.into(),
                legacy: def.legacy,
            }),
            vis: ast::Visibility::Inherited,
            tokens: None,
        })
    }

    fn crates_untracked(&self) -> Vec<CrateNum>
    {
        let mut result = vec![];
        self.iter_crate_data(|cnum, _| result.push(cnum));
        result
    }

    fn extern_mod_stmt_cnum_untracked(&self, emod_id: ast::NodeId) -> Option<CrateNum>
    {
        self.do_extern_mod_stmt_cnum(emod_id)
    }

    fn postorder_cnums_untracked(&self) -> Vec<CrateNum> {
        self.do_postorder_cnums_untracked()
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
}
