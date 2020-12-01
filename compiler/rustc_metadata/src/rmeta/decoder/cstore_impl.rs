use crate::creader::{CStore, LoadedMacro};
use crate::foreign_modules;
use crate::link_args;
use crate::native_libs;
use crate::rmeta::{self, encoder};

use rustc_ast as ast;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_data_structures::stable_map::FxHashMap;
use rustc_data_structures::svh::Svh;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_middle::hir::exports::Export;
use rustc_middle::middle::cstore::ForeignModule;
use rustc_middle::middle::cstore::{CrateSource, CrateStore, EncodedMetadata};
use rustc_middle::middle::exported_symbols::ExportedSymbol;
use rustc_middle::middle::stability::DeprecationEntry;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::utils::NativeLibKind;
use rustc_session::{CrateDisambiguator, Session};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::symbol::Symbol;

use rustc_data_structures::sync::Lrc;
use rustc_span::ExpnId;
use smallvec::SmallVec;
use std::any::Any;

macro_rules! provide {
    (<$lt:tt> $tcx:ident, $def_id:ident, $other:ident, $cdata:ident,
      $($name:ident => $compute:block)*) => {
        pub fn provide_extern(providers: &mut Providers) {
            $(fn $name<$lt>(
                $tcx: TyCtxt<$lt>,
                def_id_arg: ty::query::query_keys::$name<$lt>,
            ) -> ty::query::query_values::$name<$lt> {
                let _prof_timer =
                    $tcx.prof.generic_activity(concat!("metadata_decode_entry_", stringify!($name)));

                #[allow(unused_variables)]
                let ($def_id, $other) = def_id_arg.into_args();
                assert!(!$def_id.is_local());

                let $cdata = CStore::from_tcx($tcx).get_crate_data($def_id.krate);

                if $tcx.dep_graph.is_fully_enabled() {
                    let crate_dep_node_index = $cdata.get_crate_dep_node_index($tcx);
                    $tcx.dep_graph.read_index(crate_dep_node_index);
                }

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
    fn into_args(self) -> (DefId, DefId) {
        (self, self)
    }
}

impl IntoArgs for CrateNum {
    fn into_args(self) -> (DefId, DefId) {
        (self.as_def_id(), self.as_def_id())
    }
}

impl IntoArgs for (CrateNum, DefId) {
    fn into_args(self) -> (DefId, DefId) {
        (self.0.as_def_id(), self.1)
    }
}

provide! { <'tcx> tcx, def_id, other, cdata,
    type_of => { cdata.get_type(def_id.index, tcx) }
    generics_of => { cdata.get_generics(def_id.index, tcx.sess) }
    explicit_predicates_of => { cdata.get_explicit_predicates(def_id.index, tcx) }
    inferred_outlives_of => { cdata.get_inferred_outlives(def_id.index, tcx) }
    super_predicates_of => { cdata.get_super_predicates(def_id.index, tcx) }
    explicit_item_bounds => { cdata.get_explicit_item_bounds(def_id.index, tcx) }
    trait_def => { cdata.get_trait_def(def_id.index, tcx.sess) }
    adt_def => { cdata.get_adt_def(def_id.index, tcx) }
    adt_destructor => {
        let _ = cdata;
        tcx.calculate_dtor(def_id, |_,_| Ok(()))
    }
    variances_of => { tcx.arena.alloc_from_iter(cdata.get_item_variances(def_id.index)) }
    associated_item_def_ids => {
        let mut result = SmallVec::<[_; 8]>::new();
        cdata.each_child_of_item(def_id.index,
          |child| result.push(child.res.def_id()), tcx.sess);
        tcx.arena.alloc_slice(&result)
    }
    associated_item => { cdata.get_associated_item(def_id.index, tcx.sess) }
    impl_trait_ref => { cdata.get_impl_trait(def_id.index, tcx) }
    impl_polarity => { cdata.get_impl_polarity(def_id.index) }
    coerce_unsized_info => {
        cdata.get_coerce_unsized_info(def_id.index).unwrap_or_else(|| {
            bug!("coerce_unsized_info: `{:?}` is missing its info", def_id);
        })
    }
    optimized_mir => { tcx.arena.alloc(cdata.get_optimized_mir(tcx, def_id.index)) }
    promoted_mir => { tcx.arena.alloc(cdata.get_promoted_mir(tcx, def_id.index)) }
    mir_abstract_const => { cdata.get_mir_abstract_const(tcx, def_id.index) }
    unused_generic_params => { cdata.get_unused_generic_params(def_id.index) }
    mir_const_qualif => { cdata.mir_const_qualif(def_id.index) }
    fn_sig => { cdata.fn_sig(def_id.index, tcx) }
    inherent_impls => { cdata.get_inherent_implementations_for_type(tcx, def_id.index) }
    is_const_fn_raw => { cdata.is_const_fn_raw(def_id.index) }
    asyncness => { cdata.asyncness(def_id.index) }
    is_foreign_item => { cdata.is_foreign_item(def_id.index) }
    static_mutability => { cdata.static_mutability(def_id.index) }
    generator_kind => { cdata.generator_kind(def_id.index) }
    def_kind => { cdata.def_kind(def_id.index) }
    def_span => { cdata.get_span(def_id.index, &tcx.sess) }
    lookup_stability => {
        cdata.get_stability(def_id.index).map(|s| tcx.intern_stability(s))
    }
    lookup_const_stability => {
        cdata.get_const_stability(def_id.index).map(|s| tcx.intern_const_stability(s))
    }
    lookup_deprecation_entry => {
        cdata.get_deprecation(def_id.index).map(DeprecationEntry::external)
    }
    item_attrs => { tcx.arena.alloc_from_iter(
        cdata.get_item_attrs(def_id.index, tcx.sess)
    ) }
    fn_arg_names => { cdata.get_fn_param_names(tcx, def_id.index) }
    rendered_const => { cdata.get_rendered_const(def_id.index) }
    impl_parent => { cdata.get_parent_impl(def_id.index) }
    trait_of_item => { cdata.get_trait_of_item(def_id.index) }
    is_mir_available => { cdata.is_item_mir_available(def_id.index) }

    dylib_dependency_formats => { cdata.get_dylib_dependency_formats(tcx) }
    is_panic_runtime => { cdata.root.panic_runtime }
    is_compiler_builtins => { cdata.root.compiler_builtins }
    has_global_allocator => { cdata.root.has_global_allocator }
    has_panic_handler => { cdata.root.has_panic_handler }
    is_profiler_runtime => { cdata.root.profiler_runtime }
    panic_strategy => { cdata.root.panic_strategy }
    extern_crate => {
        let r = *cdata.extern_crate.lock();
        r.map(|c| &*tcx.arena.alloc(c))
    }
    is_no_builtins => { cdata.root.no_builtins }
    symbol_mangling_version => { cdata.root.symbol_mangling_version }
    impl_defaultness => { cdata.get_impl_defaultness(def_id.index) }
    reachable_non_generics => {
        let reachable_non_generics = tcx
            .exported_symbols(cdata.cnum)
            .iter()
            .filter_map(|&(exported_symbol, export_level)| {
                if let ExportedSymbol::NonGeneric(def_id) = exported_symbol {
                    Some((def_id, export_level))
                } else {
                    None
                }
            })
            .collect();

        reachable_non_generics
    }
    native_libraries => { Lrc::new(cdata.get_native_libraries(tcx.sess)) }
    foreign_modules => { cdata.get_foreign_modules(tcx) }
    plugin_registrar_fn => {
        cdata.root.plugin_registrar_fn.map(|index| {
            DefId { krate: def_id.krate, index }
        })
    }
    proc_macro_decls_static => {
        cdata.root.proc_macro_data.as_ref().map(|data| {
            DefId {
                krate: def_id.krate,
                index: data.proc_macro_decls_static,
            }
        })
    }
    crate_disambiguator => { cdata.root.disambiguator }
    crate_hash => { cdata.root.hash }
    crate_host_hash => { cdata.host_hash }
    original_crate_name => { cdata.root.name }

    extra_filename => { cdata.root.extra_filename.clone() }

    implementations_of_trait => {
        cdata.get_implementations_for_trait(tcx, Some(other))
    }

    all_trait_implementations => {
        cdata.get_implementations_for_trait(tcx, None)
    }

    visibility => { cdata.get_visibility(def_id.index) }
    dep_kind => {
        let r = *cdata.dep_kind.lock();
        r
    }
    crate_name => { cdata.root.name }
    item_children => {
        let mut result = SmallVec::<[_; 8]>::new();
        cdata.each_child_of_item(def_id.index, |child| result.push(child), tcx.sess);
        tcx.arena.alloc_slice(&result)
    }
    defined_lib_features => { cdata.get_lib_features(tcx) }
    defined_lang_items => { cdata.get_lang_items(tcx) }
    diagnostic_items => { cdata.get_diagnostic_items() }
    missing_lang_items => { cdata.get_missing_lang_items(tcx) }

    missing_extern_crate_item => {
        let r = matches!(*cdata.extern_crate.borrow(), Some(extern_crate) if !extern_crate.is_direct());
        r
    }

    used_crate_source => { Lrc::new(cdata.source.clone()) }

    exported_symbols => {
        let syms = cdata.exported_symbols(tcx);

        // FIXME rust-lang/rust#64319, rust-lang/rust#64872: We want
        // to block export of generics from dylibs, but we must fix
        // rust-lang/rust#65890 before we can do that robustly.

        syms
    }

    crate_extern_paths => { cdata.source().paths().cloned().collect() }
    expn_that_defined => { cdata.get_expn_that_defined(def_id.index, tcx.sess) }
}

pub fn provide(providers: &mut Providers) {
    // FIXME(#44234) - almost all of these queries have no sub-queries and
    // therefore no actual inputs, they're just reading tables calculated in
    // resolve! Does this work? Unsure! That's what the issue is about
    *providers = Providers {
        is_dllimport_foreign_item: |tcx, id| match tcx.native_library_kind(id) {
            Some(NativeLibKind::Dylib | NativeLibKind::RawDylib | NativeLibKind::Unspecified) => {
                true
            }
            _ => false,
        },
        is_statically_included_foreign_item: |tcx, id| {
            matches!(
                tcx.native_library_kind(id),
                Some(NativeLibKind::StaticBundle | NativeLibKind::StaticNoBundle)
            )
        },
        native_library_kind: |tcx, id| {
            tcx.native_libraries(id.krate)
                .iter()
                .filter(|lib| native_libs::relevant_lib(&tcx.sess, lib))
                .find(|lib| {
                    let fm_id = match lib.foreign_module {
                        Some(id) => id,
                        None => return false,
                    };
                    let map = tcx.foreign_modules(id.krate);
                    map.get(&fm_id)
                        .expect("failed to find foreign module")
                        .foreign_items
                        .contains(&id)
                })
                .map(|l| l.kind)
        },
        native_libraries: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            Lrc::new(native_libs::collect(tcx))
        },
        foreign_modules: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            let modules: FxHashMap<DefId, ForeignModule> =
                foreign_modules::collect(tcx).into_iter().map(|m| (m.def_id, m)).collect();
            Lrc::new(modules)
        },
        link_args: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            Lrc::new(link_args::collect(tcx))
        },

        // Returns a map from a sufficiently visible external item (i.e., an
        // external item that is visible from at least one local module) to a
        // sufficiently visible parent (considering modules that re-export the
        // external item to be parents).
        visible_parent_map: |tcx, cnum| {
            use std::collections::hash_map::Entry;
            use std::collections::vec_deque::VecDeque;

            assert_eq!(cnum, LOCAL_CRATE);
            let mut visible_parent_map: DefIdMap<DefId> = Default::default();

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
            let mut crates: Vec<CrateNum> = (*tcx.crates()).to_owned();
            crates.sort();

            for &cnum in crates.iter() {
                // Ignore crates without a corresponding local `extern crate` item.
                if tcx.missing_extern_crate_item(cnum) {
                    continue;
                }

                bfs_queue.push_back(DefId { krate: cnum, index: CRATE_DEF_INDEX });
            }

            // (restrict scope of mutable-borrow of `visible_parent_map`)
            {
                let visible_parent_map = &mut visible_parent_map;
                let mut add_child =
                    |bfs_queue: &mut VecDeque<_>, child: &Export<hir::HirId>, parent: DefId| {
                        if child.vis != ty::Visibility::Public {
                            return;
                        }

                        if let Some(child) = child.res.opt_def_id() {
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
                        }
                    };

                while let Some(def) = bfs_queue.pop_front() {
                    for child in tcx.item_children(def).iter() {
                        add_child(bfs_queue, child, def);
                    }
                }
            }

            visible_parent_map
        },

        dependency_formats: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            Lrc::new(crate::dependency_format::calculate(tcx))
        },
        has_global_allocator: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            CStore::from_tcx(tcx).has_global_allocator()
        },
        postorder_cnums: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            tcx.arena.alloc_slice(&CStore::from_tcx(tcx).crate_dependencies_in_postorder(cnum))
        },

        ..*providers
    };
}

impl CStore {
    pub fn struct_field_names_untracked(&self, def: DefId, sess: &Session) -> Vec<Spanned<Symbol>> {
        self.get_crate_data(def.krate).get_struct_field_names(def.index, sess)
    }

    pub fn item_children_untracked(
        &self,
        def_id: DefId,
        sess: &Session,
    ) -> Vec<Export<hir::HirId>> {
        let mut result = vec![];
        self.get_crate_data(def_id.krate).each_child_of_item(
            def_id.index,
            |child| result.push(child),
            sess,
        );
        result
    }

    pub fn load_macro_untracked(&self, id: DefId, sess: &Session) -> LoadedMacro {
        let _prof_timer = sess.prof.generic_activity("metadata_load_macro");

        let data = self.get_crate_data(id.krate);
        if data.root.is_proc_macro_crate() {
            return LoadedMacro::ProcMacro(data.load_proc_macro(id.index, sess));
        }

        let span = data.get_span(id.index, sess);

        let attrs = data.get_item_attrs(id.index, sess).collect();

        let ident = data.item_ident(id.index, sess);

        LoadedMacro::MacroDef(
            ast::Item {
                ident,
                id: ast::DUMMY_NODE_ID,
                span,
                attrs,
                kind: ast::ItemKind::MacroDef(data.get_macro(id.index, sess)),
                vis: ast::Visibility {
                    span: span.shrink_to_lo(),
                    kind: ast::VisibilityKind::Inherited,
                    tokens: None,
                },
                tokens: None,
            },
            data.root.edition,
        )
    }

    pub fn associated_item_cloned_untracked(&self, def: DefId, sess: &Session) -> ty::AssocItem {
        self.get_crate_data(def.krate).get_associated_item(def.index, sess)
    }

    pub fn crate_source_untracked(&self, cnum: CrateNum) -> CrateSource {
        self.get_crate_data(cnum).source.clone()
    }

    pub fn get_span_untracked(&self, def_id: DefId, sess: &Session) -> Span {
        self.get_crate_data(def_id.krate).get_span(def_id.index, sess)
    }

    pub fn item_generics_num_lifetimes(&self, def_id: DefId, sess: &Session) -> usize {
        self.get_crate_data(def_id.krate).get_generics(def_id.index, sess).own_counts().lifetimes
    }

    pub fn module_expansion_untracked(&self, def_id: DefId, sess: &Session) -> ExpnId {
        self.get_crate_data(def_id.krate).module_expansion(def_id.index, sess)
    }
}

impl CrateStore for CStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn crate_name_untracked(&self, cnum: CrateNum) -> Symbol {
        self.get_crate_data(cnum).root.name
    }

    fn crate_is_private_dep_untracked(&self, cnum: CrateNum) -> bool {
        self.get_crate_data(cnum).private_dep
    }

    fn crate_disambiguator_untracked(&self, cnum: CrateNum) -> CrateDisambiguator {
        self.get_crate_data(cnum).root.disambiguator
    }

    fn crate_hash_untracked(&self, cnum: CrateNum) -> Svh {
        self.get_crate_data(cnum).root.hash
    }

    /// Returns the `DefKey` for a given `DefId`. This indicates the
    /// parent `DefId` as well as some idea of what kind of data the
    /// `DefId` refers to.
    fn def_key(&self, def: DefId) -> DefKey {
        self.get_crate_data(def.krate).def_key(def.index)
    }

    fn def_kind(&self, def: DefId) -> DefKind {
        self.get_crate_data(def.krate).def_kind(def.index)
    }

    fn def_path(&self, def: DefId) -> DefPath {
        self.get_crate_data(def.krate).def_path(def.index)
    }

    fn def_path_hash(&self, def: DefId) -> DefPathHash {
        self.get_crate_data(def.krate).def_path_hash(def.index)
    }

    fn all_def_path_hashes_and_def_ids(&self, cnum: CrateNum) -> Vec<(DefPathHash, DefId)> {
        self.get_crate_data(cnum).all_def_path_hashes_and_def_ids()
    }

    fn num_def_ids(&self, cnum: CrateNum) -> usize {
        self.get_crate_data(cnum).num_def_ids()
    }

    // See `CrateMetadataRef::def_path_hash_to_def_id` for more details
    fn def_path_hash_to_def_id(
        &self,
        cnum: CrateNum,
        index_guess: u32,
        hash: DefPathHash,
    ) -> Option<DefId> {
        self.get_crate_data(cnum).def_path_hash_to_def_id(cnum, index_guess, hash)
    }

    fn crates_untracked(&self) -> Vec<CrateNum> {
        let mut result = vec![];
        self.iter_crate_data(|cnum, _| result.push(cnum));
        result
    }

    fn encode_metadata(&self, tcx: TyCtxt<'_>) -> EncodedMetadata {
        encoder::encode_metadata(tcx)
    }

    fn metadata_encoding_version(&self) -> &[u8] {
        rmeta::METADATA_HEADER
    }

    fn allocator_kind(&self) -> Option<AllocatorKind> {
        self.allocator_kind()
    }
}
