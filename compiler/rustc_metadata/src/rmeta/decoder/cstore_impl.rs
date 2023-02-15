use crate::creader::{CStore, LoadedMacro};
use crate::foreign_modules;
use crate::native_libs;
use crate::rmeta::table::IsDefault;
use crate::rmeta::AttrFlags;

use rustc_ast as ast;
use rustc_attr::Deprecation;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::exported_symbols::ExportedSymbol;
use rustc_middle::middle::stability::DeprecationEntry;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::query::{ExternProviders, Providers};
use rustc_middle::ty::{self, TyCtxt, Visibility};
use rustc_session::cstore::{CrateSource, CrateStore};
use rustc_session::{Session, StableCrateId};
use rustc_span::hygiene::{ExpnHash, ExpnId};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::symbol::{kw, Symbol};

use rustc_data_structures::sync::Lrc;
use std::any::Any;

use super::{Decodable, DecodeContext, DecodeIterator};

trait ProcessQueryValue<'tcx, T> {
    fn process_decoded(self, _tcx: TyCtxt<'tcx>, _err: impl Fn() -> !) -> T;
}

impl<T> ProcessQueryValue<'_, Option<T>> for Option<T> {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, _err: impl Fn() -> !) -> Option<T> {
        self
    }
}

impl<T> ProcessQueryValue<'_, T> for Option<T> {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, err: impl Fn() -> !) -> T {
        if let Some(value) = self { value } else { err() }
    }
}

impl<'tcx, T: ArenaAllocatable<'tcx>> ProcessQueryValue<'tcx, &'tcx T> for Option<T> {
    #[inline(always)]
    fn process_decoded(self, tcx: TyCtxt<'tcx>, err: impl Fn() -> !) -> &'tcx T {
        if let Some(value) = self { tcx.arena.alloc(value) } else { err() }
    }
}

impl<T, E> ProcessQueryValue<'_, Result<Option<T>, E>> for Option<T> {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, _err: impl Fn() -> !) -> Result<Option<T>, E> {
        Ok(self)
    }
}

impl<'a, 'tcx, T: Copy + Decodable<DecodeContext<'a, 'tcx>>> ProcessQueryValue<'tcx, &'tcx [T]>
    for Option<DecodeIterator<'a, 'tcx, T>>
{
    #[inline(always)]
    fn process_decoded(self, tcx: TyCtxt<'tcx>, _err: impl Fn() -> !) -> &'tcx [T] {
        if let Some(iter) = self { tcx.arena.alloc_from_iter(iter) } else { &[] }
    }
}

impl ProcessQueryValue<'_, Option<DeprecationEntry>> for Option<Deprecation> {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, _err: impl Fn() -> !) -> Option<DeprecationEntry> {
        self.map(DeprecationEntry::external)
    }
}

macro_rules! provide_one {
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident, $name:ident => { table }) => {
        provide_one! {
            $tcx, $def_id, $other, $cdata, $name => {
                $cdata
                    .root
                    .tables
                    .$name
                    .get($cdata, $def_id.index)
                    .map(|lazy| lazy.decode(($cdata, $tcx)))
                    .process_decoded($tcx, || panic!("{:?} does not have a {:?}", $def_id, stringify!($name)))
            }
        }
    };
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident, $name:ident => { table_defaulted_array }) => {
        provide_one! {
            $tcx, $def_id, $other, $cdata, $name => {
                let lazy = $cdata.root.tables.$name.get($cdata, $def_id.index);
                if lazy.is_default() { &[] } else { $tcx.arena.alloc_from_iter(lazy.decode(($cdata, $tcx))) }
            }
        }
    };
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident, $name:ident => { table_direct }) => {
        provide_one! {
            $tcx, $def_id, $other, $cdata, $name => {
                // We don't decode `table_direct`, since it's not a Lazy, but an actual value
                $cdata
                    .root
                    .tables
                    .$name
                    .get($cdata, $def_id.index)
                    .process_decoded($tcx, || panic!("{:?} does not have a {:?}", $def_id, stringify!($name)))
            }
        }
    };
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident, $name:ident => $compute:block) => {
        fn $name<'tcx>(
            $tcx: TyCtxt<'tcx>,
            def_id_arg: ty::query::query_keys::$name<'tcx>,
        ) -> ty::query::query_values::$name<'tcx> {
            let _prof_timer =
                $tcx.prof.generic_activity(concat!("metadata_decode_entry_", stringify!($name)));

            #[allow(unused_variables)]
            let ($def_id, $other) = def_id_arg.into_args();
            assert!(!$def_id.is_local());

            // External query providers call `crate_hash` in order to register a dependency
            // on the crate metadata. The exception is `crate_hash` itself, which obviously
            // doesn't need to do this (and can't, as it would cause a query cycle).
            use rustc_middle::dep_graph::DepKind;
            if DepKind::$name != DepKind::crate_hash && $tcx.dep_graph.is_fully_enabled() {
                $tcx.ensure().crate_hash($def_id.krate);
            }

            let $cdata = CStore::from_tcx($tcx).get_crate_data($def_id.krate);

            $compute
        }
    };
}

macro_rules! provide {
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident,
      $($name:ident => { $($compute:tt)* })*) => {
        pub fn provide_extern(providers: &mut ExternProviders) {
            $(provide_one! {
                $tcx, $def_id, $other, $cdata, $name => { $($compute)* }
            })*

            *providers = ExternProviders {
                $($name,)*
                ..*providers
            };
        }
    }
}

// small trait to work around different signature queries all being defined via
// the macro above.
trait IntoArgs {
    type Other;
    fn into_args(self) -> (DefId, Self::Other);
}

impl IntoArgs for DefId {
    type Other = ();
    fn into_args(self) -> (DefId, ()) {
        (self, ())
    }
}

impl IntoArgs for CrateNum {
    type Other = ();
    fn into_args(self) -> (DefId, ()) {
        (self.as_def_id(), ())
    }
}

impl IntoArgs for (CrateNum, DefId) {
    type Other = DefId;
    fn into_args(self) -> (DefId, DefId) {
        (self.0.as_def_id(), self.1)
    }
}

impl<'tcx> IntoArgs for ty::InstanceDef<'tcx> {
    type Other = ();
    fn into_args(self) -> (DefId, ()) {
        (self.def_id(), ())
    }
}

impl IntoArgs for (CrateNum, SimplifiedType) {
    type Other = SimplifiedType;
    fn into_args(self) -> (DefId, SimplifiedType) {
        (self.0.as_def_id(), self.1)
    }
}

provide! { tcx, def_id, other, cdata,
    explicit_item_bounds => { table_defaulted_array }
    explicit_predicates_of => { table }
    generics_of => { table }
    inferred_outlives_of => { table_defaulted_array }
    super_predicates_of => { table }
    type_of => { table }
    variances_of => { table }
    fn_sig => { table }
    codegen_fn_attrs => { table }
    impl_trait_ref => { table }
    const_param_default => { table }
    object_lifetime_default => { table }
    thir_abstract_const => { table }
    optimized_mir => { table }
    mir_for_ctfe => { table }
    mir_generator_witnesses => { table }
    promoted_mir => { table }
    def_span => { table }
    def_ident_span => { table }
    lookup_stability => { table }
    lookup_const_stability => { table }
    lookup_default_body_stability => { table }
    lookup_deprecation_entry => { table }
    params_in_repr => { table }
    unused_generic_params => { table }
    opt_def_kind => { table_direct }
    impl_parent => { table }
    impl_polarity => { table_direct }
    impl_defaultness => { table_direct }
    constness => { table_direct }
    coerce_unsized_info => { table }
    mir_const_qualif => { table }
    rendered_const => { table }
    asyncness => { table_direct }
    fn_arg_names => { table }
    generator_kind => { table }
    trait_def => { table }
    deduced_param_attrs => { table }
    is_type_alias_impl_trait => {
        debug_assert_eq!(tcx.def_kind(def_id), DefKind::OpaqueTy);
        cdata.root.tables.is_type_alias_impl_trait.get(cdata, def_id.index)
    }
    collect_return_position_impl_trait_in_trait_tys => {
        Ok(cdata
            .root
            .tables
            .trait_impl_trait_tys
            .get(cdata, def_id.index)
            .map(|lazy| lazy.decode((cdata, tcx)))
            .process_decoded(tcx, || panic!("{def_id:?} does not have trait_impl_trait_tys")))
     }

    visibility => { cdata.get_visibility(def_id.index) }
    adt_def => { cdata.get_adt_def(def_id.index, tcx) }
    adt_destructor => {
        let _ = cdata;
        tcx.calculate_dtor(def_id, |_,_| Ok(()))
    }
    associated_item_def_ids => {
        tcx.arena.alloc_from_iter(cdata.get_associated_item_def_ids(def_id.index, tcx.sess))
    }
    associated_item => { cdata.get_associated_item(def_id.index, tcx.sess) }
    inherent_impls => { cdata.get_inherent_implementations_for_type(tcx, def_id.index) }
    is_foreign_item => { cdata.is_foreign_item(def_id.index) }
    item_attrs => { tcx.arena.alloc_from_iter(cdata.get_item_attrs(def_id.index, tcx.sess)) }
    is_mir_available => { cdata.is_item_mir_available(def_id.index) }
    is_ctfe_mir_available => { cdata.is_ctfe_mir_available(def_id.index) }

    dylib_dependency_formats => { cdata.get_dylib_dependency_formats(tcx) }
    is_private_dep => { cdata.private_dep }
    is_panic_runtime => { cdata.root.panic_runtime }
    is_compiler_builtins => { cdata.root.compiler_builtins }
    has_global_allocator => { cdata.root.has_global_allocator }
    has_alloc_error_handler => { cdata.root.has_alloc_error_handler }
    has_panic_handler => { cdata.root.has_panic_handler }
    is_profiler_runtime => { cdata.root.profiler_runtime }
    required_panic_strategy => { cdata.root.required_panic_strategy }
    panic_in_drop_strategy => { cdata.root.panic_in_drop_strategy }
    extern_crate => {
        let r = *cdata.extern_crate.lock();
        r.map(|c| &*tcx.arena.alloc(c))
    }
    is_no_builtins => { cdata.root.no_builtins }
    symbol_mangling_version => { cdata.root.symbol_mangling_version }
    reachable_non_generics => {
        let reachable_non_generics = tcx
            .exported_symbols(cdata.cnum)
            .iter()
            .filter_map(|&(exported_symbol, export_info)| {
                if let ExportedSymbol::NonGeneric(def_id) = exported_symbol {
                    Some((def_id, export_info))
                } else {
                    None
                }
            })
            .collect();

        reachable_non_generics
    }
    native_libraries => { cdata.get_native_libraries(tcx.sess).collect() }
    foreign_modules => { cdata.get_foreign_modules(tcx.sess).map(|m| (m.def_id, m)).collect() }
    crate_hash => { cdata.root.hash }
    crate_host_hash => { cdata.host_hash }
    crate_name => { cdata.root.name }

    extra_filename => { cdata.root.extra_filename.clone() }

    traits_in_crate => { tcx.arena.alloc_from_iter(cdata.get_traits()) }
    trait_impls_in_crate => { tcx.arena.alloc_from_iter(cdata.get_trait_impls()) }
    implementations_of_trait => { cdata.get_implementations_of_trait(tcx, other) }
    crate_incoherent_impls => { cdata.get_incoherent_impls(tcx, other) }

    dep_kind => {
        let r = *cdata.dep_kind.lock();
        r
    }
    module_children => {
        tcx.arena.alloc_from_iter(cdata.get_module_children(def_id.index, tcx.sess))
    }
    defined_lib_features => { cdata.get_lib_features(tcx) }
    stability_implications => {
        cdata.get_stability_implications(tcx).iter().copied().collect()
    }
    is_intrinsic => { cdata.get_is_intrinsic(def_id.index) }
    defined_lang_items => { cdata.get_lang_items(tcx) }
    diagnostic_items => { cdata.get_diagnostic_items() }
    missing_lang_items => { cdata.get_missing_lang_items(tcx) }

    missing_extern_crate_item => {
        let r = matches!(*cdata.extern_crate.borrow(), Some(extern_crate) if !extern_crate.is_direct());
        r
    }

    used_crate_source => { Lrc::clone(&cdata.source) }
    debugger_visualizers => { cdata.get_debugger_visualizers() }

    exported_symbols => {
        let syms = cdata.exported_symbols(tcx);

        // FIXME rust-lang/rust#64319, rust-lang/rust#64872: We want
        // to block export of generics from dylibs, but we must fix
        // rust-lang/rust#65890 before we can do that robustly.

        syms
    }

    crate_extern_paths => { cdata.source().paths().cloned().collect() }
    expn_that_defined => { cdata.get_expn_that_defined(def_id.index, tcx.sess) }
    generator_diagnostic_data => { cdata.get_generator_diagnostic_data(tcx, def_id.index) }
    is_doc_hidden => { cdata.get_attr_flags(def_id.index).contains(AttrFlags::IS_DOC_HIDDEN) }
    doc_link_resolutions => { tcx.arena.alloc(cdata.get_doc_link_resolutions(def_id.index)) }
    doc_link_traits_in_scope => {
        tcx.arena.alloc_from_iter(cdata.get_doc_link_traits_in_scope(def_id.index))
    }
}

pub(in crate::rmeta) fn provide(providers: &mut Providers) {
    // FIXME(#44234) - almost all of these queries have no sub-queries and
    // therefore no actual inputs, they're just reading tables calculated in
    // resolve! Does this work? Unsure! That's what the issue is about
    *providers = Providers {
        allocator_kind: |tcx, ()| CStore::from_tcx(tcx).allocator_kind(),
        alloc_error_handler_kind: |tcx, ()| CStore::from_tcx(tcx).alloc_error_handler_kind(),
        is_private_dep: |_tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            false
        },
        native_library: |tcx, id| {
            tcx.native_libraries(id.krate)
                .iter()
                .filter(|lib| native_libs::relevant_lib(&tcx.sess, lib))
                .find(|lib| {
                    let Some(fm_id) = lib.foreign_module else {
                        return false;
                    };
                    let map = tcx.foreign_modules(id.krate);
                    map.get(&fm_id)
                        .expect("failed to find foreign module")
                        .foreign_items
                        .contains(&id)
                })
        },
        native_libraries: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            native_libs::collect(tcx)
        },
        foreign_modules: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            foreign_modules::collect(tcx).into_iter().map(|m| (m.def_id, m)).collect()
        },

        // Returns a map from a sufficiently visible external item (i.e., an
        // external item that is visible from at least one local module) to a
        // sufficiently visible parent (considering modules that re-export the
        // external item to be parents).
        visible_parent_map: |tcx, ()| {
            use std::collections::hash_map::Entry;
            use std::collections::vec_deque::VecDeque;

            let mut visible_parent_map: DefIdMap<DefId> = Default::default();
            // This is a secondary visible_parent_map, storing the DefId of
            // parents that re-export the child as `_` or module parents
            // which are `#[doc(hidden)]`. Since we prefer paths that don't
            // do this, merge this map at the end, only if we're missing
            // keys from the former.
            // This is a rudimentary check that does not catch all cases,
            // just the easiest.
            let mut fallback_map: Vec<(DefId, DefId)> = Default::default();

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

            for &cnum in tcx.crates(()) {
                // Ignore crates without a corresponding local `extern crate` item.
                if tcx.missing_extern_crate_item(cnum) {
                    continue;
                }

                bfs_queue.push_back(cnum.as_def_id());
            }

            let mut add_child = |bfs_queue: &mut VecDeque<_>, child: &ModChild, parent: DefId| {
                if !child.vis.is_public() {
                    return;
                }

                if let Some(def_id) = child.res.opt_def_id() {
                    if child.ident.name == kw::Underscore {
                        fallback_map.push((def_id, parent));
                        return;
                    }

                    if tcx.is_doc_hidden(parent) {
                        fallback_map.push((def_id, parent));
                        return;
                    }

                    match visible_parent_map.entry(def_id) {
                        Entry::Occupied(mut entry) => {
                            // If `child` is defined in crate `cnum`, ensure
                            // that it is mapped to a parent in `cnum`.
                            if def_id.is_local() && entry.get().is_local() {
                                entry.insert(parent);
                            }
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(parent);
                            if matches!(
                                child.res,
                                Res::Def(DefKind::Mod | DefKind::Enum | DefKind::Trait, _)
                            ) {
                                bfs_queue.push_back(def_id);
                            }
                        }
                    }
                }
            };

            while let Some(def) = bfs_queue.pop_front() {
                for child in tcx.module_children(def).iter() {
                    add_child(bfs_queue, child, def);
                }
            }

            // Fill in any missing entries with the less preferable path.
            // If this path re-exports the child as `_`, we still use this
            // path in a diagnostic that suggests importing `::*`.

            for (child, parent) in fallback_map {
                visible_parent_map.entry(child).or_insert(parent);
            }

            visible_parent_map
        },

        dependency_formats: |tcx, ()| Lrc::new(crate::dependency_format::calculate(tcx)),
        has_global_allocator: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            CStore::from_tcx(tcx).has_global_allocator()
        },
        has_alloc_error_handler: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            CStore::from_tcx(tcx).has_alloc_error_handler()
        },
        postorder_cnums: |tcx, ()| {
            tcx.arena
                .alloc_slice(&CStore::from_tcx(tcx).crate_dependencies_in_postorder(LOCAL_CRATE))
        },
        crates: |tcx, ()| tcx.arena.alloc_from_iter(CStore::from_tcx(tcx).crates_untracked()),
        ..*providers
    };
}

impl CStore {
    pub fn struct_field_names_untracked<'a>(
        &'a self,
        def: DefId,
        sess: &'a Session,
    ) -> impl Iterator<Item = Spanned<Symbol>> + 'a {
        self.get_crate_data(def.krate).get_struct_field_names(def.index, sess)
    }

    pub fn struct_field_visibilities_untracked(
        &self,
        def: DefId,
    ) -> impl Iterator<Item = Visibility<DefId>> + '_ {
        self.get_crate_data(def.krate).get_struct_field_visibilities(def.index)
    }

    pub fn ctor_untracked(&self, def: DefId) -> Option<(CtorKind, DefId)> {
        self.get_crate_data(def.krate).get_ctor(def.index)
    }

    pub fn visibility_untracked(&self, def: DefId) -> Visibility<DefId> {
        self.get_crate_data(def.krate).get_visibility(def.index)
    }

    pub fn module_children_untracked<'a>(
        &'a self,
        def_id: DefId,
        sess: &'a Session,
    ) -> impl Iterator<Item = ModChild> + 'a {
        self.get_crate_data(def_id.krate).get_module_children(def_id.index, sess)
    }

    pub fn load_macro_untracked(&self, id: DefId, sess: &Session) -> LoadedMacro {
        let _prof_timer = sess.prof.generic_activity("metadata_load_macro");

        let data = self.get_crate_data(id.krate);
        if data.root.is_proc_macro_crate() {
            return LoadedMacro::ProcMacro(data.load_proc_macro(id.index, sess));
        }

        let span = data.get_span(id.index, sess);

        LoadedMacro::MacroDef(
            ast::Item {
                ident: data.item_ident(id.index, sess),
                id: ast::DUMMY_NODE_ID,
                span,
                attrs: data.get_item_attrs(id.index, sess).collect(),
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

    pub fn fn_has_self_parameter_untracked(&self, def: DefId, sess: &Session) -> bool {
        self.get_crate_data(def.krate).get_fn_has_self_parameter(def.index, sess)
    }

    pub fn crate_source_untracked(&self, cnum: CrateNum) -> Lrc<CrateSource> {
        self.get_crate_data(cnum).source.clone()
    }

    pub fn get_span_untracked(&self, def_id: DefId, sess: &Session) -> Span {
        self.get_crate_data(def_id.krate).get_span(def_id.index, sess)
    }

    pub fn def_kind(&self, def: DefId) -> DefKind {
        self.get_crate_data(def.krate).def_kind(def.index)
    }

    pub fn crates_untracked(&self) -> impl Iterator<Item = CrateNum> + '_ {
        self.iter_crate_data().map(|(cnum, _)| cnum)
    }

    pub fn item_generics_num_lifetimes(&self, def_id: DefId, sess: &Session) -> usize {
        self.get_crate_data(def_id.krate).get_generics(def_id.index, sess).own_counts().lifetimes
    }

    pub fn module_expansion_untracked(&self, def_id: DefId, sess: &Session) -> ExpnId {
        self.get_crate_data(def_id.krate).module_expansion(def_id.index, sess)
    }

    /// Only public-facing way to traverse all the definitions in a non-local crate.
    /// Critically useful for this third-party project: <https://github.com/hacspec/hacspec>.
    /// See <https://github.com/rust-lang/rust/pull/85889> for context.
    pub fn num_def_ids_untracked(&self, cnum: CrateNum) -> usize {
        self.get_crate_data(cnum).num_def_ids()
    }

    pub fn item_attrs_untracked<'a>(
        &'a self,
        def_id: DefId,
        sess: &'a Session,
    ) -> impl Iterator<Item = ast::Attribute> + 'a {
        self.get_crate_data(def_id.krate).get_item_attrs(def_id.index, sess)
    }

    pub fn get_proc_macro_quoted_span_untracked(
        &self,
        cnum: CrateNum,
        id: usize,
        sess: &Session,
    ) -> Span {
        self.get_crate_data(cnum).get_proc_macro_quoted_span(id, sess)
    }
}

impl CrateStore for CStore {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn untracked_as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn crate_name(&self, cnum: CrateNum) -> Symbol {
        self.get_crate_data(cnum).root.name
    }

    fn stable_crate_id(&self, cnum: CrateNum) -> StableCrateId {
        self.get_crate_data(cnum).root.stable_crate_id
    }

    fn stable_crate_id_to_crate_num(&self, stable_crate_id: StableCrateId) -> CrateNum {
        self.stable_crate_ids[&stable_crate_id]
    }

    /// Returns the `DefKey` for a given `DefId`. This indicates the
    /// parent `DefId` as well as some idea of what kind of data the
    /// `DefId` refers to.
    fn def_key(&self, def: DefId) -> DefKey {
        self.get_crate_data(def.krate).def_key(def.index)
    }

    fn def_path(&self, def: DefId) -> DefPath {
        self.get_crate_data(def.krate).def_path(def.index)
    }

    fn def_path_hash(&self, def: DefId) -> DefPathHash {
        self.get_crate_data(def.krate).def_path_hash(def.index)
    }

    fn def_path_hash_to_def_id(&self, cnum: CrateNum, hash: DefPathHash) -> DefId {
        let def_index = self.get_crate_data(cnum).def_path_hash_to_def_index(hash);
        DefId { krate: cnum, index: def_index }
    }

    fn expn_hash_to_expn_id(
        &self,
        sess: &Session,
        cnum: CrateNum,
        index_guess: u32,
        hash: ExpnHash,
    ) -> ExpnId {
        self.get_crate_data(cnum).expn_hash_to_expn_id(sess, index_guess, hash)
    }

    fn import_source_files(&self, sess: &Session, cnum: CrateNum) {
        let cdata = self.get_crate_data(cnum);
        for file_index in 0..cdata.root.source_map.size() {
            cdata.imported_source_file(file_index as u32, sess);
        }
    }
}
