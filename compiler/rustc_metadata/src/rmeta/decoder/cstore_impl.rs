use std::any::Any;
use std::mem;
use std::sync::Arc;

use rustc_attr_parsing::Deprecation;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::bug;
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::exported_symbols::ExportedSymbol;
use rustc_middle::middle::stability::DeprecationEntry;
use rustc_middle::query::{ExternProviders, LocalCrate};
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::util::Providers;
use rustc_session::cstore::{CrateStore, ExternCrate};
use rustc_session::{Session, StableCrateId};
use rustc_span::hygiene::ExpnId;
use rustc_span::{Span, Symbol, kw};

use super::{Decodable, DecodeContext, DecodeIterator};
use crate::creader::{CStore, LoadedMacro};
use crate::rmeta::AttrFlags;
use crate::rmeta::table::IsDefault;
use crate::{foreign_modules, native_libs};

trait ProcessQueryValue<'tcx, T> {
    fn process_decoded(self, _tcx: TyCtxt<'tcx>, _err: impl Fn() -> !) -> T;
}

impl<T> ProcessQueryValue<'_, T> for T {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, _err: impl Fn() -> !) -> T {
        self
    }
}

impl<'tcx, T> ProcessQueryValue<'tcx, ty::EarlyBinder<'tcx, T>> for T {
    #[inline(always)]
    fn process_decoded(self, _tcx: TyCtxt<'_>, _err: impl Fn() -> !) -> ty::EarlyBinder<'tcx, T> {
        ty::EarlyBinder::bind(self)
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
    fn process_decoded(self, tcx: TyCtxt<'tcx>, err: impl Fn() -> !) -> &'tcx [T] {
        if let Some(iter) = self { tcx.arena.alloc_from_iter(iter) } else { err() }
    }
}

impl<'a, 'tcx, T: Copy + Decodable<DecodeContext<'a, 'tcx>>>
    ProcessQueryValue<'tcx, ty::EarlyBinder<'tcx, &'tcx [T]>>
    for Option<DecodeIterator<'a, 'tcx, T>>
{
    #[inline(always)]
    fn process_decoded(
        self,
        tcx: TyCtxt<'tcx>,
        err: impl Fn() -> !,
    ) -> ty::EarlyBinder<'tcx, &'tcx [T]> {
        ty::EarlyBinder::bind(if let Some(iter) = self {
            tcx.arena.alloc_from_iter(iter)
        } else {
            err()
        })
    }
}

impl<'a, 'tcx, T: Copy + Decodable<DecodeContext<'a, 'tcx>>>
    ProcessQueryValue<'tcx, Option<&'tcx [T]>> for Option<DecodeIterator<'a, 'tcx, T>>
{
    #[inline(always)]
    fn process_decoded(self, tcx: TyCtxt<'tcx>, _err: impl Fn() -> !) -> Option<&'tcx [T]> {
        if let Some(iter) = self { Some(&*tcx.arena.alloc_from_iter(iter)) } else { None }
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
                let value = if lazy.is_default() {
                    &[] as &[_]
                } else {
                    $tcx.arena.alloc_from_iter(lazy.decode(($cdata, $tcx)))
                };
                value.process_decoded($tcx, || panic!("{:?} does not have a {:?}", $def_id, stringify!($name)))
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
            def_id_arg: rustc_middle::query::queries::$name::Key<'tcx>,
        ) -> rustc_middle::query::queries::$name::ProvidedValue<'tcx> {
            let _prof_timer =
                $tcx.prof.generic_activity(concat!("metadata_decode_entry_", stringify!($name)));

            #[allow(unused_variables)]
            let ($def_id, $other) = def_id_arg.into_args();
            assert!(!$def_id.is_local());

            // External query providers call `crate_hash` in order to register a dependency
            // on the crate metadata. The exception is `crate_hash` itself, which obviously
            // doesn't need to do this (and can't, as it would cause a query cycle).
            use rustc_middle::dep_graph::dep_kinds;
            if dep_kinds::$name != dep_kinds::crate_hash && $tcx.dep_graph.is_fully_enabled() {
                $tcx.ensure_ok().crate_hash($def_id.krate);
            }

            let cdata = rustc_data_structures::sync::FreezeReadGuard::map(CStore::from_tcx($tcx), |c| {
                c.get_crate_data($def_id.krate).cdata
            });
            let $cdata = crate::creader::CrateMetadataRef {
                cdata: &cdata,
                cstore: &CStore::from_tcx($tcx),
            };

            $compute
        }
    };
}

macro_rules! provide {
    ($tcx:ident, $def_id:ident, $other:ident, $cdata:ident,
      $($name:ident => { $($compute:tt)* })*) => {
        fn provide_extern(providers: &mut ExternProviders) {
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

impl<'tcx> IntoArgs for ty::InstanceKind<'tcx> {
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
    explicit_item_self_bounds => { table_defaulted_array }
    explicit_predicates_of => { table }
    generics_of => { table }
    inferred_outlives_of => { table_defaulted_array }
    explicit_super_predicates_of => { table_defaulted_array }
    explicit_implied_predicates_of => { table_defaulted_array }
    type_of => { table }
    type_alias_is_lazy => { table_direct }
    variances_of => { table }
    fn_sig => { table }
    codegen_fn_attrs => { table }
    impl_trait_header => { table }
    const_param_default => { table }
    object_lifetime_default => { table }
    thir_abstract_const => { table }
    optimized_mir => { table }
    mir_for_ctfe => { table }
    closure_saved_names_of_captured_variables => { table }
    mir_coroutine_witnesses => { table }
    promoted_mir => { table }
    def_span => { table }
    def_ident_span => { table }
    lookup_stability => { table }
    lookup_const_stability => { table }
    lookup_default_body_stability => { table }
    lookup_deprecation_entry => { table }
    params_in_repr => { table }
    def_kind => { cdata.def_kind(def_id.index) }
    impl_parent => { table }
    defaultness => { table_direct }
    constness => { table_direct }
    const_conditions => { table }
    explicit_implied_const_bounds => { table_defaulted_array }
    coerce_unsized_info => {
        Ok(cdata
            .root
            .tables
            .coerce_unsized_info
            .get(cdata, def_id.index)
            .map(|lazy| lazy.decode((cdata, tcx)))
            .process_decoded(tcx, || panic!("{def_id:?} does not have coerce_unsized_info"))) }
    mir_const_qualif => { table }
    rendered_const => { table }
    rendered_precise_capturing_args => { table }
    asyncness => { table_direct }
    fn_arg_names => { table }
    coroutine_kind => { table_direct }
    coroutine_for_closure => { table }
    coroutine_by_move_body_def_id => { table }
    eval_static_initializer => {
        Ok(cdata
            .root
            .tables
            .eval_static_initializer
            .get(cdata, def_id.index)
            .map(|lazy| lazy.decode((cdata, tcx)))
            .unwrap_or_else(|| panic!("{def_id:?} does not have eval_static_initializer")))
    }
    trait_def => { table }
    deduced_param_attrs => {
        // FIXME: `deduced_param_attrs` has some sketchy encoding settings,
        // where we don't encode unless we're optimizing, doing codegen,
        // and not incremental (see `encoder.rs`). I don't think this is right!
        cdata
            .root
            .tables
            .deduced_param_attrs
            .get(cdata, def_id.index)
            .map(|lazy| {
                &*tcx.arena.alloc_from_iter(lazy.decode((cdata, tcx)))
            })
            .unwrap_or_default()
    }
    opaque_ty_origin => { table }
    assumed_wf_types_for_rpitit => { table }
    collect_return_position_impl_trait_in_trait_tys => {
        Ok(cdata
            .root
            .tables
            .trait_impl_trait_tys
            .get(cdata, def_id.index)
            .map(|lazy| lazy.decode((cdata, tcx)))
            .process_decoded(tcx, || panic!("{def_id:?} does not have trait_impl_trait_tys")))
    }

    associated_types_for_impl_traits_in_associated_fn => { table_defaulted_array }

    visibility => { cdata.get_visibility(def_id.index) }
    adt_def => { cdata.get_adt_def(def_id.index, tcx) }
    adt_destructor => { table }
    adt_async_destructor => { table }
    associated_item_def_ids => {
        tcx.arena.alloc_from_iter(cdata.get_associated_item_or_field_def_ids(def_id.index))
    }
    associated_item => { cdata.get_associated_item(def_id.index, tcx.sess) }
    inherent_impls => { cdata.get_inherent_implementations_for_type(tcx, def_id.index) }
    attrs_for_def => { tcx.arena.alloc_from_iter(cdata.get_item_attrs(def_id.index, tcx.sess)) }
    is_mir_available => { cdata.is_item_mir_available(def_id.index) }
    is_ctfe_mir_available => { cdata.is_ctfe_mir_available(def_id.index) }
    cross_crate_inlinable => { table_direct }

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
    extern_crate => { cdata.extern_crate.map(|c| &*tcx.arena.alloc(c)) }
    is_no_builtins => { cdata.root.no_builtins }
    symbol_mangling_version => { cdata.root.symbol_mangling_version }
    specialization_enabled_in => { cdata.root.specialization_enabled_in }
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
    crate_hash => { cdata.root.header.hash }
    crate_host_hash => { cdata.host_hash }
    crate_name => { cdata.root.header.name }
    num_extern_def_ids => { cdata.num_def_ids() }

    extra_filename => { cdata.root.extra_filename.clone() }

    traits => { tcx.arena.alloc_from_iter(cdata.get_traits()) }
    trait_impls_in_crate => { tcx.arena.alloc_from_iter(cdata.get_trait_impls()) }
    implementations_of_trait => { cdata.get_implementations_of_trait(tcx, other) }
    crate_incoherent_impls => { cdata.get_incoherent_impls(tcx, other) }

    dep_kind => { cdata.dep_kind }
    module_children => {
        tcx.arena.alloc_from_iter(cdata.get_module_children(def_id.index, tcx.sess))
    }
    lib_features => { cdata.get_lib_features() }
    stability_implications => {
        cdata.get_stability_implications(tcx).iter().copied().collect()
    }
    stripped_cfg_items => { cdata.get_stripped_cfg_items(cdata.cnum, tcx) }
    intrinsic_raw => { cdata.get_intrinsic(def_id.index) }
    defined_lang_items => { cdata.get_lang_items(tcx) }
    diagnostic_items => { cdata.get_diagnostic_items() }
    missing_lang_items => { cdata.get_missing_lang_items(tcx) }

    missing_extern_crate_item => {
        matches!(cdata.extern_crate, Some(extern_crate) if !extern_crate.is_direct())
    }

    used_crate_source => { Arc::clone(&cdata.source) }
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
    is_doc_hidden => { cdata.get_attr_flags(def_id.index).contains(AttrFlags::IS_DOC_HIDDEN) }
    doc_link_resolutions => { tcx.arena.alloc(cdata.get_doc_link_resolutions(def_id.index)) }
    doc_link_traits_in_scope => {
        tcx.arena.alloc_from_iter(cdata.get_doc_link_traits_in_scope(def_id.index))
    }
}

pub(in crate::rmeta) fn provide(providers: &mut Providers) {
    provide_cstore_hooks(providers);
    providers.queries = rustc_middle::query::Providers {
        allocator_kind: |tcx, ()| CStore::from_tcx(tcx).allocator_kind(),
        alloc_error_handler_kind: |tcx, ()| CStore::from_tcx(tcx).alloc_error_handler_kind(),
        is_private_dep: |_tcx, LocalCrate| false,
        native_library: |tcx, id| {
            tcx.native_libraries(id.krate)
                .iter()
                .filter(|lib| native_libs::relevant_lib(tcx.sess, lib))
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
        native_libraries: native_libs::collect,
        foreign_modules: foreign_modules::collect,

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

        dependency_formats: |tcx, ()| Arc::new(crate::dependency_format::calculate(tcx)),
        has_global_allocator: |tcx, LocalCrate| CStore::from_tcx(tcx).has_global_allocator(),
        has_alloc_error_handler: |tcx, LocalCrate| CStore::from_tcx(tcx).has_alloc_error_handler(),
        postorder_cnums: |tcx, ()| {
            tcx.arena
                .alloc_slice(&CStore::from_tcx(tcx).crate_dependencies_in_postorder(LOCAL_CRATE))
        },
        crates: |tcx, ()| {
            // The list of loaded crates is now frozen in query cache,
            // so make sure cstore is not mutably accessed from here on.
            tcx.untracked().cstore.freeze();
            tcx.arena.alloc_from_iter(CStore::from_tcx(tcx).iter_crate_data().map(|(cnum, _)| cnum))
        },
        used_crates: |tcx, ()| {
            // The list of loaded crates is now frozen in query cache,
            // so make sure cstore is not mutably accessed from here on.
            tcx.untracked().cstore.freeze();
            tcx.arena.alloc_from_iter(
                CStore::from_tcx(tcx)
                    .iter_crate_data()
                    .filter_map(|(cnum, data)| data.used().then_some(cnum)),
            )
        },
        ..providers.queries
    };
    provide_extern(&mut providers.extern_queries);
}

impl CStore {
    pub fn ctor_untracked(&self, def: DefId) -> Option<(CtorKind, DefId)> {
        self.get_crate_data(def.krate).get_ctor(def.index)
    }

    pub fn load_macro_untracked(&self, id: DefId, tcx: TyCtxt<'_>) -> LoadedMacro {
        let sess = tcx.sess;
        let _prof_timer = sess.prof.generic_activity("metadata_load_macro");

        let data = self.get_crate_data(id.krate);
        if data.root.is_proc_macro_crate() {
            LoadedMacro::ProcMacro(data.load_proc_macro(id.index, tcx))
        } else {
            LoadedMacro::MacroDef {
                def: data.get_macro(id.index, sess),
                ident: data.item_ident(id.index, sess),
                attrs: data.get_item_attrs(id.index, sess).collect(),
                span: data.get_span(id.index, sess),
                edition: data.root.edition,
            }
        }
    }

    pub fn def_span_untracked(&self, def_id: DefId, sess: &Session) -> Span {
        self.get_crate_data(def_id.krate).get_span(def_id.index, sess)
    }

    pub fn def_kind_untracked(&self, def: DefId) -> DefKind {
        self.get_crate_data(def.krate).def_kind(def.index)
    }

    pub fn expn_that_defined_untracked(&self, def_id: DefId, sess: &Session) -> ExpnId {
        self.get_crate_data(def_id.krate).get_expn_that_defined(def_id.index, sess)
    }

    /// Only public-facing way to traverse all the definitions in a non-local crate.
    /// Critically useful for this third-party project: <https://github.com/hacspec/hacspec>.
    /// See <https://github.com/rust-lang/rust/pull/85889> for context.
    pub fn num_def_ids_untracked(&self, cnum: CrateNum) -> usize {
        self.get_crate_data(cnum).num_def_ids()
    }

    pub fn get_proc_macro_quoted_span_untracked(
        &self,
        cnum: CrateNum,
        id: usize,
        sess: &Session,
    ) -> Span {
        self.get_crate_data(cnum).get_proc_macro_quoted_span(id, sess)
    }

    pub fn set_used_recursively(&mut self, cnum: CrateNum) {
        let cmeta = self.get_crate_data_mut(cnum);
        if !cmeta.used {
            cmeta.used = true;
            let dependencies = mem::take(&mut cmeta.dependencies);
            for &dep_cnum in &dependencies {
                self.set_used_recursively(dep_cnum);
            }
            self.get_crate_data_mut(cnum).dependencies = dependencies;
        }
    }

    pub(crate) fn update_extern_crate(&mut self, cnum: CrateNum, extern_crate: ExternCrate) {
        let cmeta = self.get_crate_data_mut(cnum);
        if cmeta.update_extern_crate(extern_crate) {
            // Propagate the extern crate info to dependencies if it was updated.
            let extern_crate = ExternCrate { dependency_of: cnum, ..extern_crate };
            let dependencies = mem::take(&mut cmeta.dependencies);
            for &dep_cnum in &dependencies {
                self.update_extern_crate(dep_cnum, extern_crate);
            }
            self.get_crate_data_mut(cnum).dependencies = dependencies;
        }
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
        self.get_crate_data(cnum).root.header.name
    }

    fn stable_crate_id(&self, cnum: CrateNum) -> StableCrateId {
        self.get_crate_data(cnum).root.stable_crate_id
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
}

fn provide_cstore_hooks(providers: &mut Providers) {
    providers.hooks.def_path_hash_to_def_id_extern = |tcx, hash, stable_crate_id| {
        // If this is a DefPathHash from an upstream crate, let the CrateStore map
        // it to a DefId.
        let cstore = CStore::from_tcx(tcx);
        let cnum = *tcx
            .untracked()
            .stable_crate_ids
            .read()
            .get(&stable_crate_id)
            .unwrap_or_else(|| bug!("uninterned StableCrateId: {stable_crate_id:?}"));
        assert_ne!(cnum, LOCAL_CRATE);
        let def_index = cstore.get_crate_data(cnum).def_path_hash_to_def_index(hash);
        DefId { krate: cnum, index: def_index }
    };

    providers.hooks.expn_hash_to_expn_id = |tcx, cnum, index_guess, hash| {
        let cstore = CStore::from_tcx(tcx);
        cstore.get_crate_data(cnum).expn_hash_to_expn_id(tcx.sess, index_guess, hash)
    };
    providers.hooks.import_source_files = |tcx, cnum| {
        let cstore = CStore::from_tcx(tcx);
        let cdata = cstore.get_crate_data(cnum);
        for file_index in 0..cdata.root.source_map.size() {
            cdata.imported_source_file(file_index as u32, tcx.sess);
        }
    };
}
