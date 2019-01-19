use dep_graph::SerializedDepNodeIndex;
use dep_graph::DepNode;
use hir::def_id::DefId;
use mir::interpret::GlobalId;
use ty::{self, TyCtxt};
use ty::query::queries;
use ty::query::Query;
use ty::query::QueryCache;
use util::profiling::ProfileCategory;

use std::borrow::Cow;
use std::hash::Hash;
use std::fmt::Debug;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::stable_hasher::HashStable;
use ich::StableHashingContext;

// Query configuration and description traits.

type CowStr = Cow<'static, str>;

pub trait QueryConfig<'tcx> {
    const NAME: &'static str;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value: Clone + for<'a> HashStable<StableHashingContext<'a>>;
}

pub(super) trait QueryAccessors<'tcx>: QueryConfig<'tcx> {
    fn query(key: Self::Key) -> Query<'tcx>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: TyCtxt<'a, 'tcx, '_>) -> &'a Lock<QueryCache<'tcx, Self>>;

    fn to_dep_node(tcx: TyCtxt<'_, 'tcx, '_>, key: &Self::Key) -> DepNode;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: TyCtxt<'_, 'tcx, '_>, key: Self::Key) -> Self::Value;

    fn handle_cycle_error(tcx: TyCtxt<'_, 'tcx, '_>) -> Self::Value;
}

macro_rules! cache_on_disk {
    (|$def_id:pat| $cond:expr) => {
        #[inline]
        fn cache_on_disk($def_id: Self::Key) -> bool { $cond }
    };
}

pub(super) trait QueryDescription<'tcx>: QueryAccessors<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: Self::Key) -> CowStr;

    cache_on_disk!(|_| false);

    fn try_load_from_disk(
        _: TyCtxt<'_, 'tcx, 'tcx>,
        _: SerializedDepNodeIndex
    ) -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<'tcx, M: QueryAccessors<'tcx, Key=DefId>> QueryDescription<'tcx> for M {
    default fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        if !tcx.sess.verbose() {
            format!("processing `{}`", tcx.item_path_str(def_id)).into()
        } else {
            let name = unsafe { ::std::intrinsics::type_name::<M>() };
            format!("processing {:?} with query `{}`", def_id, name).into()
        }
    }
}

macro_rules! impl_uncacheable_query {
    () => {};
    ($query_name:ident, $descr:literal, |$tcx:pat, $key:pat| $arg:expr; $($rest:tt)*) => {
        impl<'tcx> QueryDescription<'tcx> for queries::$query_name<'tcx> {
            fn describe($tcx: TyCtxt<'_, '_, '_>, $key: Self::Key) -> CowStr {
                format!($descr, $arg).into()
            }
        }
        impl_uncacheable_query!($($rest)*);
    };
    ($query_name:ident, $description:expr; $($rest:tt)*) => {
        impl<'tcx> QueryDescription<'tcx> for queries::$query_name<'tcx> {
            fn describe(_tcx: TyCtxt<'_, '_, '_>, _: Self::Key) -> CowStr {
                $description.into()
            }
        }
        impl_uncacheable_query!($($rest)*);
    };
    (bug $query_name:ident; $($rest:tt)*) => {
        impl<'tcx> QueryDescription<'tcx> for queries::$query_name<'tcx> {
            fn describe(_tcx: TyCtxt<'_, '_, '_>, _: Self::Key) -> CowStr {
                bug!(stringify!($query_name))
            }
        }
        impl_uncacheable_query!($($rest)*);
    };
}

impl_uncacheable_query! {
    check_mod_attrs, "checking attributes in {}", |tcx, key| key.describe_as_module(tcx);
    check_mod_unstable_api_usage, "checking for unstable API usage in {}",
        |tcx, key| key.describe_as_module(tcx);
    check_mod_loops, "checking loops in {}", |tcx, key| key.describe_as_module(tcx);
    check_mod_item_types, "checking item types in {}", |tcx, key| key.describe_as_module(tcx);
    check_mod_privacy, "checking privacy in {}", |tcx, key| key.describe_as_module(tcx);
    check_mod_intrinsics, "checking intrinsics in {}", |tcx, key| key.describe_as_module(tcx);
    check_mod_liveness, "checking liveness of variables in {}",
        |tcx, key| key.describe_as_module(tcx);
    collect_mod_item_types, "collecting item types in {}", |tcx, key| key.describe_as_module(tcx);
    normalize_projection_ty, "normalizing `{:?}`", |_, goal| goal;
    implied_outlives_bounds, "computing implied outlives bounds for `{:?}`", |_, goal| goal;
    dropck_outlives, "computing dropck types for `{:?}`", |_, goal| goal;
    normalize_ty_after_erasing_regions, "normalizing `{:?}`", |_, goal| goal;
    evaluate_obligation, "evaluating trait selection obligation `{}`", |_, goal| goal.value.value;
    evaluate_goal, "evaluating trait selection obligation `{}`", |_, goal| goal.value.goal;
    type_op_ascribe_user_type,  "evaluating `type_op_ascribe_user_type` `{:?}`", |_, goal| goal;
    type_op_eq, "evaluating `type_op_eq` `{:?}`", |_, goal| goal;
    type_op_subtype, "evaluating `type_op_subtype` `{:?}`", |_, goal| goal;
    type_op_prove_predicate, "evaluating `type_op_prove_predicate` `{:?}`", |_, goal| goal;
    type_op_normalize_ty, "normalizing `{:?}`", |_, goal| goal;
    type_op_normalize_predicate, "normalizing `{:?}`", |_, goal| goal;
    type_op_normalize_poly_fn_sig, "normalizing `{:?}`", |_, goal| goal;
    type_op_normalize_fn_sig,  "normalizing `{:?}`", |_, goal| goal;
    is_copy_raw, "computing whether `{}` is `Copy`", |_, env| env.value;
    is_sized_raw, "computing whether `{}` is `Sized`", |_, env| env.value;
    is_freeze_raw, "computing whether `{}` is freeze", |_, env| env.value;
    needs_drop_raw, "computing whether `{}` needs drop", |_, env| env.value;
    layout_raw, "computing layout of `{}`", |_, env| env.value;
    super_predicates_of, "computing the supertraits of `{}`",
        |tcx, def_id| tcx.item_path_str(def_id);
    erase_regions_ty, "erasing regions from `{:?}`", |_, ty| ty;
    type_param_predicates, "computing the bounds for type parameter `{}`", |tcx, (_, def_id)| {
        let id = tcx.hir().as_local_node_id(def_id).unwrap();
        tcx.hir().ty_param_name(id)
    };
    coherent_trait, "coherence checking all impls of trait `{}`",
        |tcx, def_id| tcx.item_path_str(def_id);
    upstream_monomorphizations, "collecting available upstream monomorphizations `{:?}`", |_, k| k;
    crate_inherent_impls, "all inherent impls defined in crate `{:?}`", |_, k| k;
    crate_inherent_impls_overlap_check,
        "check for overlap between inherent impls defined in this crate";
    crate_variances, "computing the variances for items in this crate";
    inferred_outlives_crate, "computing the inferred outlives predicates for items in this crate";
    mir_shims, "generating MIR shim for `{}`", |tcx, def| tcx.item_path_str(def.def_id());
    privacy_access_levels, "privacy access levels";
    typeck_item_bodies, "type-checking all item bodies";
    reachable_set, "reachability";
    mir_keys, "getting a list of all mir_keys";
    bug describe_def;
    bug def_span;
    bug lookup_stability;
    bug lookup_deprecation_entry;
    bug item_attrs;
    bug is_reachable_non_generic;
    bug fn_arg_names;
    bug impl_parent;
    bug trait_of_item;
    rvalue_promotable_map, "checking which parts of `{}` are promotable to static",
        |tcx, def_id| tcx.item_path_str(def_id);
    is_mir_available, "checking if item is mir available: `{}`",
        |tcx, def_id| tcx.item_path_str(def_id);
    trait_impls_of, "trait impls of `{}`", |tcx, def_id| tcx.item_path_str(def_id);
    is_object_safe, "determine object safety of trait `{}`",
        |tcx, def_id| tcx.item_path_str(def_id);
    is_const_fn_raw, "checking if item is const fn: `{}`", |tcx, def_id| tcx.item_path_str(def_id);
    dylib_dependency_formats, "dylib dependency formats of crate";
    is_panic_runtime, "checking if the crate is_panic_runtime";
    is_compiler_builtins, "checking if the crate is_compiler_builtins";
    has_global_allocator, "checking if the crate has_global_allocator";
    has_panic_handler, "checking if the crate has_panic_handler";
    extern_crate, "getting crate's ExternCrateData";
    lint_levels, "computing the lint levels for items in this crate";
    specializes, "computing whether impls specialize one another";
    in_scope_traits_map, "traits in scope at a block";
    is_no_builtins, "test whether a crate has #![no_builtins]";
    panic_strategy, "query a crate's configured panic strategy";
    is_profiler_runtime, "query a crate is #![profiler_runtime]";
    is_sanitizer_runtime, "query a crate is #![sanitizer_runtime]";
    reachable_non_generics, "looking up the exported symbols of a crate";
    native_libraries,  "looking up the native libraries of a linked crate";
    foreign_modules, "looking up the foreign modules of a linked crate";
    entry_fn, "looking up the entry function of a crate";
    plugin_registrar_fn, "looking up the plugin registrar for a crate";
    proc_macro_decls_static, "looking up the derive registrar for a crate";
    crate_disambiguator, "looking up the disambiguator a crate";
    crate_hash, "looking up the hash a crate";
    original_crate_name, "looking up the original name a crate";
    extra_filename, "looking up the extra filename for a crate";
    implementations_of_trait, "looking up implementations of a trait in a crate";
    all_trait_implementations, "looking up all (?) trait implementations";
    link_args, "looking up link arguments for a crate";
    resolve_lifetimes, "resolving lifetimes";
    named_region_map, "looking up a named region";
    is_late_bound_map, "testing if a region is late bound";
    object_lifetime_defaults_map, "looking up lifetime defaults for a region";
    dep_kind, "fetching what a dependency looks like";
    crate_name, "fetching what a crate is named";
    get_lib_features, "calculating the lib features map";
    defined_lib_features, "calculating the lib features defined in a crate";
    get_lang_items, "calculating the lang items map";
    defined_lang_items, "calculating the lang items defined in a crate";
    missing_lang_items, "calculating the missing lang items in a crate";
    visible_parent_map, "calculating the visible parent map";
    missing_extern_crate_item, "seeing if we're missing an `extern crate` item for this crate";
    used_crate_source, "looking at the source for a crate";
    postorder_cnums, "generating a postorder list of CrateNums";
    maybe_unused_extern_crates, "looking up all possibly unused extern crates";
    stability_index, "calculating the stability index for the local crate";
    all_traits, "fetching all foreign and local traits";
    all_crate_nums, "fetching all foreign CrateNum instances";
    exported_symbols, "exported_symbols";
    collect_and_partition_mono_items, "collect_and_partition_mono_items";
    codegen_unit, "codegen_unit";
    output_filenames, "output_filenames";
    features_query, "looking up enabled feature gates";
    vtable_methods, "finding all methods for trait {}", |tcx, key| tcx.item_path_str(key.def_id());
    substitute_normalize_and_test_predicates, "testing substituted normalized predicates:`{}`",
        |tcx, key| tcx.item_path_str(key.0);
    method_autoderef_steps, "computing autoderef types for `{:?}`", |_, goal| goal;
    target_features_whitelist, "looking up the whitelist of target features";
    instance_def_size_estimate, "estimating size for `{}`",
        |tcx, def| tcx.item_path_str(def.def_id());
    program_clauses_for, "generating chalk-style clauses";
    program_clauses_for_env, "generating chalk-style clauses for environment";
    environment, "return a chalk-style environment";
    wasm_import_module_map, "wasm import module map";
    dllimport_foreign_items, "wasm import module map";
}

impl<'tcx> QueryDescription<'tcx> for queries::const_eval<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>) -> CowStr {
        format!(
            "const-evaluating + checking `{}`",
            tcx.item_path_str(key.value.instance.def.def_id()),
        ).into()
    }

    cache_on_disk!(|_| true);

    #[inline]
    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id).map(Ok)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::const_eval_raw<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>) -> CowStr {
        format!("const-evaluating `{}`", tcx.item_path_str(key.value.instance.def.def_id())).into()
    }

    cache_on_disk!(|_| true);

    #[inline]
    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id).map(Ok)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::symbol_name<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, instance: ty::Instance<'tcx>) -> CowStr {
        format!("computing the symbol for `{}`", instance).into()
    }

    cache_on_disk!(|_| true);

    #[inline]
    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::const_is_rvalue_promotable_to_static<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> CowStr {
        format!("const checking if rvalue is promotable to static `{}`",
            tcx.item_path_str(def_id)).into()
    }

    cache_on_disk!(|_| true);

    #[inline]
    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::codegen_fulfill_obligation<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>,
                key: (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)) -> CowStr {
        format!("checking if `{}` fulfills its obligations", tcx.item_path_str(key.1.def_id()))
            .into()
    }

    cache_on_disk!(|_| true);

    #[inline]
    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::typeck_tables_of<'tcx> {
    cache_on_disk!(|def_id| def_id.is_local());

    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache
            .try_load_query_result(tcx, id)
            .map(|tables: ty::TypeckTables<'tcx>| tcx.alloc_tables(tables))
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::optimized_mir<'tcx> {
    cache_on_disk!(|def_id| def_id.is_local());

    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache
            .try_load_query_result(tcx, id)
            .map(|x: ::mir::Mir<'tcx>| tcx.alloc_mir(x))
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::generics_of<'tcx> {
    cache_on_disk!(|def_id| def_id.is_local());

    fn try_load_from_disk<'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        id: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        tcx.queries.on_disk_cache
            .try_load_query_result::<ty::Generics>(tcx, id)
            .map(|x| tcx.alloc_generics(x))
    }
}

macro_rules! impl_disk_cacheable_query(
    ($($query_name:ident, |$key:pat| $cond:expr;)+) => { $(
        impl<'tcx> QueryDescription<'tcx> for queries::$query_name<'tcx> {
            cache_on_disk!(|$key| $cond);

            #[inline]
            fn try_load_from_disk<'a>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                id: SerializedDepNodeIndex,
            ) -> Option<Self::Value> {
                tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
            }
        }
    )+ }
);

impl_disk_cacheable_query! {
    unsafety_check_result, |def_id| def_id.is_local();
    borrowck, |def_id| def_id.is_local();
    mir_borrowck, |def_id| def_id.is_local();
    mir_const_qualif, |def_id| def_id.is_local();
    check_match, |def_id| def_id.is_local();
    def_symbol_name, |_| true;
    type_of, |def_id| def_id.is_local();
    predicates_of, |def_id| def_id.is_local();
    used_trait_imports, |def_id| def_id.is_local();
    codegen_fn_attrs, |_| true;
    specialization_graph_of, |_| true;
}
