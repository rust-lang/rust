use crate::dep_graph::SerializedDepNodeIndex;
use crate::dep_graph::DepNode;
use crate::hir::def_id::{CrateNum, DefId, DefIndex};
use crate::mir::interpret::GlobalId;
use crate::traits;
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal, CanonicalTyGoal,
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpEqGoal, CanonicalTypeOpNormalizeGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpSubtypeGoal,
};
use crate::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use crate::ty::subst::SubstsRef;
use crate::ty::query::queries;
use crate::ty::query::Query;
use crate::ty::query::QueryCache;
use crate::ty::query::plumbing::CycleError;
use crate::util::profiling::ProfileCategory;

use std::borrow::Cow;
use std::hash::Hash;
use std::fmt::Debug;
use syntax_pos::symbol::InternedString;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::fingerprint::Fingerprint;
use crate::ich::StableHashingContext;

// Query configuration and description traits.

pub trait QueryConfig<'tcx> {
    const NAME: &'static str;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value: Clone;
}

pub(crate) trait QueryAccessors<'tcx>: QueryConfig<'tcx> {
    fn query(key: Self::Key) -> Query<'tcx>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: TyCtxt<'a, 'tcx, '_>) -> &'a Lock<QueryCache<'tcx, Self>>;

    fn to_dep_node(tcx: TyCtxt<'_, 'tcx, '_>, key: &Self::Key) -> DepNode;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: TyCtxt<'_, 'tcx, '_>, key: Self::Key) -> Self::Value;

    fn hash_result(
        hcx: &mut StableHashingContext<'_>,
        result: &Self::Value
    ) -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: TyCtxt<'_, 'tcx, '_>, error: CycleError<'tcx>) -> Self::Value;
}

pub(crate) trait QueryDescription<'tcx>: QueryAccessors<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: Self::Key) -> Cow<'static, str>;

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _: Self::Key) -> bool {
        false
    }

    fn try_load_from_disk(_: TyCtxt<'_, 'tcx, 'tcx>,
                          _: SerializedDepNodeIndex)
                          -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<'tcx, M: QueryAccessors<'tcx, Key=DefId>> QueryDescription<'tcx> for M {
    default fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        if !tcx.sess.verbose() {
            format!("processing `{}`", tcx.def_path_str(def_id)).into()
        } else {
            let name = unsafe { ::std::intrinsics::type_name::<M>() };
            format!("processing {:?} with query `{}`", def_id, name).into()
        }
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_attrs<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking attributes in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_unstable_api_usage<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking for unstable API usage in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_loops<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking loops in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_item_types<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking item types in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_privacy<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking privacy in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_intrinsics<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking intrinsics in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_liveness<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking liveness of variables in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_mod_impl_wf<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("checking that impls are well-formed in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::collect_mod_item_types<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: DefId,
    ) -> Cow<'static, str> {
        format!("collecting item types in {}", key.describe_as_module(tcx)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::normalize_projection_ty<'tcx> {
    fn describe(
        _tcx: TyCtxt<'_, '_, '_>,
        goal: CanonicalProjectionGoal<'tcx>,
    ) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::implied_outlives_bounds<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTyGoal<'tcx>) -> Cow<'static, str> {
        format!("computing implied outlives bounds for `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::dropck_outlives<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTyGoal<'tcx>) -> Cow<'static, str> {
        format!("computing dropck types for `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::normalize_ty_after_erasing_regions<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: ParamEnvAnd<'tcx, Ty<'tcx>>) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::evaluate_obligation<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalPredicateGoal<'tcx>) -> Cow<'static, str> {
        format!("evaluating trait selection obligation `{}`", goal.value.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::evaluate_goal<'tcx> {
    fn describe(
        _tcx: TyCtxt<'_, '_, '_>,
        goal: traits::ChalkCanonicalGoal<'tcx>
    ) -> Cow<'static, str> {
        format!("evaluating trait selection obligation `{}`", goal.value.goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_ascribe_user_type<'tcx> {
    fn describe(
        _tcx: TyCtxt<'_, '_, '_>,
        goal: CanonicalTypeOpAscribeUserTypeGoal<'tcx>,
    ) -> Cow<'static, str> {
        format!("evaluating `type_op_ascribe_user_type` `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_eq<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTypeOpEqGoal<'tcx>) -> Cow<'static, str> {
        format!("evaluating `type_op_eq` `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_subtype<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTypeOpSubtypeGoal<'tcx>)
                -> Cow<'static, str> {
        format!("evaluating `type_op_subtype` `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_prove_predicate<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTypeOpProvePredicateGoal<'tcx>)
                -> Cow<'static, str> {
        format!("evaluating `type_op_prove_predicate` `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_normalize_ty<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>,
                goal: CanonicalTypeOpNormalizeGoal<'tcx, Ty<'tcx>>) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_normalize_predicate<'tcx> {
    fn describe(
        _tcx: TyCtxt<'_, '_, '_>,
        goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::Predicate<'tcx>>,
    ) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_normalize_poly_fn_sig<'tcx> {
    fn describe(
        _tcx: TyCtxt<'_, '_, '_>,
        goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::PolyFnSig<'tcx>>,
    ) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_op_normalize_fn_sig<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>,
                goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::FnSig<'tcx>>) -> Cow<'static, str> {
        format!("normalizing `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_copy_raw<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                -> Cow<'static, str> {
        format!("computing whether `{}` is `Copy`", env.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_sized_raw<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                -> Cow<'static, str> {
        format!("computing whether `{}` is `Sized`", env.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_freeze_raw<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                -> Cow<'static, str> {
        format!("computing whether `{}` is freeze", env.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::needs_drop_raw<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                -> Cow<'static, str> {
        format!("computing whether `{}` needs drop", env.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::layout_raw<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                -> Cow<'static, str> {
        format!("computing layout of `{}`", env.value).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::super_predicates_of<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("computing the supertraits of `{}`",
                tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::erase_regions_ty<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, ty: Ty<'tcx>) -> Cow<'static, str> {
        format!("erasing regions from `{:?}`", ty).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::type_param_predicates<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, (_, def_id): (DefId, DefId)) -> Cow<'static, str> {
        let id = tcx.hir().as_local_hir_id(def_id).unwrap();
        format!("computing the bounds for type parameter `{}`",
                tcx.hir().ty_param_name(id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::coherent_trait<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("coherence checking all impls of trait `{}`",
                tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::upstream_monomorphizations<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, k: CrateNum) -> Cow<'static, str> {
        format!("collecting available upstream monomorphizations `{:?}`", k).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_inherent_impls<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, k: CrateNum) -> Cow<'static, str> {
        format!("all inherent impls defined in crate `{:?}`", k).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_inherent_impls_overlap_check<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "check for overlap between inherent impls defined in this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_variances<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "computing the variances for items in this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::inferred_outlives_crate<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "computing the inferred outlives predicates for items in this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::mir_shims<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def: ty::InstanceDef<'tcx>) -> Cow<'static, str> {
        format!("generating MIR shim for `{}`",
                tcx.def_path_str(def.def_id())).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::privacy_access_levels<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "privacy access levels".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::check_private_in_public<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "checking for private elements in public interfaces".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::typeck_item_bodies<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "type-checking all item bodies".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::reachable_set<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "reachability".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::const_eval<'tcx> {
    fn describe(
        tcx: TyCtxt<'_, '_, '_>,
        key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
    ) -> Cow<'static, str> {
        format!(
            "const-evaluating + checking `{}`",
            tcx.def_path_str(key.value.instance.def.def_id()),
        ).into()
    }

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _key: Self::Key) -> bool {
        true
    }

    #[inline]
    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id).map(Ok)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::const_eval_raw<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
        -> Cow<'static, str>
    {
        format!("const-evaluating `{}`", tcx.def_path_str(key.value.instance.def.def_id())).into()
    }

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _key: Self::Key) -> bool {
        true
    }

    #[inline]
    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id).map(Ok)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::mir_keys<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "getting a list of all mir_keys".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::symbol_name<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, instance: ty::Instance<'tcx>) -> Cow<'static, str> {
        format!("computing the symbol for `{}`", instance).into()
    }

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _: Self::Key) -> bool {
        true
    }

    #[inline]
    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::describe_def<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("describe_def")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::def_span<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("def_span")
    }
}


impl<'tcx> QueryDescription<'tcx> for queries::lookup_stability<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("stability")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::lookup_deprecation_entry<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("deprecation")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::item_attrs<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("item_attrs")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_reachable_non_generic<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("is_reachable_non_generic")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::fn_arg_names<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("fn_arg_names")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::impl_parent<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("impl_parent")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::trait_of_item<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        bug!("trait_of_item")
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::const_is_rvalue_promotable_to_static<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("const checking if rvalue is promotable to static `{}`",
            tcx.def_path_str(def_id)).into()
    }

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _: Self::Key) -> bool {
        true
    }

    #[inline]
    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::rvalue_promotable_map<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("checking which parts of `{}` are promotable to static",
                tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_mir_available<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("checking if item is mir available: `{}`",
                tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::codegen_fulfill_obligation<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>,
                key: (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)) -> Cow<'static, str> {
        format!("checking if `{}` fulfills its obligations", tcx.def_path_str(key.1.def_id()))
            .into()
    }

    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, _: Self::Key) -> bool {
        true
    }

    #[inline]
    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::trait_impls_of<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("trait impls of `{}`", tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_object_safe<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("determine object safety of trait `{}`", tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_const_fn_raw<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Cow<'static, str> {
        format!("checking if item is const fn: `{}`", tcx.def_path_str(def_id)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::dylib_dependency_formats<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "dylib dependency formats of crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_compiler_builtins<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "checking if the crate is_compiler_builtins".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::has_global_allocator<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "checking if the crate has_global_allocator".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::has_panic_handler<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "checking if the crate has_panic_handler".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::extern_crate<'tcx> {
    fn describe(_: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        "getting crate's ExternCrateData".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::analysis<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "running analysis passes on this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::lint_levels<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "computing the lint levels for items in this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::specializes<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: (DefId, DefId)) -> Cow<'static, str> {
        "computing whether impls specialize one another".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::in_scope_traits_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefIndex) -> Cow<'static, str> {
        "traits in scope at a block".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_no_builtins<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "test whether a crate has #![no_builtins]".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::panic_strategy<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "query a crate's configured panic strategy".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_profiler_runtime<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "query a crate is #![profiler_runtime]".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_sanitizer_runtime<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "query a crate is #![sanitizer_runtime]".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::reachable_non_generics<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the exported symbols of a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::foreign_modules<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the foreign modules of a linked crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::entry_fn<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the entry function of a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::plugin_registrar_fn<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the plugin registrar for a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::proc_macro_decls_static<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the derive registrar for a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_disambiguator<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the disambiguator a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_hash<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the hash a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::original_crate_name<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the original name a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::extra_filename<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the extra filename for a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::implementations_of_trait<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: (CrateNum, DefId)) -> Cow<'static, str> {
        "looking up implementations of a trait in a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::all_trait_implementations<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up all (?) trait implementations".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::link_args<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up link arguments for a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::resolve_lifetimes<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "resolving lifetimes".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::named_region_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefIndex) -> Cow<'static, str> {
        "looking up a named region".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::is_late_bound_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefIndex) -> Cow<'static, str> {
        "testing if a region is late bound".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::object_lifetime_defaults_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefIndex) -> Cow<'static, str> {
        "looking up lifetime defaults for a region".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::dep_kind<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "fetching what a dependency looks like".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::crate_name<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "fetching what a crate is named".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::get_lib_features<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the lib features map".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::defined_lib_features<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the lib features defined in a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::get_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the lang items map".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::defined_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the lang items defined in a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::missing_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the missing lang items in a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::visible_parent_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the visible parent map".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::missing_extern_crate_item<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "seeing if we're missing an `extern crate` item for this crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::used_crate_source<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking at the source for a crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::postorder_cnums<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "generating a postorder list of CrateNums".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::maybe_unused_extern_crates<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up all possibly unused extern crates".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::stability_index<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "calculating the stability index for the local crate".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::all_traits<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "fetching all foreign and local traits".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::all_crate_nums<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "fetching all foreign CrateNum instances".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::exported_symbols<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "exported_symbols".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::collect_and_partition_mono_items<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "collect_and_partition_mono_items".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::codegen_unit<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: InternedString) -> Cow<'static, str> {
        "codegen_unit".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::output_filenames<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "output_filenames".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::vtable_methods<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: ty::PolyTraitRef<'tcx> ) -> Cow<'static, str> {
        format!("finding all methods for trait {}", tcx.def_path_str(key.def_id())).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::features_query<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up enabled feature gates".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::typeck_tables_of<'tcx> {
    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, def_id: Self::Key) -> bool {
        def_id.is_local()
    }

    fn try_load_from_disk(tcx: TyCtxt<'_, 'tcx, 'tcx>,
                          id: SerializedDepNodeIndex)
                          -> Option<Self::Value> {
        let typeck_tables: Option<ty::TypeckTables<'tcx>> = tcx
            .queries.on_disk_cache
            .try_load_query_result(tcx, id);

        typeck_tables.map(|tables| tcx.alloc_tables(tables))
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::optimized_mir<'tcx> {
    #[inline]
    fn cache_on_disk(_: TyCtxt<'_, 'tcx, 'tcx>, def_id: Self::Key) -> bool {
        def_id.is_local()
    }

    fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: SerializedDepNodeIndex)
                              -> Option<Self::Value> {
        let mir: Option<crate::mir::Mir<'tcx>> = tcx.queries.on_disk_cache
                                               .try_load_query_result(tcx, id);
        mir.map(|x| tcx.alloc_mir(x))
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::substitute_normalize_and_test_predicates<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, key: (DefId, SubstsRef<'tcx>)) -> Cow<'static, str> {
        format!("testing substituted normalized predicates:`{}`", tcx.def_path_str(key.0)).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::method_autoderef_steps<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, goal: CanonicalTyGoal<'tcx>) -> Cow<'static, str> {
        format!("computing autoderef types for `{:?}`", goal).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::target_features_whitelist<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "looking up the whitelist of target features".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::instance_def_size_estimate<'tcx> {
    fn describe(tcx: TyCtxt<'_, '_, '_>, def: ty::InstanceDef<'tcx>) -> Cow<'static, str> {
        format!("estimating size for `{}`", tcx.def_path_str(def.def_id())).into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::program_clauses_for<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        "generating chalk-style clauses".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::program_clauses_for_env<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: traits::Environment<'tcx>) -> Cow<'static, str> {
        "generating chalk-style clauses for environment".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::environment<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: DefId) -> Cow<'static, str> {
        "return a chalk-style environment".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::wasm_import_module_map<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "wasm import module map".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::dllimport_foreign_items<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "wasm import module map".into()
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::backend_optimization_level<'tcx> {
    fn describe(_tcx: TyCtxt<'_, '_, '_>, _: CrateNum) -> Cow<'static, str> {
        "optimization level used by backend".into()
    }
}

macro_rules! impl_disk_cacheable_query(
    ($query_name:ident, |$tcx:tt, $key:tt| $cond:expr) => {
        impl<'tcx> QueryDescription<'tcx> for queries::$query_name<'tcx> {
            #[inline]
            fn cache_on_disk($tcx: TyCtxt<'_, 'tcx, 'tcx>, $key: Self::Key) -> bool {
                $cond
            }

            #[inline]
            fn try_load_from_disk<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      id: SerializedDepNodeIndex)
                                      -> Option<Self::Value> {
                tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
            }
        }
    }
);

impl_disk_cacheable_query!(mir_borrowck, |tcx, def_id| {
    def_id.is_local() && tcx.is_closure(def_id)
});

impl_disk_cacheable_query!(unsafety_check_result, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(borrowck, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(mir_const_qualif, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(check_match, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(def_symbol_name, |_, _| true);
impl_disk_cacheable_query!(predicates_of, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(used_trait_imports, |_, def_id| def_id.is_local());
impl_disk_cacheable_query!(codegen_fn_attrs, |_, _| true);
impl_disk_cacheable_query!(specialization_graph_of, |_, _| true);
