use crate::rustc_middle::ty::DefIdTree;
use rustc_hir::{def::DefKind, def_id::DefId};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { assumed_wf_types, ..*providers };
}

fn assumed_wf_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> ty::EarlyBinder<ty::Binder<'tcx, &'tcx ty::List<Ty<'tcx>>>> {
    match tcx.def_kind(def_id) {
        DefKind::Fn => ty::EarlyBinder(from_fn_sig(tcx, def_id)),
        DefKind::AssocFn => {
            let from_sig = from_fn_sig(tcx, def_id);
            let from_impl = tcx.assumed_wf_types(tcx.parent(def_id)).subst_identity();

            let assumed_wf_types = from_impl.no_bound_vars().unwrap().into_iter();
            let assumed_wf_types = assumed_wf_types.chain(from_sig.skip_binder());
            ty::EarlyBinder(from_sig.rebind(tcx.mk_type_list(assumed_wf_types)))
        }
        DefKind::Impl => {
            let unnormalized = match tcx.impl_trait_ref(def_id) {
                Some(trait_ref) => tcx.mk_type_list(trait_ref.substs.types()),
                // Only the impl self type
                None => tcx.intern_type_list(&[tcx.type_of(def_id)]),
            };

            // FIXME(@lcnr): rustc currently does not check wf for types
            // pre-normalization, meaning that implied bounds from unnormalized types
            // are sometimes incorrect. See #100910 for more details.
            //
            // Not adding the unnormalized types here mostly fixes that, except
            // that there are projections which are still ambiguous in the item definition
            // but do normalize successfully when using the item, see #98543.
            let normalized =
                normalize_ignoring_obligations(tcx, tcx.param_env(def_id), unnormalized)
                    .unwrap_or_else(|_| ty::List::empty());
            ty::EarlyBinder(ty::Binder::dummy(normalized))
        }
        DefKind::AssocConst | DefKind::AssocTy => tcx.assumed_wf_types(tcx.parent(def_id)),
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::TyParam
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static(_)
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::ImplTraitPlaceholder
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::Generator => ty::EarlyBinder(ty::Binder::dummy(ty::List::empty())),
    }
}

fn from_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> ty::Binder<'tcx, &'tcx ty::List<Ty<'tcx>>> {
    // FIXME(#84533): We probably shouldn't use output types in implied bounds.
    // This would reject this fn `fn f<'a, 'b>() -> &'a &'b () { .. }`.
    let unnormalized = tcx.fn_sig(def_id).map_bound(|sig| sig.inputs_and_output);
    let normalized = normalize_ignoring_obligations(tcx, tcx.param_env(def_id), unnormalized)
        .unwrap_or_else(|_| ty::Binder::dummy(ty::List::empty()));

    // FIXME(#105948): Use unnormalized types for implied bounds as well.
    normalized
}

fn normalize_ignoring_obligations<'tcx, T: TypeFoldable<'tcx>>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: T,
) -> Result<T, NoSolution> {
    use rustc_infer::infer::TyCtxtInferExt as _;
    use rustc_infer::traits::ObligationCause;
    use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt as _;

    let infcx = tcx.infer_ctxt().build();
    let normalized = infcx
        .at(&ObligationCause::dummy(), param_env)
        .query_normalize(value)
        .map(|normalized| infcx.resolve_vars_if_possible(normalized.value))?;
    assert!(!normalized.needs_infer());
    Ok(normalized)
}
