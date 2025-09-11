use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::has_non_exhaustive_attr;
use clippy_utils::ty::implements_trait_with_env;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, ClauseKind, GenericParamDefKind, ParamEnv, TraitPredicate, Ty, TyCtxt, Upcast};
use rustc_span::{Span, sym};

use super::DERIVE_PARTIAL_EQ_WITHOUT_EQ;

/// Implementation of the `DERIVE_PARTIAL_EQ_WITHOUT_EQ` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, span: Span, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    if let ty::Adt(adt, args) = ty.kind()
        && cx.tcx.visibility(adt.did()).is_public()
        && let Some(eq_trait_def_id) = cx.tcx.get_diagnostic_item(sym::Eq)
        && let Some(def_id) = trait_ref.trait_def_id()
        && cx.tcx.is_diagnostic_item(sym::PartialEq, def_id)
        && !has_non_exhaustive_attr(cx.tcx, *adt)
        && !ty_implements_eq_trait(cx.tcx, ty, eq_trait_def_id)
        && let typing_env = typing_env_for_derived_eq(cx.tcx, adt.did(), eq_trait_def_id)
        && let Some(local_def_id) = adt.did().as_local()
        // If all of our fields implement `Eq`, we can implement `Eq` too
        && adt
            .all_fields()
            .map(|f| f.ty(cx.tcx, args))
            .all(|ty| implements_trait_with_env(cx.tcx, typing_env, ty, eq_trait_def_id, None, &[]))
    {
        span_lint_hir_and_then(
            cx,
            DERIVE_PARTIAL_EQ_WITHOUT_EQ,
            cx.tcx.local_def_id_to_hir_id(local_def_id),
            span.ctxt().outer_expn_data().call_site,
            "you are deriving `PartialEq` and can implement `Eq`",
            |diag| {
                diag.span_suggestion(
                    span.ctxt().outer_expn_data().call_site,
                    "consider deriving `Eq` as well",
                    "PartialEq, Eq",
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn ty_implements_eq_trait<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, eq_trait_id: DefId) -> bool {
    tcx.non_blanket_impls_for_ty(eq_trait_id, ty).next().is_some()
}

/// Creates the `ParamEnv` used for the given type's derived `Eq` impl.
fn typing_env_for_derived_eq(tcx: TyCtxt<'_>, did: DefId, eq_trait_id: DefId) -> ty::TypingEnv<'_> {
    // Initial map from generic index to param def.
    // Vec<(param_def, needs_eq)>
    let mut params = tcx
        .generics_of(did)
        .own_params
        .iter()
        .map(|p| (p, matches!(p.kind, GenericParamDefKind::Type { .. })))
        .collect::<Vec<_>>();

    let ty_predicates = tcx.predicates_of(did).predicates;
    for (p, _) in ty_predicates {
        if let ClauseKind::Trait(p) = p.kind().skip_binder()
            && p.trait_ref.def_id == eq_trait_id
            && let ty::Param(self_ty) = p.trait_ref.self_ty().kind()
        {
            // Flag types which already have an `Eq` bound.
            params[self_ty.index as usize].1 = false;
        }
    }

    let param_env = ParamEnv::new(tcx.mk_clauses_from_iter(ty_predicates.iter().map(|&(p, _)| p).chain(
        params.iter().filter(|&&(_, needs_eq)| needs_eq).map(|&(param, _)| {
            ClauseKind::Trait(TraitPredicate {
                trait_ref: ty::TraitRef::new(tcx, eq_trait_id, [tcx.mk_param_from_def(param)]),
                polarity: ty::PredicatePolarity::Positive,
            })
            .upcast(tcx)
        }),
    )));
    ty::TypingEnv {
        typing_mode: ty::TypingMode::non_body_analysis(),
        param_env,
    }
}
