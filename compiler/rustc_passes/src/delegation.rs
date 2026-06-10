use rustc_data_structures::fx::FxIndexMap;
use rustc_macros::Diagnostic;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

pub fn check_glob_and_list_delegations_target_expr(tcx: TyCtxt<'_>) {
    let mut delegations_by_group_id = FxIndexMap::default();

    for &id in tcx.resolutions(()).delegation_infos.keys() {
        if let Some(info) = tcx.hir_opt_delegation_info(id)
            && let Some((group_id, unused_target_expr)) = info.group_id
        {
            delegations_by_group_id
                .entry(group_id)
                .or_insert_with(|| (true, tcx.def_span(id)))
                .0 &= unused_target_expr;
        }
    }

    for (_, (unused_target_expr, span)) in delegations_by_group_id {
        if unused_target_expr {
            tcx.dcx().emit_err(DelegationTargetExprDeletedEverywhere { span });
        }
    }
}

#[derive(Diagnostic)]
#[diag("unused target expression is specified for glob or list delegation")]
struct DelegationTargetExprDeletedEverywhere {
    #[primary_span]
    pub span: Span,
}
