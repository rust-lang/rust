use rustc_hir::def::DefKind;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNREACHABLE_CODE;

use crate::diagnostics::UnreachableDueToUninhabited;

/// Lint unreachable code due to uninhabited values from function calls,
/// and remove return edges from those calls.
pub(super) struct LintAndRemoveUninhabited;

impl<'tcx> crate::MirPass<'tcx> for LintAndRemoveUninhabited {
    #[tracing::instrument(level = "debug", skip_all)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id().expect_local();
        tracing::debug!(?def_id);
        let parent_module = tcx.parent_module_from_def_id(def_id).to_def_id();
        let typing_env = body.typing_env(tcx);

        // check if the function's return type is inhabited
        // this was added here because of this regression
        // https://github.com/rust-lang/rust/issues/149571
        let return_ty_is_inhabited = matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
            && body.local_decls[RETURN_PLACE].ty.is_inhabited_from(tcx, parent_module, typing_env);

        let mut lints = vec![];
        for bbdata in body.basic_blocks.as_mut() {
            let term = bbdata.terminator_mut();
            let TerminatorKind::Call { ref mut target, destination, .. } = term.kind else {
                continue;
            };
            let Some(target_bb) = *target else { continue };

            let ty = destination.ty(&body.local_decls, tcx).ty;
            let ty_is_inhabited = ty.is_inhabited_from(tcx, parent_module, typing_env);
            if !ty_is_inhabited {
                // Unreachable code warnings are already emitted during type checking.
                // However, during type checking, full type information is being
                // calculated but not yet available, so the check for diverging
                // expressions due to uninhabited result types is pretty crude and
                // only checks whether ty.is_never(). Here, we have full type
                // information available and can issue warnings for less obviously
                // uninhabited types (e.g. empty enums). The check above is used so
                // that we do not emit the same warning twice if the uninhabited type
                // is indeed `!`.
                if !ty.is_never() && return_ty_is_inhabited {
                    lints.push((target_bb, ty, term.source_info.span));
                }

                // The presence or absence of a return edge affects control-flow sensitive
                // MIR checks and ultimately whether code is accepted or not. We can only
                // omit the return edge if a return type is visibly uninhabited to a module
                // that makes the call.
                *target = None;
            }
        }

        for (target_bb, orig_ty, orig_span) in lints {
            if orig_span.in_external_macro(tcx.sess.source_map()) {
                continue;
            }

            let Some((target_loc, descr)) = find_unreachable_code_from(target_bb, body) else {
                continue;
            };
            let lint_root = body.source_scopes[target_loc.scope]
                .local_data
                .as_ref()
                .unwrap_crate_local()
                .lint_root;
            tcx.emit_node_span_lint(
                UNREACHABLE_CODE,
                lint_root,
                target_loc.span,
                UnreachableDueToUninhabited {
                    expr: target_loc.span,
                    orig: orig_span,
                    descr,
                    ty: orig_ty,
                },
            );
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Starting at a target unreachable block, find some user code to lint as unreachable
#[tracing::instrument(level = "debug", skip(body), ret)]
fn find_unreachable_code_from<'tcx>(
    bb: BasicBlock,
    body: &Body<'tcx>,
) -> Option<(SourceInfo, &'static str)> {
    let bb = &body.basic_blocks[bb];
    for stmt in &bb.statements {
        match &stmt.kind {
            // Ignore the implicit `()` return place assignment for unit functions/blocks
            StatementKind::Assign((_, Rvalue::Use(Operand::Constant(const_), _)))
                if const_.ty().is_unit() =>
            {
                continue;
            }
            // Ignore return value plumbing. After a call returning a non-`!`
            // uninhabited type, a tail expression can be unreachable while
            // still being needed to satisfy the surrounding return type.
            StatementKind::Assign((place, _)) if place.as_local() == Some(RETURN_PLACE) => {
                continue;
            }
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                continue;
            }
            StatementKind::FakeRead(..) => return Some((stmt.source_info, "definition")),
            _ => return Some((stmt.source_info, "expression")),
        }
    }

    let term = bb.terminator();
    match term.kind {
        // No user code in this bb, and our goto target may be reachable via other paths
        TerminatorKind::Goto { .. } | TerminatorKind::Return => None,
        _ => Some((term.source_info, "expression")),
    }
}
