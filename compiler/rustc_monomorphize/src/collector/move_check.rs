use rustc_session::lint::builtin::LARGE_ASSIGNMENTS;
use tracing::debug;

use super::*;
use crate::errors::LargeAssignmentsLint;

pub(super) struct MoveCheckState {
    /// Spans for move size lints already emitted. Helps avoid duplicate lints.
    move_size_spans: Vec<Span>,
    /// Set of functions for which it is OK to move large data into.
    skip_move_check_fns: Option<Vec<DefId>>,
}

impl MoveCheckState {
    pub(super) fn new() -> Self {
        MoveCheckState { move_size_spans: vec![], skip_move_check_fns: None }
    }
}

impl<'a, 'tcx> MirUsedCollector<'a, 'tcx> {
    pub(super) fn check_operand_move_size(
        &mut self,
        operand: &mir::Operand<'tcx>,
        location: Location,
    ) {
        let limit = self.tcx.move_size_limit();
        if limit.0 == 0 {
            return;
        }

        // This function is called by visit_operand() which visits _all_
        // operands, including TerminatorKind::Call operands. But if
        // check_fn_args_move_size() has been called, the operands have already
        // been visited. Do not visit them again.
        if self.visiting_call_terminator {
            return;
        }

        let source_info = self.body.source_info(location);
        debug!(?source_info);

        if let Some(too_large_size) = self.operand_size_if_too_large(limit, operand) {
            self.lint_large_assignment(limit.0, too_large_size, location, source_info.span);
        };
    }

    pub(super) fn check_fn_args_move_size(
        &mut self,
        callee_ty: Ty<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        fn_span: Span,
        location: Location,
    ) {
        let limit = self.tcx.move_size_limit();
        if limit.0 == 0 {
            return;
        }

        if args.is_empty() {
            return;
        }

        // Allow large moves into container types that themselves are cheap to move
        let ty::FnDef(def_id, _) = *callee_ty.kind() else {
            return;
        };
        if self
            .move_check
            .skip_move_check_fns
            .get_or_insert_with(|| build_skip_move_check_fns(self.tcx))
            .contains(&def_id)
        {
            return;
        }

        debug!(?def_id, ?fn_span);

        for arg in args {
            // Moving args into functions is typically implemented with pointer
            // passing at the llvm-ir level and not by memcpy's. So always allow
            // moving args into functions.
            let operand: &mir::Operand<'tcx> = &arg.node;
            if let mir::Operand::Move(_) = operand {
                continue;
            }

            if let Some(too_large_size) = self.operand_size_if_too_large(limit, operand) {
                self.lint_large_assignment(limit.0, too_large_size, location, arg.span);
            };
        }
    }

    fn operand_size_if_too_large(
        &mut self,
        limit: Limit,
        operand: &mir::Operand<'tcx>,
    ) -> Option<Size> {
        let ty = operand.ty(self.body, self.tcx);
        let ty = self.monomorphize(ty);
        let Ok(layout) = self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty)) else {
            return None;
        };
        if layout.size.bytes_usize() > limit.0 {
            debug!(?layout);
            Some(layout.size)
        } else {
            None
        }
    }

    fn lint_large_assignment(
        &mut self,
        limit: usize,
        too_large_size: Size,
        location: Location,
        span: Span,
    ) {
        let source_info = self.body.source_info(location);
        debug!(?source_info);
        for reported_span in &self.move_check.move_size_spans {
            if reported_span.overlaps(span) {
                return;
            }
        }
        let lint_root = source_info.scope.lint_root(&self.body.source_scopes);
        debug!(?lint_root);
        let Some(lint_root) = lint_root else {
            // This happens when the issue is in a function from a foreign crate that
            // we monomorphized in the current crate. We can't get a `HirId` for things
            // in other crates.
            // FIXME: Find out where to report the lint on. Maybe simply crate-level lint root
            // but correct span? This would make the lint at least accept crate-level lint attributes.
            return;
        };
        self.tcx.emit_node_span_lint(LARGE_ASSIGNMENTS, lint_root, span, LargeAssignmentsLint {
            span,
            size: too_large_size.bytes(),
            limit: limit as u64,
        });
        self.move_check.move_size_spans.push(span);
    }
}

fn build_skip_move_check_fns(tcx: TyCtxt<'_>) -> Vec<DefId> {
    let fns = [
        (tcx.lang_items().owned_box(), "new"),
        (tcx.get_diagnostic_item(sym::Rc), "new"),
        (tcx.get_diagnostic_item(sym::Arc), "new"),
    ];
    fns.into_iter()
        .filter_map(|(def_id, fn_name)| {
            def_id.and_then(|def_id| assoc_fn_of_type(tcx, def_id, Ident::from_str(fn_name)))
        })
        .collect::<Vec<_>>()
}
