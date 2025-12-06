use rustc_errors::MultiSpan;
use rustc_hir as hir;
use rustc_middle::span_bug;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::sym;

use crate::lints::Issue145738Diag;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// A crater only lint that checks for multiple parameters capturing a single const/const ctor
    /// inside a single `format_args!` macro expansion.
    pub ISSUE_145739,
    Deny,
    "a violation for issue 145739 is caught",
}

declare_lint_pass!(
    Issue145739 => [ISSUE_145739]
);

impl<'tcx> LateLintPass<'tcx> for Issue145739 {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx hir::Stmt<'tcx>) {
        let hir::StmtKind::Let(hir::LetStmt {
            pat:
                hir::Pat { kind: hir::PatKind::Binding(hir::BindingMode::NONE, _, ident, None), .. },
            init: Some(hir::Expr { kind: hir::ExprKind::Tup(tup), .. }),
            ..
        }) = stmt.kind
        else {
            return;
        };

        if ident.name != sym::__issue_145739 {
            return;
        }

        if tup.len() < 3 {
            span_bug!(stmt.span, "Duplicated tuple is too short");
        }

        let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg_expr) =
            tup[0].kind
        else {
            span_bug!(stmt.span, "Duplicated tuple first element is not a ref");
        };

        let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = arg_expr.kind else {
            return;
        };

        let hir::def::Res::Def(def_kind, def_id) = path.res else {
            return;
        };

        let tcx = cx.tcx;
        let typing_env = cx.typing_env();
        let (ty, kind) = match def_kind {
            hir::def::DefKind::Const => (tcx.type_of(def_id).skip_binder(), "constant"),
            hir::def::DefKind::Ctor(_, hir::def::CtorKind::Const) => {
                (tcx.type_of(tcx.parent(def_id)).skip_binder(), "constant constructor")
            }
            _ => {
                return;
            }
        };

        let lint = |is_not_freeze, needs_drop| {
            let diag = Issue145738Diag {
                const_span: tcx.def_span(def_id),
                kind,
                ty,
                is_not_freeze,
                needs_drop,
                duplicates: MultiSpan::from_spans(tup.iter().skip(1).map(|e| e.span).collect()),
            };
            cx.emit_span_lint(ISSUE_145739, stmt.span, diag);
        };

        if !ty.is_freeze(tcx, typing_env) {
            lint(true, ty.needs_drop(tcx, typing_env));
        } else if ty.needs_drop(tcx, typing_env) {
            lint(false, true);
        }
    }
}
