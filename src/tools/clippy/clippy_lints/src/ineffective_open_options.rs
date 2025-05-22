use crate::methods::method_call;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{peel_blocks, sym};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks if both `.write(true)` and `.append(true)` methods are called
    /// on a same `OpenOptions`.
    ///
    /// ### Why is this bad?
    /// `.append(true)` already enables `write(true)`, making this one
    /// superfluous.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::fs::OpenOptions;
    /// let _ = OpenOptions::new()
    ///            .write(true)
    ///            .append(true)
    ///            .create(true)
    ///            .open("file.json");
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::fs::OpenOptions;
    /// let _ = OpenOptions::new()
    ///            .append(true)
    ///            .create(true)
    ///            .open("file.json");
    /// ```
    #[clippy::version = "1.76.0"]
    pub INEFFECTIVE_OPEN_OPTIONS,
    suspicious,
    "usage of both `write(true)` and `append(true)` on same `OpenOptions`"
}

declare_lint_pass!(IneffectiveOpenOptions => [INEFFECTIVE_OPEN_OPTIONS]);

fn index_if_arg_is_boolean(args: &[Expr<'_>], call_span: Span) -> Option<Span> {
    if let [arg] = args
        && let ExprKind::Lit(lit) = peel_blocks(arg).kind
        && lit.node == LitKind::Bool(true)
    {
        // The `.` is not included in the span so we cheat a little bit to include it as well.
        Some(call_span.with_lo(call_span.lo() - BytePos(1)))
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for IneffectiveOpenOptions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some((sym::open, mut receiver, [_arg], _, _)) = method_call(expr) else {
            return;
        };
        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        match receiver_ty.peel_refs().kind() {
            ty::Adt(adt, _) if cx.tcx.is_diagnostic_item(sym::FsOpenOptions, adt.did()) => {},
            _ => return,
        }

        let mut append = None;
        let mut write = None;

        while let Some((name, recv, args, _, span)) = method_call(receiver) {
            if name == sym::append {
                append = index_if_arg_is_boolean(args, span);
            } else if name == sym::write {
                write = index_if_arg_is_boolean(args, span);
            }
            receiver = recv;
        }

        if let Some(write_span) = write
            && append.is_some()
        {
            span_lint_and_sugg(
                cx,
                INEFFECTIVE_OPEN_OPTIONS,
                write_span,
                "unnecessary use of `.write(true)` because there is `.append(true)`",
                "remove `.write(true)`",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }
}
