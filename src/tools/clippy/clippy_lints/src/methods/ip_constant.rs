use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::sym;
use smallvec::SmallVec;

use super::IP_CONSTANT;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, func: &Expr<'_>, args: &[Expr<'_>]) {
    if let ExprKind::Path(QPath::TypeRelative(ty, p)) = func.kind
        && let TyKind::Path(QPath::Resolved(_, func_path)) = ty.kind
        && p.ident.name == sym::new
        && let Some(func_def_id) = func_path.res.opt_def_id()
        && matches!(
            cx.tcx.get_diagnostic_name(func_def_id),
            Some(sym::Ipv4Addr | sym::Ipv6Addr)
        )
        && let Some(args) = args
            .iter()
            .map(|arg| {
                if let Some(Constant::Int(constant @ (0 | 1 | 127 | 255))) = ConstEvalCtxt::new(cx).eval(arg) {
                    u8::try_from(constant).ok()
                } else {
                    None
                }
            })
            .collect::<Option<SmallVec<[u8; 8]>>>()
    {
        let constant_name = match args.as_slice() {
            [0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0, 0] => "UNSPECIFIED",
            [127, 0, 0, 1] | [0, 0, 0, 0, 0, 0, 0, 1] => "LOCALHOST",
            [255, 255, 255, 255] => "BROADCAST",
            _ => return,
        };

        let mut sugg = vec![(expr.span.with_lo(p.ident.span.lo()), constant_name.to_owned())];
        let before_span = expr.span.shrink_to_lo().until(ty.span);
        if !before_span.is_empty() {
            // Remove everything before the type name
            sugg.push((before_span, String::new()));
        }

        span_lint_and_then(cx, IP_CONSTANT, expr.span, "hand-coded well-known IP address", |diag| {
            diag.multipart_suggestion_verbose("use", sugg, Applicability::MachineApplicable);
        });
    }
}
