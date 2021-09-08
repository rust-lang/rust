use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

pub(super) fn derefs_to_slice<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    fn may_slice<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> bool {
        match ty.kind() {
            ty::Slice(_) => true,
            ty::Adt(def, _) if def.is_box() => may_slice(cx, ty.boxed_ty()),
            ty::Adt(..) => is_type_diagnostic_item(cx, ty, sym::vec_type),
            ty::Array(_, size) => size.try_eval_usize(cx.tcx, cx.param_env).is_some(),
            ty::Ref(_, inner, _) => may_slice(cx, inner),
            _ => false,
        }
    }

    if let hir::ExprKind::MethodCall(path, _, [self_arg, ..], _) = &expr.kind {
        if path.ident.name == sym::iter && may_slice(cx, cx.typeck_results().expr_ty(self_arg)) {
            Some(self_arg)
        } else {
            None
        }
    } else {
        match ty.kind() {
            ty::Slice(_) => Some(expr),
            ty::Adt(def, _) if def.is_box() && may_slice(cx, ty.boxed_ty()) => Some(expr),
            ty::Ref(_, inner, _) => {
                if may_slice(cx, inner) {
                    Some(expr)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}

pub(super) fn get_hint_if_single_char_arg(
    cx: &LateContext<'_>,
    arg: &hir::Expr<'_>,
    applicability: &mut Applicability,
) -> Option<String> {
    if_chain! {
        if let hir::ExprKind::Lit(lit) = &arg.kind;
        if let ast::LitKind::Str(r, style) = lit.node;
        let string = r.as_str();
        if string.chars().count() == 1;
        then {
            let snip = snippet_with_applicability(cx, arg.span, &string, applicability);
            let ch = if let ast::StrStyle::Raw(nhash) = style {
                let nhash = nhash as usize;
                // for raw string: r##"a"##
                &snip[(nhash + 2)..(snip.len() - 1 - nhash)]
            } else {
                // for regular string: "a"
                &snip[1..(snip.len() - 1)]
            };
            let hint = format!("'{}'", if ch == "'" { "\\'" } else { ch });
            Some(hint)
        } else {
            None
        }
    }
}
