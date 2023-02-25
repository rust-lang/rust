use crate::lints::MappingToUnit;
use crate::{LateContext, LateLintPass, LintContext};

use rustc_hir::{Expr, ExprKind, HirId, Stmt, StmtKind};
use rustc_middle::{
    query::Key,
    ty::{self, Ty},
};

declare_lint! {
    /// The `map_unit_fn` lint checks for `Iterator::map` receive
    /// a callable that returns `()`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(items: &mut Vec<u8>) {
    ///     items.sort();
    /// }
    ///
    /// fn main() {
    ///     let mut x: Vec<Vec<u8>> = vec![
    ///         vec![0, 2, 1],
    ///         vec![5, 4, 3],
    ///     ];
    ///     x.iter_mut().map(foo);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Mapping to `()` is almost always a mistake.
    pub MAP_UNIT_FN,
    Warn,
    "`Iterator::map` call that discard the iterator's values"
}

declare_lint_pass!(MapUnitFn => [MAP_UNIT_FN]);

impl<'tcx> LateLintPass<'tcx> for MapUnitFn {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'_>) {
        if stmt.span.from_expansion() {
            return;
        }

        if let StmtKind::Semi(expr) = stmt.kind {
            if let ExprKind::MethodCall(path, receiver, args, span) = expr.kind {
                if path.ident.name.as_str() == "map" {
                    if receiver.span.from_expansion()
                        || args.iter().any(|e| e.span.from_expansion())
                        || !is_impl_slice(cx, receiver)
                        || !is_diagnostic_name(cx, expr.hir_id, "IteratorMap")
                    {
                        return;
                    }
                    let arg_ty = cx.typeck_results().expr_ty(&args[0]);
                    if let ty::FnDef(id, _) = arg_ty.kind() {
                        let fn_ty = cx.tcx.fn_sig(id).skip_binder();
                        let ret_ty = fn_ty.output().skip_binder();
                        if is_unit_type(ret_ty) {
                            cx.emit_spanned_lint(
                                MAP_UNIT_FN,
                                span,
                                MappingToUnit {
                                    function_label: cx.tcx.span_of_impl(*id).unwrap(),
                                    argument_label: args[0].span,
                                    map_label: arg_ty.default_span(cx.tcx),
                                    suggestion: path.ident.span,
                                    replace: "for_each".to_string(),
                                },
                            )
                        }
                    } else if let ty::Closure(id, subs) = arg_ty.kind() {
                        let cl_ty = subs.as_closure().sig();
                        let ret_ty = cl_ty.output().skip_binder();
                        if is_unit_type(ret_ty) {
                            cx.emit_spanned_lint(
                                MAP_UNIT_FN,
                                span,
                                MappingToUnit {
                                    function_label: cx.tcx.span_of_impl(*id).unwrap(),
                                    argument_label: args[0].span,
                                    map_label: arg_ty.default_span(cx.tcx),
                                    suggestion: path.ident.span,
                                    replace: "for_each".to_string(),
                                },
                            )
                        }
                    }
                }
            }
        }
    }
}

fn is_impl_slice(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id) {
            return cx.tcx.type_of(impl_id).skip_binder().is_slice();
        }
    }
    false
}

fn is_unit_type(ty: Ty<'_>) -> bool {
    ty.is_unit() || ty.is_never()
}

fn is_diagnostic_name(cx: &LateContext<'_>, id: HirId, name: &str) -> bool {
    if let Some(def_id) = cx.typeck_results().type_dependent_def_id(id) {
        if let Some(item) = cx.tcx.get_diagnostic_name(def_id) {
            if item.as_str() == name {
                return true;
            }
        }
    }
    false
}
