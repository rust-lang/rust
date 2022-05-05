use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{
    can_move_expr_to_closure, eager_or_lazy, higher, in_constant, is_else_clause, is_lang_ctor, peel_blocks,
    peel_hir_expr_while, CaptureKind,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::OptionSome;
use rustc_hir::{def::Res, BindingAnnotation, Expr, ExprKind, Mutability, PatKind, Path, QPath, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Lints usage of `if let Some(v) = ... { y } else { x }` which is more
    /// idiomatically done with `Option::map_or` (if the else bit is a pure
    /// expression) or `Option::map_or_else` (if the else bit is an impure
    /// expression).
    ///
    /// ### Why is this bad?
    /// Using the dedicated functions of the `Option` type is clearer and
    /// more concise than an `if let` expression.
    ///
    /// ### Known problems
    /// This lint uses a deliberately conservative metric for checking
    /// if the inside of either body contains breaks or continues which will
    /// cause it to not suggest a fix if either block contains a loop with
    /// continues or breaks contained within the loop.
    ///
    /// ### Example
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     5
    /// };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     let y = do_complicated_function();
    ///     y*y
    /// };
    /// ```
    ///
    /// should be
    ///
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = optional.map_or(5, |foo| foo);
    /// let _ = optional.map_or_else(||{
    ///     let y = do_complicated_function();
    ///     y*y
    /// }, |foo| foo);
    /// ```
    #[clippy::version = "1.47.0"]
    pub OPTION_IF_LET_ELSE,
    nursery,
    "reimplementation of Option::map_or"
}

declare_lint_pass!(OptionIfLetElse => [OPTION_IF_LET_ELSE]);

/// Returns true iff the given expression is the result of calling `Result::ok`
fn is_result_ok(cx: &LateContext<'_>, expr: &'_ Expr<'_>) -> bool {
    if let ExprKind::MethodCall(path, &[ref receiver], _) = &expr.kind {
        path.ident.name.as_str() == "ok"
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(receiver), sym::Result)
    } else {
        false
    }
}

/// A struct containing information about occurrences of the
/// `if let Some(..) = .. else` construct that this lint detects.
struct OptionIfLetElseOccurence {
    option: String,
    method_sugg: String,
    some_expr: String,
    none_expr: String,
}

fn format_option_in_sugg(cx: &LateContext<'_>, cond_expr: &Expr<'_>, as_ref: bool, as_mut: bool) -> String {
    format!(
        "{}{}",
        Sugg::hir_with_macro_callsite(cx, cond_expr, "..").maybe_par(),
        if as_mut {
            ".as_mut()"
        } else if as_ref {
            ".as_ref()"
        } else {
            ""
        }
    )
}

/// If this expression is the option if let/else construct we're detecting, then
/// this function returns an `OptionIfLetElseOccurence` struct with details if
/// this construct is found, or None if this construct is not found.
fn detect_option_if_let_else<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<OptionIfLetElseOccurence> {
    if_chain! {
        if !expr.span.from_expansion(); // Don't lint macros, because it behaves weirdly
        if !in_constant(cx, expr.hir_id);
        if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else: Some(if_else) })
            = higher::IfLet::hir(cx, expr);
        if !is_else_clause(cx.tcx, expr);
        if !is_result_ok(cx, let_expr); // Don't lint on Result::ok because a different lint does it already
        if let PatKind::TupleStruct(struct_qpath, [inner_pat], _) = &let_pat.kind;
        if is_lang_ctor(cx, struct_qpath, OptionSome);
        if let PatKind::Binding(bind_annotation, _, id, None) = &inner_pat.kind;
        if let Some(some_captures) = can_move_expr_to_closure(cx, if_then);
        if let Some(none_captures) = can_move_expr_to_closure(cx, if_else);
        if some_captures
            .iter()
            .filter_map(|(id, &c)| none_captures.get(id).map(|&c2| (c, c2)))
            .all(|(x, y)| x.is_imm_ref() && y.is_imm_ref());

        then {
            let capture_mut = if bind_annotation == &BindingAnnotation::Mutable { "mut " } else { "" };
            let some_body = peel_blocks(if_then);
            let none_body = peel_blocks(if_else);
            let method_sugg = if eager_or_lazy::switch_to_eager_eval(cx, none_body) { "map_or" } else { "map_or_else" };
            let capture_name = id.name.to_ident_string();
            let (as_ref, as_mut) = match &let_expr.kind {
                ExprKind::AddrOf(_, Mutability::Not, _) => (true, false),
                ExprKind::AddrOf(_, Mutability::Mut, _) => (false, true),
                _ => (bind_annotation == &BindingAnnotation::Ref, bind_annotation == &BindingAnnotation::RefMut),
            };
            let cond_expr = match let_expr.kind {
                // Pointer dereferencing happens automatically, so we can omit it in the suggestion
                ExprKind::Unary(UnOp::Deref, expr) | ExprKind::AddrOf(_, _, expr) => expr,
                _ => let_expr,
            };
            // Check if captures the closure will need conflict with borrows made in the scrutinee.
            // TODO: check all the references made in the scrutinee expression. This will require interacting
            // with the borrow checker. Currently only `<local>[.<field>]*` is checked for.
            if as_ref || as_mut {
                let e = peel_hir_expr_while(cond_expr, |e| match e.kind {
                    ExprKind::Field(e, _) | ExprKind::AddrOf(_, _, e) => Some(e),
                    _ => None,
                });
                if let ExprKind::Path(QPath::Resolved(None, Path { res: Res::Local(local_id), .. })) = e.kind {
                    match some_captures.get(local_id)
                        .or_else(|| (method_sugg == "map_or_else").then(|| ()).and_then(|_| none_captures.get(local_id)))
                    {
                        Some(CaptureKind::Value | CaptureKind::Ref(Mutability::Mut)) => return None,
                        Some(CaptureKind::Ref(Mutability::Not)) if as_mut => return None,
                        Some(CaptureKind::Ref(Mutability::Not)) | None => (),
                    }
                }
            }
            Some(OptionIfLetElseOccurence {
                option: format_option_in_sugg(cx, cond_expr, as_ref, as_mut),
                method_sugg: method_sugg.to_string(),
                some_expr: format!("|{}{}| {}", capture_mut, capture_name, Sugg::hir_with_macro_callsite(cx, some_body, "..")),
                none_expr: format!("{}{}", if method_sugg == "map_or" { "" } else { "|| " }, Sugg::hir_with_macro_callsite(cx, none_body, "..")),
            })
        } else {
            None
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for OptionIfLetElse {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let Some(detection) = detect_option_if_let_else(cx, expr) {
            span_lint_and_sugg(
                cx,
                OPTION_IF_LET_ELSE,
                expr.span,
                format!("use Option::{} instead of an if let/else", detection.method_sugg).as_str(),
                "try",
                format!(
                    "{}.{}({}, {})",
                    detection.option, detection.method_sugg, detection.none_expr, detection.some_expr,
                ),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
