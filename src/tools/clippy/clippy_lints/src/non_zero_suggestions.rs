use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use rustc_ast::ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for conversions from `NonZero` types to regular integer types,
    /// and suggests using `NonZero` types for the target as well.
    ///
    /// ### Why is this bad?
    /// Converting from `NonZero` types to regular integer types and then back to `NonZero`
    /// types is less efficient and loses the type-safety guarantees provided by `NonZero` types.
    /// Using `NonZero` types consistently can lead to more optimized code and prevent
    /// certain classes of errors related to zero values.
    ///
    /// ### Example
    /// ```no_run
    /// use std::num::{NonZeroU32, NonZeroU64};
    ///
    /// fn example(x: u64, y: NonZeroU32) {
    ///     // Bad: Converting NonZeroU32 to u64 unnecessarily
    ///     let r1 = x / u64::from(y.get());
    ///     let r2 = x % u64::from(y.get());
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::num::{NonZeroU32, NonZeroU64};
    ///
    /// fn example(x: u64, y: NonZeroU32) {
    ///     // Good: Preserving the NonZero property
    ///     let r1 = x / NonZeroU64::from(y);
    ///     let r2 = x % NonZeroU64::from(y);
    /// }
    /// ```
    #[clippy::version = "1.83.0"]
    pub NON_ZERO_SUGGESTIONS,
    restriction,
    "suggests using `NonZero#` from `u#` or `i#` for more efficient and type-safe conversions"
}

declare_lint_pass!(NonZeroSuggestions => [NON_ZERO_SUGGESTIONS]);

impl<'tcx> LateLintPass<'tcx> for NonZeroSuggestions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Binary(op, _, rhs) = expr.kind
            && matches!(op.node, BinOpKind::Div | BinOpKind::Rem)
        {
            check_non_zero_conversion(cx, rhs, Applicability::MachineApplicable);
        } else {
            // Check if the parent expression is a binary operation
            let parent_is_binary = cx.tcx.hir_parent_iter(expr.hir_id).any(|(_, node)| {
                matches!(node, rustc_hir::Node::Expr(parent_expr) if matches!(parent_expr.kind, ExprKind::Binary(..)))
            });

            if !parent_is_binary {
                check_non_zero_conversion(cx, expr, Applicability::MaybeIncorrect);
            }
        }
    }
}

fn check_non_zero_conversion(cx: &LateContext<'_>, expr: &Expr<'_>, applicability: Applicability) {
    // Check if the expression is a function call with one argument
    if let ExprKind::Call(func, [arg]) = expr.kind
        && let ExprKind::Path(qpath) = &func.kind
        && let Some(def_id) = cx.qpath_res(qpath, func.hir_id).opt_def_id()
        && let ExprKind::MethodCall(rcv_path, receiver, [], _) = &arg.kind
        && rcv_path.ident.name.as_str() == "get"
    {
        let fn_name = cx.tcx.item_name(def_id);
        let target_ty = cx.typeck_results().expr_ty(expr);
        let receiver_ty = cx.typeck_results().expr_ty(receiver);

        // Check if the receiver type is a NonZero type
        if let ty::Adt(adt_def, _) = receiver_ty.kind()
            && adt_def.is_struct()
            && cx.tcx.get_diagnostic_name(adt_def.did()) == Some(sym::NonZero)
            && let Some(target_non_zero_type) = get_target_non_zero_type(target_ty)
        {
            let arg_snippet = get_arg_snippet(cx, arg, rcv_path);
            suggest_non_zero_conversion(cx, expr, fn_name, target_non_zero_type, &arg_snippet, applicability);
        }
    }
}

fn get_arg_snippet(cx: &LateContext<'_>, arg: &Expr<'_>, rcv_path: &rustc_hir::PathSegment<'_>) -> String {
    let arg_snippet = snippet(cx, arg.span, "..");
    if let Some(index) = arg_snippet.rfind(&format!(".{}", rcv_path.ident.name)) {
        arg_snippet[..index].trim().to_string()
    } else {
        arg_snippet.to_string()
    }
}

fn suggest_non_zero_conversion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    fn_name: rustc_span::Symbol,
    target_non_zero_type: &str,
    arg_snippet: &str,
    applicability: Applicability,
) {
    let suggestion = format!("{target_non_zero_type}::{fn_name}({arg_snippet})");
    span_lint_and_sugg(
        cx,
        NON_ZERO_SUGGESTIONS,
        expr.span,
        format!("consider using `{target_non_zero_type}::{fn_name}()` for more efficient and type-safe conversion"),
        "replace with",
        suggestion,
        applicability,
    );
}

fn get_target_non_zero_type(ty: Ty<'_>) -> Option<&'static str> {
    match ty.kind() {
        ty::Uint(uint_ty) => Some(match uint_ty {
            ty::UintTy::U8 => "NonZeroU8",
            ty::UintTy::U16 => "NonZeroU16",
            ty::UintTy::U32 => "NonZeroU32",
            ty::UintTy::U64 => "NonZeroU64",
            ty::UintTy::U128 => "NonZeroU128",
            ty::UintTy::Usize => "NonZeroUsize",
        }),
        ty::Int(int_ty) => Some(match int_ty {
            ty::IntTy::I8 => "NonZeroI8",
            ty::IntTy::I16 => "NonZeroI16",
            ty::IntTy::I32 => "NonZeroI32",
            ty::IntTy::I64 => "NonZeroI64",
            ty::IntTy::I128 => "NonZeroI128",
            ty::IntTy::Isize => "NonZeroIsize",
        }),
        _ => None,
    }
}
