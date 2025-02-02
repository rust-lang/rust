use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_copy;
use clippy_utils::{get_parent_expr, is_mutable, path_to_local};
use rustc_hir::{Expr, ExprField, ExprKind, Path, QPath, StructTailExpr, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for initialization of an identical `struct` from another instance
    /// of the type, either by copying a base without setting any field or by
    /// moving all fields individually.
    ///
    /// ### Why is this bad?
    /// Readability suffers from unnecessary struct building.
    ///
    /// ### Example
    /// ```no_run
    /// struct S { s: String }
    ///
    /// let a = S { s: String::from("Hello, world!") };
    /// let b = S { ..a };
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct S { s: String }
    ///
    /// let a = S { s: String::from("Hello, world!") };
    /// let b = a;
    /// ```
    ///
    /// The struct literal ``S { ..a }`` in the assignment to ``b`` could be replaced
    /// with just ``a``.
    ///
    /// ### Known Problems
    /// Has false positives when the base is a place expression that cannot be
    /// moved out of, see [#10547](https://github.com/rust-lang/rust-clippy/issues/10547).
    ///
    /// Empty structs are ignored by the lint.
    #[clippy::version = "1.70.0"]
    pub UNNECESSARY_STRUCT_INITIALIZATION,
    nursery,
    "struct built from a base that can be written mode concisely"
}
declare_lint_pass!(UnnecessaryStruct => [UNNECESSARY_STRUCT_INITIALIZATION]);

impl LateLintPass<'_> for UnnecessaryStruct {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ExprKind::Struct(_, fields, base) = expr.kind else {
            return;
        };

        if expr.span.from_expansion() {
            // Prevent lint from hitting inside macro code
            return;
        }

        let field_path = same_path_in_all_fields(cx, expr, fields);

        let sugg = match (field_path, base) {
            (Some(&path), StructTailExpr::None | StructTailExpr::DefaultFields(_)) => {
                // all fields match, no base given
                path.span
            },
            (Some(path), StructTailExpr::Base(base))
                if base_is_suitable(cx, expr, base) && path_matches_base(path, base) =>
            {
                // all fields match, has base: ensure that the path of the base matches
                base.span
            },
            (None, StructTailExpr::Base(base)) if fields.is_empty() && base_is_suitable(cx, expr, base) => {
                // just the base, no explicit fields
                base.span
            },
            _ => return,
        };

        span_lint_and_sugg(
            cx,
            UNNECESSARY_STRUCT_INITIALIZATION,
            expr.span,
            "unnecessary struct building",
            "replace with",
            snippet(cx, sugg, "..").into_owned(),
            rustc_errors::Applicability::MachineApplicable,
        );
    }
}

fn base_is_suitable(cx: &LateContext<'_>, expr: &Expr<'_>, base: &Expr<'_>) -> bool {
    if !check_references(cx, expr, base) {
        return false;
    }

    // TODO: do not propose to replace *XX if XX is not Copy
    if let ExprKind::Unary(UnOp::Deref, target) = base.kind
        && matches!(target.kind, ExprKind::Path(..))
        && !is_copy(cx, cx.typeck_results().expr_ty(expr))
    {
        // `*base` cannot be used instead of the struct in the general case if it is not Copy.
        return false;
    }
    true
}

/// Check whether all fields of a struct assignment match.
/// Returns a [Path] item that one can obtain a span from for the lint suggestion.
///
/// Conditions that must be satisfied to trigger this variant of the lint:
///
/// - source struct of the assignment must be of same type as the destination
/// - names of destination struct fields must match the field names of the source
///
/// We don’t check here if all struct fields are assigned as the remainder may
/// be filled in from a base struct.
fn same_path_in_all_fields<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    fields: &[ExprField<'tcx>],
) -> Option<&'tcx Path<'tcx>> {
    let ty = cx.typeck_results().expr_ty(expr);

    let mut found = None;

    for f in fields {
        // fields are assigned from expression
        if let ExprKind::Field(src_expr, ident) = f.expr.kind
            // expression type matches
            && ty == cx.typeck_results().expr_ty(src_expr)
            // field name matches
            && f.ident == ident
            // assigned from a path expression
            && let ExprKind::Path(QPath::Resolved(None, src_path)) = src_expr.kind
        {
            let Some((_, p)) = found else {
                // this is the first field assignment in the list
                found = Some((src_expr, src_path));
                continue;
            };

            if p.res == src_path.res {
                // subsequent field assignment with same origin struct as before
                continue;
            }
        }
        // source of field assignment doesn’t qualify
        return None;
    }

    if let Some((src_expr, src_path)) = found
        && check_references(cx, expr, src_expr)
    {
        Some(src_path)
    } else {
        None
    }
}

fn check_references(cx: &LateContext<'_>, expr_a: &Expr<'_>, expr_b: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr_a)
        && let parent_ty = cx.typeck_results().expr_ty_adjusted(parent)
        && parent_ty.is_any_ptr()
    {
        if is_copy(cx, cx.typeck_results().expr_ty(expr_a)) && path_to_local(expr_b).is_some() {
            // When the type implements `Copy`, a reference to the new struct works on the
            // copy. Using the original would borrow it.
            return false;
        }

        if parent_ty.is_mutable_ptr() && !is_mutable(cx, expr_b) {
            // The original can be used in a mutable reference context only if it is mutable.
            return false;
        }
    }

    true
}

/// When some fields are assigned from a base struct and others individually
/// the lint applies only if the source of the field is the same as the base.
/// This is enforced here by comparing the path of the base expression;
/// needless to say the lint only applies if it (or whatever expression it is
/// a reference of) actually has a path.
fn path_matches_base(path: &Path<'_>, base: &Expr<'_>) -> bool {
    let base_path = match base.kind {
        ExprKind::Unary(UnOp::Deref, base_expr) => {
            if let ExprKind::Path(QPath::Resolved(_, base_path)) = base_expr.kind {
                base_path
            } else {
                return false;
            }
        },
        ExprKind::Path(QPath::Resolved(_, base_path)) => base_path,
        _ => return false,
    };
    path.res == base_path.res
}
