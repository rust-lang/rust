use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{
    intravisit, Body, Expr, ExprKind, FnDecl, HirId, Let, LocalSource, Mutability, Pat, PatKind, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for patterns that aren't exact representations of the types
    /// they are applied to.
    ///
    /// To satisfy this lint, you will have to adjust either the expression that is matched
    /// against or the pattern itself, as well as the bindings that are introduced by the
    /// adjusted patterns. For matching you will have to either dereference the expression
    /// with the `*` operator, or amend the patterns to explicitly match against `&<pattern>`
    /// or `&mut <pattern>` depending on the reference mutability. For the bindings you need
    /// to use the inverse. You can leave them as plain bindings if you wish for the value
    /// to be copied, but you must use `ref mut <variable>` or `ref <variable>` to construct
    /// a reference into the matched structure.
    ///
    /// If you are looking for a way to learn about ownership semantics in more detail, it
    /// is recommended to look at IDE options available to you to highlight types, lifetimes
    /// and reference semantics in your code. The available tooling would expose these things
    /// in a general way even outside of the various pattern matching mechanics. Of course
    /// this lint can still be used to highlight areas of interest and ensure a good understanding
    /// of ownership semantics.
    ///
    /// ### Why is this bad?
    /// It isn't bad in general. But in some contexts it can be desirable
    /// because it increases ownership hints in the code, and will guard against some changes
    /// in ownership.
    ///
    /// ### Example
    /// This example shows the basic adjustments necessary to satisfy the lint. Note how
    /// the matched expression is explicitly dereferenced with `*` and the `inner` variable
    /// is bound to a shared borrow via `ref inner`.
    ///
    /// ```rust,ignore
    /// // Bad
    /// let value = &Some(Box::new(23));
    /// match value {
    ///     Some(inner) => println!("{}", inner),
    ///     None => println!("none"),
    /// }
    ///
    /// // Good
    /// let value = &Some(Box::new(23));
    /// match *value {
    ///     Some(ref inner) => println!("{}", inner),
    ///     None => println!("none"),
    /// }
    /// ```
    ///
    /// The following example demonstrates one of the advantages of the more verbose style.
    /// Note how the second version uses `ref mut a` to explicitly declare `a` a shared mutable
    /// borrow, while `b` is simply taken by value. This ensures that the loop body cannot
    /// accidentally modify the wrong part of the structure.
    ///
    /// ```rust,ignore
    /// // Bad
    /// let mut values = vec![(2, 3), (3, 4)];
    /// for (a, b) in &mut values {
    ///     *a += *b;
    /// }
    ///
    /// // Good
    /// let mut values = vec![(2, 3), (3, 4)];
    /// for &mut (ref mut a, b) in &mut values {
    ///     *a += b;
    /// }
    /// ```
    #[clippy::version = "1.47.0"]
    pub PATTERN_TYPE_MISMATCH,
    restriction,
    "type of pattern does not match the expression type"
}

declare_lint_pass!(PatternTypeMismatch => [PATTERN_TYPE_MISMATCH]);

impl<'tcx> LateLintPass<'tcx> for PatternTypeMismatch {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Local(local) = stmt.kind {
            if in_external_macro(cx.sess(), local.pat.span) {
                return;
            }
            let deref_possible = match local.source {
                LocalSource::Normal => DerefPossible::Possible,
                _ => DerefPossible::Impossible,
            };
            apply_lint(cx, local.pat, deref_possible);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(_, arms, _) = expr.kind {
            for arm in arms {
                let pat = &arm.pat;
                if apply_lint(cx, pat, DerefPossible::Possible) {
                    break;
                }
            }
        }
        if let ExprKind::Let(Let { pat, .. }) = expr.kind {
            apply_lint(cx, pat, DerefPossible::Possible);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        _: HirId,
    ) {
        for param in body.params {
            apply_lint(cx, param.pat, DerefPossible::Impossible);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DerefPossible {
    Possible,
    Impossible,
}

fn apply_lint<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'_>, deref_possible: DerefPossible) -> bool {
    let maybe_mismatch = find_first_mismatch(cx, pat);
    if let Some((span, mutability, level)) = maybe_mismatch {
        span_lint_and_help(
            cx,
            PATTERN_TYPE_MISMATCH,
            span,
            "type of pattern does not match the expression type",
            None,
            &format!(
                "{}explicitly match against a `{}` pattern and adjust the enclosed variable bindings",
                match (deref_possible, level) {
                    (DerefPossible::Possible, Level::Top) => "use `*` to dereference the match expression or ",
                    _ => "",
                },
                match mutability {
                    Mutability::Mut => "&mut _",
                    Mutability::Not => "&_",
                },
            ),
        );
        true
    } else {
        false
    }
}

#[derive(Debug, Copy, Clone)]
enum Level {
    Top,
    Lower,
}

#[allow(rustc::usage_of_ty_tykind)]
fn find_first_mismatch<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'_>) -> Option<(Span, Mutability, Level)> {
    let mut result = None;
    pat.walk(|p| {
        if result.is_some() {
            return false;
        }
        if in_external_macro(cx.sess(), p.span) {
            return true;
        }
        let adjust_pat = match p.kind {
            PatKind::Or([p, ..]) => p,
            _ => p,
        };
        if let Some(adjustments) = cx.typeck_results().pat_adjustments().get(adjust_pat.hir_id) {
            if let [first, ..] = **adjustments {
                if let ty::Ref(.., mutability) = *first.kind() {
                    let level = if p.hir_id == pat.hir_id {
                        Level::Top
                    } else {
                        Level::Lower
                    };
                    result = Some((p.span, mutability, level));
                }
            }
        }
        result.is_none()
    });
    result
}
