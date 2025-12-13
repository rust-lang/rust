use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::Msrv;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::{SpanRangeExt, snippet_with_context};
use clippy_utils::sugg::{Sugg, has_enclosing_paren};
use clippy_utils::ty::implements_trait;
use clippy_utils::{parent_item_name, peel_ref_operators, sym};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BinOpKind, Expr, ExprKind, PatExprKind, PatKind, RustcVersion, StabilityLevel, StableSince};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for getting the length of something via `.len()`
    /// just to compare to zero, and suggests using `.is_empty()` where applicable.
    ///
    /// ### Why is this bad?
    /// Some structures can answer `.is_empty()` much faster
    /// than calculating their length. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// ### Example
    /// ```ignore
    /// if x.len() == 0 {
    ///     ..
    /// }
    /// if y.len() != 0 {
    ///     ..
    /// }
    /// ```
    /// instead use
    /// ```ignore
    /// if x.is_empty() {
    ///     ..
    /// }
    /// if !y.is_empty() {
    ///     ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LEN_ZERO,
    style,
    "checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparing to an empty slice such as `""` or `[]`,
    /// and suggests using `.is_empty()` where applicable.
    ///
    /// ### Why is this bad?
    /// Some structures can answer `.is_empty()` much faster
    /// than checking for equality. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// if s == "" {
    ///     ..
    /// }
    ///
    /// if arr == [] {
    ///     ..
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// if s.is_empty() {
    ///     ..
    /// }
    ///
    /// if arr.is_empty() {
    ///     ..
    /// }
    /// ```
    #[clippy::version = "1.49.0"]
    pub COMPARISON_TO_EMPTY,
    style,
    "checking `x == \"\"` or `x == []` (or similar) when `.is_empty()` could be used instead"
}

pub struct LenZero {
    msrv: Msrv,
}

impl_lint_pass!(LenZero => [LEN_ZERO, COMPARISON_TO_EMPTY]);

impl LenZero {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for LenZero {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Let(lt) = expr.kind
            && match lt.pat.kind {
                PatKind::Slice([], None, []) => true,
                PatKind::Expr(lit)
                    if let PatExprKind::Lit { lit, .. } = lit.kind
                        && let LitKind::Str(lit, _) = lit.node =>
                {
                    lit.as_str().is_empty()
                },
                _ => false,
            }
            && !expr.span.from_expansion()
            && has_is_empty(cx, lt.init, self.msrv)
        {
            let mut applicability = Applicability::MachineApplicable;

            let lit1 = peel_ref_operators(cx, lt.init);
            let lit_str = Sugg::hir_with_context(cx, lit1, lt.span.ctxt(), "_", &mut applicability).maybe_paren();

            span_lint_and_sugg(
                cx,
                COMPARISON_TO_EMPTY,
                lt.span,
                "comparison to empty slice using `if let`",
                "using `is_empty` is clearer and more explicit",
                format!("{lit_str}.is_empty()"),
                applicability,
            );
        }

        if let ExprKind::MethodCall(method, lhs_expr, [rhs_expr], _) = expr.kind
            && cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::PartialEq)
            && !expr.span.from_expansion()
        {
            self.check_empty_expr(
                cx,
                expr.span,
                lhs_expr,
                peel_ref_operators(cx, rhs_expr),
                if method.ident.name == sym::ne {
                    "!"
                } else {
                    Default::default()
                },
            );
        }

        if let ExprKind::Binary(Spanned { node: cmp, .. }, left, right) = expr.kind
            && !expr.span.from_expansion()
        {
            // expr.span might contains parenthesis, see issue #10529
            let actual_span = span_without_enclosing_paren(cx, expr.span);
            match cmp {
                BinOpKind::Eq => {
                    self.check_cmp(cx, actual_span, left, right, "", 0); // len == 0
                    self.check_cmp(cx, actual_span, right, left, "", 0); // 0 == len
                },
                BinOpKind::Ne => {
                    self.check_cmp(cx, actual_span, left, right, "!", 0); // len != 0
                    self.check_cmp(cx, actual_span, right, left, "!", 0); // 0 != len
                },
                BinOpKind::Gt => {
                    self.check_cmp(cx, actual_span, left, right, "!", 0); // len > 0
                    self.check_cmp(cx, actual_span, right, left, "", 1); // 1 > len
                },
                BinOpKind::Lt => {
                    self.check_cmp(cx, actual_span, left, right, "", 1); // len < 1
                    self.check_cmp(cx, actual_span, right, left, "!", 0); // 0 < len
                },
                BinOpKind::Ge => self.check_cmp(cx, actual_span, left, right, "!", 1), // len >= 1
                BinOpKind::Le => self.check_cmp(cx, actual_span, right, left, "!", 1), // 1 <= len
                _ => (),
            }
        }
    }
}

impl LenZero {
    fn check_cmp(
        &self,
        cx: &LateContext<'_>,
        span: Span,
        method: &Expr<'_>,
        lit: &Expr<'_>,
        op: &str,
        compare_to: u32,
    ) {
        if method.span.from_expansion() {
            return;
        }

        if let (&ExprKind::MethodCall(method_path, receiver, [], _), ExprKind::Lit(lit)) = (&method.kind, &lit.kind) {
            // check if we are in an is_empty() method
            if parent_item_name(cx, method) == Some(sym::is_empty) {
                return;
            }

            self.check_len(cx, span, method_path.ident.name, receiver, &lit.node, op, compare_to);
        } else {
            self.check_empty_expr(cx, span, method, lit, op);
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn check_len(
        &self,
        cx: &LateContext<'_>,
        span: Span,
        method_name: Symbol,
        receiver: &Expr<'_>,
        lit: &LitKind,
        op: &str,
        compare_to: u32,
    ) {
        if let LitKind::Int(lit, _) = *lit {
            // check if length is compared to the specified number
            if lit != u128::from(compare_to) {
                return;
            }

            if method_name == sym::len && has_is_empty(cx, receiver, self.msrv) {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    LEN_ZERO,
                    span,
                    format!("length comparison to {}", if compare_to == 0 { "zero" } else { "one" }),
                    format!("using `{op}is_empty` is clearer and more explicit"),
                    format!(
                        "{op}{}.is_empty()",
                        snippet_with_context(cx, receiver.span, span.ctxt(), "_", &mut applicability).0,
                    ),
                    applicability,
                );
            }
        }
    }

    fn check_empty_expr(&self, cx: &LateContext<'_>, span: Span, lit1: &Expr<'_>, lit2: &Expr<'_>, op: &str) {
        if (is_empty_array(lit2) || is_empty_string(lit2)) && has_is_empty(cx, lit1, self.msrv) {
            let mut applicability = Applicability::MachineApplicable;

            let lit1 = peel_ref_operators(cx, lit1);
            let lit_str = Sugg::hir_with_context(cx, lit1, span.ctxt(), "_", &mut applicability).maybe_paren();

            span_lint_and_sugg(
                cx,
                COMPARISON_TO_EMPTY,
                span,
                "comparison to empty slice",
                format!("using `{op}is_empty` is clearer and more explicit"),
                format!("{op}{lit_str}.is_empty()"),
                applicability,
            );
        }
    }
}

fn span_without_enclosing_paren(cx: &LateContext<'_>, span: Span) -> Span {
    let Some(snippet) = span.get_source_text(cx) else {
        return span;
    };
    if has_enclosing_paren(snippet) {
        let source_map = cx.tcx.sess.source_map();
        let left_paren = source_map.start_point(span);
        let right_parent = source_map.end_point(span);
        left_paren.between(right_parent)
    } else {
        span
    }
}

fn is_empty_string(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Str(lit, _) = lit.node
    {
        let lit = lit.as_str();
        return lit.is_empty();
    }
    false
}

fn is_empty_array(expr: &Expr<'_>) -> bool {
    if let ExprKind::Array(arr) = expr.kind {
        return arr.is_empty();
    }
    false
}

/// Checks if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext<'_>, expr: &Expr<'_>, msrv: Msrv) -> bool {
    /// Gets an `AssocItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty_and_stable(cx: &LateContext<'_>, item: &ty::AssocItem, msrv: Msrv) -> bool {
        if item.is_fn() {
            let sig = cx.tcx.fn_sig(item.def_id).skip_binder();
            let ty = sig.skip_binder();
            ty.inputs().len() == 1
                && cx.tcx.lookup_stability(item.def_id).is_none_or(|stability| {
                    if let StabilityLevel::Stable { since, .. } = stability.level {
                        let version = match since {
                            StableSince::Version(version) => version,
                            StableSince::Current => RustcVersion::CURRENT,
                            StableSince::Err(_) => return false,
                        };

                        msrv.meets(cx, version)
                    } else {
                        // Unstable fn, check if the feature is enabled.
                        cx.tcx.features().enabled(stability.feature) && msrv.current(cx).is_none()
                    }
                })
        } else {
            false
        }
    }

    /// Checks the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext<'_>, id: DefId, msrv: Msrv) -> bool {
        cx.tcx.inherent_impls(id).iter().any(|imp| {
            cx.tcx
                .associated_items(*imp)
                .filter_by_name_unhygienic(sym::is_empty)
                .any(|item| is_is_empty_and_stable(cx, item, msrv))
        })
    }

    fn ty_has_is_empty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, depth: usize, msrv: Msrv) -> bool {
        match ty.kind() {
            ty::Dynamic(tt, ..) => tt.principal().is_some_and(|principal| {
                cx.tcx
                    .associated_items(principal.def_id())
                    .filter_by_name_unhygienic(sym::is_empty)
                    .any(|item| is_is_empty_and_stable(cx, item, msrv))
            }),
            ty::Alias(ty::Projection, proj) => has_is_empty_impl(cx, proj.def_id, msrv),
            ty::Adt(id, _) => {
                has_is_empty_impl(cx, id.did(), msrv)
                    || (cx.tcx.recursion_limit().value_within_limit(depth)
                        && cx.tcx.get_diagnostic_item(sym::Deref).is_some_and(|deref_id| {
                            implements_trait(cx, ty, deref_id, &[])
                                && cx
                                    .get_associated_type(ty, deref_id, sym::Target)
                                    .is_some_and(|deref_ty| ty_has_is_empty(cx, deref_ty, depth + 1, msrv))
                        }))
            },
            ty::Array(..) | ty::Slice(..) | ty::Str => true,
            _ => false,
        }
    }

    ty_has_is_empty(cx, cx.typeck_results().expr_ty(expr).peel_refs(), 0, msrv)
}
