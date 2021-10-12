use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::ast::LitKind;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match` expressions modifying the case of a string with non-compliant arms
    ///
    /// ### Why is this bad?
    /// The arm is unreachable, which is likely a mistake
    ///
    /// ### Example
    /// ```rust,no_run
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "Bar" => {},
    ///     _ => {},
    /// }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "bar" => {},
    ///     _ => {},
    /// }
    /// ```
    pub MATCH_STR_CASE_MISMATCH,
    correctness,
    "creation of a case altering match expression with non-compliant arms"
}

declare_lint_pass!(MatchStrCaseMismatch => [MATCH_STR_CASE_MISMATCH]);

#[derive(Debug)]
enum CaseMethod {
    LowerCase,
    AsciiLowerCase,
    UpperCase,
    AsciiUppercase,
}

impl LateLintPass<'_> for MatchStrCaseMismatch {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if !in_external_macro(cx.tcx.sess, expr.span);
            if let ExprKind::Match(match_expr, arms, MatchSource::Normal) = expr.kind;
            if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty(match_expr).kind();
            if let ty::Str = ty.kind();
            then {
                let mut visitor = MatchExprVisitor {
                    cx,
                    case_method: None,
                };

                visitor.visit_expr(match_expr);

                if let Some(case_method) = visitor.case_method {
                    if let Some(bad_case) = verify_case(&case_method, arms) {
                        lint(cx, expr.span, &case_method, bad_case);
                    }
                }
            }
        }
    }
}

struct MatchExprVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    case_method: Option<CaseMethod>,
}

impl<'a, 'tcx> Visitor<'tcx> for MatchExprVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        match ex.kind {
            ExprKind::MethodCall(segment, _, [receiver], _)
                if self.case_altered(&*segment.ident.as_str(), receiver) => {},
            _ => walk_expr(self, ex),
        }
    }
}

impl<'a, 'tcx> MatchExprVisitor<'a, 'tcx> {
    fn case_altered(&mut self, segment_ident: &str, receiver: &Expr<'_>) -> bool {
        if let Some(case_method) = get_case_method(segment_ident) {
            let ty = self.cx.typeck_results().expr_ty(receiver).peel_refs();

            if is_type_diagnostic_item(self.cx, ty, sym::String) || ty.kind() == &ty::Str {
                self.case_method = Some(case_method);
                return true;
            }
        }

        false
    }
}

fn get_case_method(segment_ident_str: &str) -> Option<CaseMethod> {
    match segment_ident_str {
        "to_lowercase" => Some(CaseMethod::LowerCase),
        "to_ascii_lowercase" => Some(CaseMethod::AsciiLowerCase),
        "to_uppercase" => Some(CaseMethod::UpperCase),
        "to_ascii_uppercase" => Some(CaseMethod::AsciiUppercase),
        _ => None,
    }
}

fn verify_case(case_method: &CaseMethod, arms: &'_ [Arm<'_>]) -> Option<Span> {
    let mut bad_case = None;

    let case_check = match case_method {
        CaseMethod::LowerCase => |input: &str| -> bool { input.chars().all(char::is_lowercase) },
        CaseMethod::AsciiLowerCase => |input: &str| -> bool { input.chars().all(|c| matches!(c, 'a'..='z')) },
        CaseMethod::UpperCase => |input: &str| -> bool { input.chars().all(char::is_uppercase) },
        CaseMethod::AsciiUppercase => |input: &str| -> bool { input.chars().all(|c| matches!(c, 'A'..='Z')) },
    };

    for arm in arms {
        if_chain! {
            if let PatKind::Lit(Expr {
                                kind: ExprKind::Lit(lit),
                                ..
                            }) = arm.pat.kind;
            if let LitKind::Str(symbol, _) = lit.node;
            if !case_check(&symbol.as_str());
            then {
                bad_case = Some(lit.span);
                break;
            }
        }
    }

    bad_case
}

fn lint(cx: &LateContext<'_>, expr_span: Span, case_method: &CaseMethod, bad_case_span: Span) {
    let method_str = match case_method {
        CaseMethod::LowerCase => "to_lower_case",
        CaseMethod::AsciiLowerCase => "to_ascii_lowercase",
        CaseMethod::UpperCase => "to_uppercase",
        CaseMethod::AsciiUppercase => "to_ascii_uppercase",
    };

    span_lint_and_help(
        cx,
        MATCH_STR_CASE_MISMATCH,
        expr_span,
        "this `match` expression alters case, but has non-compliant arms",
        Some(bad_case_span),
        &*format!("consider changing the case of this arm to respect `{}`", method_str),
    );
}
