use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match` expressions modifying the case of a string with non-compliant arms
    ///
    /// ### Why is this bad?
    /// The arm is unreachable, which is likely a mistake
    ///
    /// ### Example
    /// ```rust
    /// # let text = "Foo";
    ///
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "Bar" => {},
    ///     _ => {},
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # let text = "Foo";
    ///
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "bar" => {},
    ///     _ => {},
    /// }
    /// ```
    #[clippy::version = "1.58.0"]
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

impl<'tcx> LateLintPass<'tcx> for MatchStrCaseMismatch {
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
                    if let Some((bad_case_span, bad_case_sym)) = verify_case(&case_method, arms) {
                        lint(cx, &case_method, bad_case_span, bad_case_sym.as_str());
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
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        match ex.kind {
            ExprKind::MethodCall(segment, [receiver], _) if self.case_altered(segment.ident.as_str(), receiver) => {},
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

fn verify_case<'a>(case_method: &'a CaseMethod, arms: &'a [Arm<'_>]) -> Option<(Span, Symbol)> {
    let case_check = match case_method {
        CaseMethod::LowerCase => |input: &str| -> bool { input.chars().all(|c| c.to_lowercase().next() == Some(c)) },
        CaseMethod::AsciiLowerCase => |input: &str| -> bool { !input.chars().any(|c| c.is_ascii_uppercase()) },
        CaseMethod::UpperCase => |input: &str| -> bool { input.chars().all(|c| c.to_uppercase().next() == Some(c)) },
        CaseMethod::AsciiUppercase => |input: &str| -> bool { !input.chars().any(|c| c.is_ascii_lowercase()) },
    };

    for arm in arms {
        if_chain! {
            if let PatKind::Lit(Expr {
                                kind: ExprKind::Lit(lit),
                                ..
                            }) = arm.pat.kind;
            if let LitKind::Str(symbol, _) = lit.node;
            let input = symbol.as_str();
            if !case_check(input);
            then {
                return Some((lit.span, symbol));
            }
        }
    }

    None
}

fn lint(cx: &LateContext<'_>, case_method: &CaseMethod, bad_case_span: Span, bad_case_str: &str) {
    let (method_str, suggestion) = match case_method {
        CaseMethod::LowerCase => ("to_lowercase", bad_case_str.to_lowercase()),
        CaseMethod::AsciiLowerCase => ("to_ascii_lowercase", bad_case_str.to_ascii_lowercase()),
        CaseMethod::UpperCase => ("to_uppercase", bad_case_str.to_uppercase()),
        CaseMethod::AsciiUppercase => ("to_ascii_uppercase", bad_case_str.to_ascii_uppercase()),
    };

    span_lint_and_sugg(
        cx,
        MATCH_STR_CASE_MISMATCH,
        bad_case_span,
        "this `match` arm has a differing case than its expression",
        &*format!("consider changing the case of this arm to respect `{}`", method_str),
        format!("\"{}\"", suggestion),
        Applicability::MachineApplicable,
    );
}
