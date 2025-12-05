use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{ExprUseNode, expr_use_ctxt, numeric_literal};
use rustc_ast::ast::{LitFloatType, LitKind};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, FloatTy};
use rustc_session::impl_lint_pass;
use std::fmt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for float literals with a precision greater
    /// than that supported by the underlying type.
    ///
    /// The lint is suppressed for literals with over `const_literal_digits_threshold` digits.
    ///
    /// ### Why is this bad?
    /// Rust will truncate the literal silently.
    ///
    /// ### Example
    /// ```no_run
    /// let v: f32 = 0.123_456_789_9;
    /// println!("{}", v); //  0.123_456_789
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let v: f64 = 0.123_456_789_9;
    /// println!("{}", v); //  0.123_456_789_9
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXCESSIVE_PRECISION,
    style,
    "excessive precision for float literal"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for whole number float literals that
    /// cannot be represented as the underlying type without loss.
    ///
    /// ### Why restrict this?
    /// If the value was intended to be exact, it will not be.
    /// This may be especially surprising when the lost precision is to the left of the decimal point.
    ///
    /// ### Example
    /// ```no_run
    /// let _: f32 = 16_777_217.0; // 16_777_216.0
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let _: f32 = 16_777_216.0;
    /// let _: f64 = 16_777_217.0;
    /// ```
    #[clippy::version = "1.43.0"]
    pub LOSSY_FLOAT_LITERAL,
    restriction,
    "lossy whole number float literals"
}

pub struct FloatLiteral {
    const_literal_digits_threshold: usize,
}

impl_lint_pass!(FloatLiteral => [
    EXCESSIVE_PRECISION, LOSSY_FLOAT_LITERAL
]);

impl FloatLiteral {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            const_literal_digits_threshold: conf.const_literal_digits_threshold,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for FloatLiteral {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let hir::ExprKind::Lit(lit) = expr.kind
            && let LitKind::Float(sym, lit_float_ty) = lit.node
            && let ty::Float(fty) = *cx.typeck_results().expr_ty(expr).kind()
        {
            let sym_str = sym.as_str();
            let formatter = FloatFormat::new(sym_str);
            // Try to bail out if the float is for sure fine.
            // If its within the 2 decimal digits of being out of precision we
            // check if the parsed representation is the same as the string
            // since we'll need the truncated string anyway.
            let digits = count_digits(sym_str);
            let max = max_digits(fty);
            let type_suffix = match lit_float_ty {
                LitFloatType::Suffixed(FloatTy::F16) => Some("f16"),
                LitFloatType::Suffixed(FloatTy::F32) => Some("f32"),
                LitFloatType::Suffixed(FloatTy::F64) => Some("f64"),
                LitFloatType::Suffixed(FloatTy::F128) => Some("f128"),
                LitFloatType::Unsuffixed => None,
            };
            let (is_whole, is_inf, mut float_str) = match fty {
                FloatTy::F16 | FloatTy::F128 => {
                    // FIXME(f16_f128): do a check like the others when parsing is available
                    return;
                },
                FloatTy::F32 => {
                    let value = sym_str.parse::<f32>().unwrap();

                    (value.fract() == 0.0, value.is_infinite(), formatter.format(value))
                },
                FloatTy::F64 => {
                    let value = sym_str.parse::<f64>().unwrap();

                    (value.fract() == 0.0, value.is_infinite(), formatter.format(value))
                },
            };

            if is_inf {
                return;
            }

            if is_whole && !sym_str.contains(['e', 'E']) {
                // Normalize the literal by stripping the fractional portion
                if sym_str.split('.').next().unwrap() != float_str {
                    span_lint_and_then(
                        cx,
                        LOSSY_FLOAT_LITERAL,
                        expr.span,
                        "literal cannot be represented as the underlying type without loss of precision",
                        |diag| {
                            // If the type suffix is missing the suggestion would be
                            // incorrectly interpreted as an integer so adding a `.0`
                            // suffix to prevent that.
                            if type_suffix.is_none() {
                                float_str.push_str(".0");
                            }
                            diag.span_suggestion_verbose(
                                expr.span,
                                "consider changing the type or replacing it with",
                                numeric_literal::format(&float_str, type_suffix, true),
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
            } else if digits > max as usize && count_digits(&float_str) < digits {
                if digits >= self.const_literal_digits_threshold
                    && matches!(expr_use_ctxt(cx, expr).use_node(cx), ExprUseNode::ConstStatic(_))
                {
                    // If a big enough number of digits is specified and it's a constant
                    // we assume the user is definining a constant, and excessive precision is ok
                    return;
                }
                span_lint_and_then(
                    cx,
                    EXCESSIVE_PRECISION,
                    expr.span,
                    "float has excessive precision",
                    |diag| {
                        if digits >= self.const_literal_digits_threshold
                            && let Some(let_stmt) = maybe_let_stmt(cx, expr)
                        {
                            diag.span_note(let_stmt.span, "consider making it a `const` item");
                        }
                        diag.span_suggestion_verbose(
                            expr.span,
                            "consider changing the type or truncating it to",
                            numeric_literal::format(&float_str, type_suffix, true),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }
    }
}

#[must_use]
fn max_digits(fty: FloatTy) -> u32 {
    match fty {
        FloatTy::F16 => f16::DIGITS,
        FloatTy::F32 => f32::DIGITS,
        FloatTy::F64 => f64::DIGITS,
        FloatTy::F128 => f128::DIGITS,
    }
}

/// Counts the digits excluding leading zeros
#[must_use]
fn count_digits(s: &str) -> usize {
    // Note that s does not contain the `f{16,32,64,128}` suffix, and underscores have been stripped
    s.chars()
        .filter(|c| *c != '-' && *c != '.')
        .take_while(|c| *c != 'e' && *c != 'E')
        .fold(0, |count, c| {
            // leading zeros
            if c == '0' && count == 0 { count } else { count + 1 }
        })
}

enum FloatFormat {
    LowerExp,
    UpperExp,
    Normal,
}
impl FloatFormat {
    #[must_use]
    fn new(s: &str) -> Self {
        s.chars()
            .find_map(|x| match x {
                'e' => Some(Self::LowerExp),
                'E' => Some(Self::UpperExp),
                _ => None,
            })
            .unwrap_or(Self::Normal)
    }
    fn format<T>(&self, f: T) -> String
    where
        T: fmt::UpperExp + fmt::LowerExp + fmt::Display,
    {
        match self {
            Self::LowerExp => format!("{f:e}"),
            Self::UpperExp => format!("{f:E}"),
            Self::Normal => format!("{f}"),
        }
    }
}

fn maybe_let_stmt<'a>(cx: &LateContext<'a>, expr: &hir::Expr<'_>) -> Option<&'a hir::LetStmt<'a>> {
    let parent = cx.tcx.parent_hir_node(expr.hir_id);
    match parent {
        hir::Node::LetStmt(let_stmt) => Some(let_stmt),
        _ => None,
    }
}
