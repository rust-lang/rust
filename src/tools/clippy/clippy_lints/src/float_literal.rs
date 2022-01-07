use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::numeric_literal;
use if_chain::if_chain;
use rustc_ast::ast::{self, LitFloatType, LitKind};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, FloatTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::fmt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for float literals with a precision greater
    /// than that supported by the underlying type.
    ///
    /// ### Why is this bad?
    /// Rust will truncate the literal silently.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let v: f32 = 0.123_456_789_9;
    /// println!("{}", v); //  0.123_456_789
    ///
    /// // Good
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
    /// ### Why is this bad?
    /// Rust will silently lose precision during
    /// conversion to a float.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let _: f32 = 16_777_217.0; // 16_777_216.0
    ///
    /// // Good
    /// let _: f32 = 16_777_216.0;
    /// let _: f64 = 16_777_217.0;
    /// ```
    #[clippy::version = "1.43.0"]
    pub LOSSY_FLOAT_LITERAL,
    restriction,
    "lossy whole number float literals"
}

declare_lint_pass!(FloatLiteral => [EXCESSIVE_PRECISION, LOSSY_FLOAT_LITERAL]);

impl<'tcx> LateLintPass<'tcx> for FloatLiteral {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        let ty = cx.typeck_results().expr_ty(expr);
        if_chain! {
            if let ty::Float(fty) = *ty.kind();
            if let hir::ExprKind::Lit(ref lit) = expr.kind;
            if let LitKind::Float(sym, lit_float_ty) = lit.node;
            then {
                let sym_str = sym.as_str();
                let formatter = FloatFormat::new(sym_str);
                // Try to bail out if the float is for sure fine.
                // If its within the 2 decimal digits of being out of precision we
                // check if the parsed representation is the same as the string
                // since we'll need the truncated string anyway.
                let digits = count_digits(sym_str);
                let max = max_digits(fty);
                let type_suffix = match lit_float_ty {
                    LitFloatType::Suffixed(ast::FloatTy::F32) => Some("f32"),
                    LitFloatType::Suffixed(ast::FloatTy::F64) => Some("f64"),
                    LitFloatType::Unsuffixed => None
                };
                let (is_whole, mut float_str) = match fty {
                    FloatTy::F32 => {
                        let value = sym_str.parse::<f32>().unwrap();

                        (value.fract() == 0.0, formatter.format(value))
                    },
                    FloatTy::F64 => {
                        let value = sym_str.parse::<f64>().unwrap();

                        (value.fract() == 0.0, formatter.format(value))
                    },
                };

                if is_whole && !sym_str.contains(|c| c == 'e' || c == 'E') {
                    // Normalize the literal by stripping the fractional portion
                    if sym_str.split('.').next().unwrap() != float_str {
                        // If the type suffix is missing the suggestion would be
                        // incorrectly interpreted as an integer so adding a `.0`
                        // suffix to prevent that.
                        if type_suffix.is_none() {
                            float_str.push_str(".0");
                        }

                        span_lint_and_sugg(
                            cx,
                            LOSSY_FLOAT_LITERAL,
                            expr.span,
                            "literal cannot be represented as the underlying type without loss of precision",
                            "consider changing the type or replacing it with",
                            numeric_literal::format(&float_str, type_suffix, true),
                            Applicability::MachineApplicable,
                        );
                    }
                } else if digits > max as usize && float_str.len() < sym_str.len() {
                    span_lint_and_sugg(
                        cx,
                        EXCESSIVE_PRECISION,
                        expr.span,
                        "float has excessive precision",
                        "consider changing the type or truncating it to",
                        numeric_literal::format(&float_str, type_suffix, true),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
    }
}

#[must_use]
fn max_digits(fty: FloatTy) -> u32 {
    match fty {
        FloatTy::F32 => f32::DIGITS,
        FloatTy::F64 => f64::DIGITS,
    }
}

/// Counts the digits excluding leading zeros
#[must_use]
fn count_digits(s: &str) -> usize {
    // Note that s does not contain the f32/64 suffix, and underscores have been stripped
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
            Self::LowerExp => format!("{:e}", f),
            Self::UpperExp => format!("{:E}", f),
            Self::Normal => format!("{}", f),
        }
    }
}
