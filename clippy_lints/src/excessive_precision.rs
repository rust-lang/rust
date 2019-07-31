use crate::utils::span_lint_and_sugg;
use if_chain::if_chain;
use rustc::hir;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::f32;
use std::f64;
use std::fmt;
use syntax::ast::*;
use syntax_pos::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks for float literals with a precision greater
    /// than that supported by the underlying type
    ///
    /// **Why is this bad?** Rust will truncate the literal silently.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // Bad
    /// let v: f32 = 0.123_456_789_9;
    /// println!("{}", v); //  0.123_456_789
    ///
    /// // Good
    /// let v: f64 = 0.123_456_789_9;
    /// println!("{}", v); //  0.123_456_789_9
    /// ```
    pub EXCESSIVE_PRECISION,
    style,
    "excessive precision for float literal"
}

declare_lint_pass!(ExcessivePrecision => [EXCESSIVE_PRECISION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ExcessivePrecision {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if_chain! {
            let ty = cx.tables.expr_ty(expr);
            if let ty::Float(fty) = ty.sty;
            if let hir::ExprKind::Lit(ref lit) = expr.node;
            if let LitKind::Float(sym, _) | LitKind::FloatUnsuffixed(sym) = lit.node;
            if let Some(sugg) = self.check(sym, fty);
            then {
                span_lint_and_sugg(
                    cx,
                    EXCESSIVE_PRECISION,
                    expr.span,
                    "float has excessive precision",
                    "consider changing the type or truncating it to",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

impl ExcessivePrecision {
    // None if nothing to lint, Some(suggestion) if lint necessary
    fn check(self, sym: Symbol, fty: FloatTy) -> Option<String> {
        let max = max_digits(fty);
        let sym_str = sym.as_str();
        if dot_zero_exclusion(&sym_str) {
            return None;
        }
        // Try to bail out if the float is for sure fine.
        // If its within the 2 decimal digits of being out of precision we
        // check if the parsed representation is the same as the string
        // since we'll need the truncated string anyway.
        let digits = count_digits(&sym_str);
        if digits > max as usize {
            let formatter = FloatFormat::new(&sym_str);
            let sr = match fty {
                FloatTy::F32 => sym_str.parse::<f32>().map(|f| formatter.format(f)),
                FloatTy::F64 => sym_str.parse::<f64>().map(|f| formatter.format(f)),
            };
            // We know this will parse since we are in LatePass
            let s = sr.unwrap();

            if sym_str == s {
                None
            } else {
                let di = super::literal_representation::DigitInfo::new(&s, true);
                Some(di.grouping_hint())
            }
        } else {
            None
        }
    }
}

/// Should we exclude the float because it has a `.0` or `.` suffix
/// Ex `1_000_000_000.0`
/// Ex `1_000_000_000.`
fn dot_zero_exclusion(s: &str) -> bool {
    if let Some(after_dec) = s.split('.').nth(1) {
        let mut decpart = after_dec.chars().take_while(|c| *c != 'e' || *c != 'E');

        match decpart.next() {
            Some('0') => decpart.count() == 0,
            Some(_) => false,
            None => true,
        }
    } else {
        false
    }
}

fn max_digits(fty: FloatTy) -> u32 {
    match fty {
        FloatTy::F32 => f32::DIGITS,
        FloatTy::F64 => f64::DIGITS,
    }
}

/// Counts the digits excluding leading zeros
fn count_digits(s: &str) -> usize {
    // Note that s does not contain the f32/64 suffix, and underscores have been stripped
    s.chars()
        .filter(|c| *c != '-' && *c != '.')
        .take_while(|c| *c != 'e' && *c != 'E')
        .fold(0, |count, c| {
            // leading zeros
            if c == '0' && count == 0 {
                count
            } else {
                count + 1
            }
        })
}

enum FloatFormat {
    LowerExp,
    UpperExp,
    Normal,
}
impl FloatFormat {
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
