use rustc::hir;
use rustc::lint::*;
use rustc::ty::TypeVariants;
use std::f32;
use std::f64;
use std::fmt;
use syntax::ast::*;
use syntax_pos::symbol::Symbol;
use utils::span_lint_and_sugg;

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
/// Insert a short example of code that triggers the lint
///    let v: f32 = 0.123_456_789_9;
///    println!("{}", v); //  0.123_456_789
///
/// // Good
/// Insert a short example of improved code that doesn't trigger the lint
///    let v: f64 = 0.123_456_789_9;
///    println!("{}", v); //  0.123_456_789_9
/// ```
declare_clippy_lint! {
    pub EXCESSIVE_PRECISION,
    style,
    "excessive precision for float literal"
}

pub struct ExcessivePrecision;

impl LintPass for ExcessivePrecision {
    fn get_lints(&self) -> LintArray {
        lint_array!(EXCESSIVE_PRECISION)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ExcessivePrecision {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if_chain! {
            let ty = cx.tables.expr_ty(expr);
            if let TypeVariants::TyFloat(ref fty) = ty.sty;
            if let hir::ExprLit(ref lit) = expr.node;
            if let LitKind::Float(ref sym, _) | LitKind::FloatUnsuffixed(ref sym) = lit.node;
            if let Some(sugg) = self.check(sym, fty);
            then {
                span_lint_and_sugg(
                    cx,
                    EXCESSIVE_PRECISION,
                    expr.span,
                    "float has excessive precision",
                    "consider changing the type or truncating it to",
                    sugg,
                );
            }
        }
    }
}

impl ExcessivePrecision {
    // None if nothing to lint, Some(suggestion) if lint neccessary
    fn check(&self, sym: &Symbol, fty: &FloatTy) -> Option<String> {
        let max = max_digits(fty);
        let sym_str = sym.as_str();
        let formatter = FloatFormat::new(&sym_str);
        let digits = count_digits(&sym_str);
        // Try to bail out if the float is for sure fine.
        // If its within the 2 decimal digits of being out of precision we
        // check if the parsed representation is the same as the string
        // since we'll need the truncated string anyway.
        if digits > max as usize {
            let sr = match *fty {
                FloatTy::F32 => sym_str.parse::<f32>().map(|f| formatter.format(f)),
                FloatTy::F64 => sym_str.parse::<f64>().map(|f| formatter.format(f)),
            };
            // We know this will parse since we are in LatePass
            let s = sr.unwrap();

            if sym_str == s {
                None
            } else {
                Some(s)
            }
        } else {
            None
        }
    }
}

fn max_digits(fty: &FloatTy) -> u32 {
    match fty {
        FloatTy::F32 => f32::DIGITS,
        FloatTy::F64 => f64::DIGITS,
    }
}

fn count_digits(s: &str) -> usize {
    s.chars()
        .filter(|c| *c != '-' || *c != '.')
        .take_while(|c| *c != 'e' || *c != 'E')
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
                'e' => Some(FloatFormat::LowerExp),
                'E' => Some(FloatFormat::UpperExp),
                _ => None,
            })
            .unwrap_or(FloatFormat::Normal)
    }
    fn format<T>(&self, f: T) -> String
    where T: fmt::UpperExp + fmt::LowerExp + fmt::Display {
        match self {
            FloatFormat::LowerExp => format!("{:e}", f),
            FloatFormat::UpperExp => format!("{:E}", f),
            FloatFormat::Normal => format!("{}", f),
        }
    }
}
