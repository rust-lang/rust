//! lint on manually implemented checked conversions that could be transformed into `try_from`

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{in_constant, is_integer_literal, SpanlessEq};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOp, BinOpKind, Expr, ExprKind, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit bounds checking when casting.
    ///
    /// ### Why is this bad?
    /// Reduces the readability of statements & is error prone.
    ///
    /// ### Example
    /// ```rust
    /// # let foo: u32 = 5;
    /// foo <= i32::MAX as u32;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let foo = 1;
    /// # #[allow(unused)]
    /// i32::try_from(foo).is_ok();
    /// ```
    #[clippy::version = "1.37.0"]
    pub CHECKED_CONVERSIONS,
    pedantic,
    "`try_from` could replace manual bounds checking when casting"
}

pub struct CheckedConversions {
    msrv: Msrv,
}

impl CheckedConversions {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(CheckedConversions => [CHECKED_CONVERSIONS]);

impl<'tcx> LateLintPass<'tcx> for CheckedConversions {
    fn check_expr(&mut self, cx: &LateContext<'_>, item: &Expr<'_>) {
        if !self.msrv.meets(msrvs::TRY_FROM) {
            return;
        }

        let result = if_chain! {
            if !in_constant(cx, item.hir_id);
            if !in_external_macro(cx.sess(), item.span);
            if let ExprKind::Binary(op, left, right) = &item.kind;

            then {
                match op.node {
                    BinOpKind::Ge | BinOpKind::Le => single_check(item),
                    BinOpKind::And => double_check(cx, left, right),
                    _ => None,
                }
            } else {
                None
            }
        };

        if let Some(cv) = result {
            if let Some(to_type) = cv.to_type {
                let mut applicability = Applicability::MachineApplicable;
                let snippet = snippet_with_applicability(cx, cv.expr_to_cast.span, "_", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    CHECKED_CONVERSIONS,
                    item.span,
                    "checked cast can be simplified",
                    "try",
                    format!("{to_type}::try_from({snippet}).is_ok()"),
                    applicability,
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

/// Searches for a single check from unsigned to _ is done
/// todo: check for case signed -> larger unsigned == only x >= 0
fn single_check<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<Conversion<'tcx>> {
    check_upper_bound(expr).filter(|cv| cv.cvt == ConversionType::FromUnsigned)
}

/// Searches for a combination of upper & lower bound checks
fn double_check<'a>(cx: &LateContext<'_>, left: &'a Expr<'_>, right: &'a Expr<'_>) -> Option<Conversion<'a>> {
    let upper_lower = |l, r| {
        let upper = check_upper_bound(l);
        let lower = check_lower_bound(r);

        upper.zip(lower).and_then(|(l, r)| l.combine(r, cx))
    };

    upper_lower(left, right).or_else(|| upper_lower(right, left))
}

/// Contains the result of a tried conversion check
#[derive(Clone, Debug)]
struct Conversion<'a> {
    cvt: ConversionType,
    expr_to_cast: &'a Expr<'a>,
    to_type: Option<&'a str>,
}

/// The kind of conversion that is checked
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConversionType {
    SignedToUnsigned,
    SignedToSigned,
    FromUnsigned,
}

impl<'a> Conversion<'a> {
    /// Combine multiple conversions if the are compatible
    pub fn combine(self, other: Self, cx: &LateContext<'_>) -> Option<Conversion<'a>> {
        if self.is_compatible(&other, cx) {
            // Prefer a Conversion that contains a type-constraint
            Some(if self.to_type.is_some() { self } else { other })
        } else {
            None
        }
    }

    /// Checks if two conversions are compatible
    /// same type of conversion, same 'castee' and same 'to type'
    pub fn is_compatible(&self, other: &Self, cx: &LateContext<'_>) -> bool {
        (self.cvt == other.cvt)
            && (SpanlessEq::new(cx).eq_expr(self.expr_to_cast, other.expr_to_cast))
            && (self.has_compatible_to_type(other))
    }

    /// Checks if the to-type is the same (if there is a type constraint)
    fn has_compatible_to_type(&self, other: &Self) -> bool {
        match (self.to_type, other.to_type) {
            (Some(l), Some(r)) => l == r,
            _ => true,
        }
    }

    /// Try to construct a new conversion if the conversion type is valid
    fn try_new(expr_to_cast: &'a Expr<'_>, from_type: &str, to_type: &'a str) -> Option<Conversion<'a>> {
        ConversionType::try_new(from_type, to_type).map(|cvt| Conversion {
            cvt,
            expr_to_cast,
            to_type: Some(to_type),
        })
    }

    /// Construct a new conversion without type constraint
    fn new_any(expr_to_cast: &'a Expr<'_>) -> Conversion<'a> {
        Conversion {
            cvt: ConversionType::SignedToUnsigned,
            expr_to_cast,
            to_type: None,
        }
    }
}

impl ConversionType {
    /// Creates a conversion type if the type is allowed & conversion is valid
    #[must_use]
    fn try_new(from: &str, to: &str) -> Option<Self> {
        if UINTS.contains(&from) {
            Some(Self::FromUnsigned)
        } else if SINTS.contains(&from) {
            if UINTS.contains(&to) {
                Some(Self::SignedToUnsigned)
            } else if SINTS.contains(&to) {
                Some(Self::SignedToSigned)
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Check for `expr <= (to_type::MAX as from_type)`
fn check_upper_bound<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<Conversion<'tcx>> {
    if_chain! {
         if let ExprKind::Binary(ref op, left, right) = &expr.kind;
         if let Some((candidate, check)) = normalize_le_ge(op, left, right);
         if let Some((from, to)) = get_types_from_cast(check, INTS, "max_value", "MAX");

         then {
             Conversion::try_new(candidate, from, to)
         } else {
            None
        }
    }
}

/// Check for `expr >= 0|(to_type::MIN as from_type)`
fn check_lower_bound<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<Conversion<'tcx>> {
    fn check_function<'a>(candidate: &'a Expr<'a>, check: &'a Expr<'a>) -> Option<Conversion<'a>> {
        (check_lower_bound_zero(candidate, check)).or_else(|| (check_lower_bound_min(candidate, check)))
    }

    // First of we need a binary containing the expression & the cast
    if let ExprKind::Binary(ref op, left, right) = &expr.kind {
        normalize_le_ge(op, right, left).and_then(|(l, r)| check_function(l, r))
    } else {
        None
    }
}

/// Check for `expr >= 0`
fn check_lower_bound_zero<'a>(candidate: &'a Expr<'_>, check: &'a Expr<'_>) -> Option<Conversion<'a>> {
    is_integer_literal(check, 0).then(|| Conversion::new_any(candidate))
}

/// Check for `expr >= (to_type::MIN as from_type)`
fn check_lower_bound_min<'a>(candidate: &'a Expr<'_>, check: &'a Expr<'_>) -> Option<Conversion<'a>> {
    if let Some((from, to)) = get_types_from_cast(check, SINTS, "min_value", "MIN") {
        Conversion::try_new(candidate, from, to)
    } else {
        None
    }
}

/// Tries to extract the from- and to-type from a cast expression
fn get_types_from_cast<'a>(
    expr: &'a Expr<'_>,
    types: &'a [&str],
    func: &'a str,
    assoc_const: &'a str,
) -> Option<(&'a str, &'a str)> {
    // `to_type::max_value() as from_type`
    // or `to_type::MAX as from_type`
    let call_from_cast: Option<(&Expr<'_>, &str)> = if_chain! {
        // to_type::max_value(), from_type
        if let ExprKind::Cast(limit, from_type) = &expr.kind;
        if let TyKind::Path(ref from_type_path) = &from_type.kind;
        if let Some(from_sym) = int_ty_to_sym(from_type_path);

        then {
            Some((limit, from_sym))
        } else {
            None
        }
    };

    // `from_type::from(to_type::max_value())`
    let limit_from: Option<(&Expr<'_>, &str)> = call_from_cast.or_else(|| {
        if_chain! {
            // `from_type::from, to_type::max_value()`
            if let ExprKind::Call(from_func, [limit]) = &expr.kind;
            // `from_type::from`
            if let ExprKind::Path(ref path) = &from_func.kind;
            if let Some(from_sym) = get_implementing_type(path, INTS, "from");

            then {
                Some((limit, from_sym))
            } else {
                None
            }
        }
    });

    if let Some((limit, from_type)) = limit_from {
        match limit.kind {
            // `from_type::from(_)`
            ExprKind::Call(path, _) => {
                if let ExprKind::Path(ref path) = path.kind {
                    // `to_type`
                    if let Some(to_type) = get_implementing_type(path, types, func) {
                        return Some((from_type, to_type));
                    }
                }
            },
            // `to_type::MAX`
            ExprKind::Path(ref path) => {
                if let Some(to_type) = get_implementing_type(path, types, assoc_const) {
                    return Some((from_type, to_type));
                }
            },
            _ => {},
        }
    };
    None
}

/// Gets the type which implements the called function
fn get_implementing_type<'a>(path: &QPath<'_>, candidates: &'a [&str], function: &str) -> Option<&'a str> {
    if_chain! {
        if let QPath::TypeRelative(ty, path) = &path;
        if path.ident.name.as_str() == function;
        if let TyKind::Path(QPath::Resolved(None, tp)) = &ty.kind;
        if let [int] = tp.segments;
        then {
            let name = int.ident.name.as_str();
            candidates.iter().find(|c| &name == *c).copied()
        } else {
            None
        }
    }
}

/// Gets the type as a string, if it is a supported integer
fn int_ty_to_sym<'tcx>(path: &QPath<'_>) -> Option<&'tcx str> {
    if_chain! {
        if let QPath::Resolved(_, path) = *path;
        if let [ty] = path.segments;
        then {
            let name = ty.ident.name.as_str();
            INTS.iter().find(|c| &name == *c).copied()
        } else {
            None
        }
    }
}

/// Will return the expressions as if they were expr1 <= expr2
fn normalize_le_ge<'a>(op: &BinOp, left: &'a Expr<'a>, right: &'a Expr<'a>) -> Option<(&'a Expr<'a>, &'a Expr<'a>)> {
    match op.node {
        BinOpKind::Le => Some((left, right)),
        BinOpKind::Ge => Some((right, left)),
        _ => None,
    }
}

// Constants
const UINTS: &[&str] = &["u8", "u16", "u32", "u64", "usize"];
const SINTS: &[&str] = &["i8", "i16", "i32", "i64", "isize"];
const INTS: &[&str] = &["u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize"];
