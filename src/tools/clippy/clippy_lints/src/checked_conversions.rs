use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{SpanlessEq, is_in_const_context, is_integer_literal};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit bounds checking when casting.
    ///
    /// ### Why is this bad?
    /// Reduces the readability of statements & is error prone.
    ///
    /// ### Example
    /// ```no_run
    /// # let foo: u32 = 5;
    /// foo <= i32::MAX as u32;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv.clone(),
        }
    }
}

impl_lint_pass!(CheckedConversions => [CHECKED_CONVERSIONS]);

impl LateLintPass<'_> for CheckedConversions {
    fn check_expr(&mut self, cx: &LateContext<'_>, item: &Expr<'_>) {
        if let ExprKind::Binary(op, lhs, rhs) = item.kind
            && let (lt1, gt1, op2) = match op.node {
                BinOpKind::Le => (lhs, rhs, None),
                BinOpKind::Ge => (rhs, lhs, None),
                BinOpKind::And
                    if let ExprKind::Binary(op1, lhs1, rhs1) = lhs.kind
                        && let ExprKind::Binary(op2, lhs2, rhs2) = rhs.kind
                        && let Some((lt1, gt1)) = read_le_ge(op1.node, lhs1, rhs1)
                        && let Some((lt2, gt2)) = read_le_ge(op2.node, lhs2, rhs2) =>
                {
                    (lt1, gt1, Some((lt2, gt2)))
                },
                _ => return,
            }
            && !in_external_macro(cx.sess(), item.span)
            && !is_in_const_context(cx)
            && self.msrv.meets(msrvs::TRY_FROM)
            && let Some(cv) = match op2 {
                // todo: check for case signed -> larger unsigned == only x >= 0
                None => check_upper_bound(lt1, gt1).filter(|cv| cv.cvt == ConversionType::FromUnsigned),
                Some((lt2, gt2)) => {
                    let upper_lower = |lt1, gt1, lt2, gt2| {
                        check_upper_bound(lt1, gt1)
                            .zip(check_lower_bound(lt2, gt2))
                            .and_then(|(l, r)| l.combine(r, cx))
                    };
                    upper_lower(lt1, gt1, lt2, gt2).or_else(|| upper_lower(lt2, gt2, lt1, gt1))
                },
            }
            && let Some(to_type) = cv.to_type
        {
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

    extract_msrv_attr!(LateContext);
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

/// Attempts to read either `<=` or `>=` with a normalized operand order.
fn read_le_ge<'tcx>(
    op: BinOpKind,
    lhs: &'tcx Expr<'tcx>,
    rhs: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    match op {
        BinOpKind::Le => Some((lhs, rhs)),
        BinOpKind::Ge => Some((rhs, lhs)),
        _ => None,
    }
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
fn check_upper_bound<'tcx>(lt: &'tcx Expr<'tcx>, gt: &'tcx Expr<'tcx>) -> Option<Conversion<'tcx>> {
    if let Some((from, to)) = get_types_from_cast(gt, INTS, "max_value", "MAX") {
        Conversion::try_new(lt, from, to)
    } else {
        None
    }
}

/// Check for `expr >= 0|(to_type::MIN as from_type)`
fn check_lower_bound<'tcx>(lt: &'tcx Expr<'tcx>, gt: &'tcx Expr<'tcx>) -> Option<Conversion<'tcx>> {
    check_lower_bound_zero(gt, lt).or_else(|| check_lower_bound_min(gt, lt))
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
    let call_from_cast: Option<(&Expr<'_>, &str)> = if let ExprKind::Cast(limit, from_type) = &expr.kind
        // to_type::max_value(), from_type
        && let TyKind::Path(from_type_path) = &from_type.kind
        && let Some(from_sym) = int_ty_to_sym(from_type_path)
    {
        Some((limit, from_sym))
    } else {
        None
    };

    // `from_type::from(to_type::max_value())`
    let limit_from: Option<(&Expr<'_>, &str)> = call_from_cast.or_else(|| {
        if let ExprKind::Call(from_func, [limit]) = &expr.kind
            // `from_type::from, to_type::max_value()`
            // `from_type::from`
            && let ExprKind::Path(path) = &from_func.kind
            && let Some(from_sym) = get_implementing_type(path, INTS, "from")
        {
            Some((limit, from_sym))
        } else {
            None
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
    if let QPath::TypeRelative(ty, path) = &path
        && path.ident.name.as_str() == function
        && let TyKind::Path(QPath::Resolved(None, tp)) = &ty.kind
        && let [int] = tp.segments
    {
        let name = int.ident.name.as_str();
        candidates.iter().find(|c| &name == *c).copied()
    } else {
        None
    }
}

/// Gets the type as a string, if it is a supported integer
fn int_ty_to_sym<'tcx>(path: &QPath<'_>) -> Option<&'tcx str> {
    if let QPath::Resolved(_, path) = *path
        && let [ty] = path.segments
    {
        let name = ty.ident.name.as_str();
        INTS.iter().find(|c| &name == *c).copied()
    } else {
        None
    }
}

// Constants
const UINTS: &[&str] = &["u8", "u16", "u32", "u64", "usize"];
const SINTS: &[&str] = &["i8", "i16", "i32", "i64", "isize"];
const INTS: &[&str] = &["u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize"];
