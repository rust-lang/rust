use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sym;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_span::Symbol;

pub fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    arith_lhs: &hir::Expr<'_>,
    arith_rhs: &hir::Expr<'_>,
    unwrap_arg: &hir::Expr<'_>,
    arith: Symbol,
) {
    let ty = cx.typeck_results().expr_ty(arith_lhs);
    if !ty.is_integral() {
        return;
    }

    let Some(mm) = is_min_or_max(cx, unwrap_arg) else {
        return;
    };

    let Some(checked_arith) = CheckedArith::new(arith) else {
        return;
    };

    {
        use self::MinMax::{Max, Min};
        use self::Sign::{Neg, Pos};
        use CheckedArith::{Add, Mul, Sub};

        if ty.is_signed() {
            let Some(sign) = lit_sign(arith_rhs) else {
                return;
            };

            match (&checked_arith, sign, mm) {
                (Add, Pos, Max) | (Add, Neg, Min) | (Sub, Neg, Max) | (Sub, Pos, Min) => (),
                // "mul" is omitted because lhs can be negative.
                _ => return,
            }
        } else {
            match (mm, &checked_arith) {
                (Max, Add | Mul) | (Min, Sub) => (),
                _ => return,
            }
        }
    }

    let mut applicability = Applicability::MachineApplicable;
    let saturating_arith = checked_arith.as_saturating();
    span_lint_and_sugg(
        cx,
        super::MANUAL_SATURATING_ARITHMETIC,
        expr.span,
        "manual saturating arithmetic",
        format!("consider using `{saturating_arith}`"),
        format!(
            "{}.{saturating_arith}({})",
            snippet_with_applicability(cx, arith_lhs.span, "..", &mut applicability),
            snippet_with_applicability(cx, arith_rhs.span, "..", &mut applicability),
        ),
        applicability,
    );
}

enum CheckedArith {
    Add,
    Sub,
    Mul,
}

impl CheckedArith {
    fn new(sym: Symbol) -> Option<Self> {
        let res = match sym {
            sym::checked_add => Self::Add,
            sym::checked_sub => Self::Sub,
            sym::checked_mul => Self::Mul,
            _ => return None,
        };
        Some(res)
    }

    fn as_saturating(&self) -> &'static str {
        match self {
            Self::Add => "saturating_add",
            Self::Sub => "saturating_sub",
            Self::Mul => "saturating_mul",
        }
    }
}

#[derive(PartialEq, Eq)]
enum MinMax {
    Min,
    Max,
}

fn is_min_or_max(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<MinMax> {
    // `T::max_value()` `T::min_value()` inherent methods
    if let hir::ExprKind::Call(func, []) = &expr.kind
        && let hir::ExprKind::Path(hir::QPath::TypeRelative(_, segment)) = &func.kind
    {
        match segment.ident.name {
            sym::max_value => return Some(MinMax::Max),
            sym::min_value => return Some(MinMax::Min),
            _ => {},
        }
    }

    let ty = cx.typeck_results().expr_ty(expr);

    // `T::MAX` and `T::MIN` constants
    if let hir::ExprKind::Path(hir::QPath::TypeRelative(base, seg)) = expr.kind
        && matches!(base.basic_res(), Res::PrimTy(_))
    {
        match seg.ident.name {
            sym::MAX => return Some(MinMax::Max),
            sym::MIN => return Some(MinMax::Min),
            _ => {},
        }
    }

    // Literals
    let bits = cx.layout_of(ty).unwrap().size.bits();
    let (minval, maxval): (u128, u128) = if ty.is_signed() {
        let minval = 1 << (bits - 1);
        let mut maxval = !(1 << (bits - 1));
        if bits != 128 {
            maxval &= (1 << bits) - 1;
        }
        (minval, maxval)
    } else {
        (0, if bits == 128 { !0 } else { (1 << bits) - 1 })
    };

    let check_lit = |expr: &hir::Expr<'_>, check_min: bool| {
        if let hir::ExprKind::Lit(lit) = &expr.kind
            && let ast::LitKind::Int(value, _) = lit.node
        {
            if value == maxval {
                return Some(MinMax::Max);
            }

            if check_min && value == minval {
                return Some(MinMax::Min);
            }
        }

        None
    };

    if let r @ Some(_) = check_lit(expr, !ty.is_signed()) {
        return r;
    }

    if ty.is_signed()
        && let hir::ExprKind::Unary(hir::UnOp::Neg, val) = &expr.kind
    {
        return check_lit(val, true);
    }

    None
}

#[derive(PartialEq, Eq)]
enum Sign {
    Pos,
    Neg,
}

fn lit_sign(expr: &hir::Expr<'_>) -> Option<Sign> {
    if let hir::ExprKind::Unary(hir::UnOp::Neg, inner) = &expr.kind {
        if let hir::ExprKind::Lit(..) = &inner.kind {
            return Some(Sign::Neg);
        }
    } else if let hir::ExprKind::Lit(..) = &expr.kind {
        return Some(Sign::Pos);
    }

    None
}
