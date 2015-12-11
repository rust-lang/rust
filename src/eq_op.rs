use rustc::lint::*;
use rustc_front::hir::*;
use rustc_front::util as ast_util;
use syntax::ptr::P;

use consts::constant;
use utils::span_lint;

/// **What it does:** This lint checks for equal operands to comparisons and bitwise binary operators (`&`, `|` and `^`). It is `Warn` by default.
///
/// **Why is this bad?** This is usually just a typo.
///
/// **Known problems:** False negatives: We had some false positives regarding calls (notably [racer](https://github.com/phildawes/racer) had one instance of `x.pop() && x.pop()`), so we removed matching any function or method calls. We may introduce a whitelist of known pure functions in the future.
///
/// **Example:** `x + 1 == x + 1`
declare_lint! {
    pub EQ_OP,
    Warn,
    "equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)"
}

#[derive(Copy,Clone)]
pub struct EqOp;

impl LintPass for EqOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(EQ_OP)
    }
}

impl LateLintPass for EqOp {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprBinary(ref op, ref left, ref right) = e.node {
            if is_cmp_or_bit(op) && is_exp_equal(cx, left, right) {
                span_lint(cx, EQ_OP, e.span, &format!(
                    "equal expressions as operands to {}",
                        ast_util::binop_to_string(op.node)));
            }
        }
    }
}

pub fn is_exp_equal(cx: &LateContext, left : &Expr, right : &Expr) -> bool {
    if let (Some(l), Some(r)) = (constant(cx, left), constant(cx, right)) {
        if l == r {
            return true;
        }
    }
    match (&left.node, &right.node) {
        (&ExprField(ref lfexp, ref lfident),
                &ExprField(ref rfexp, ref rfident)) =>
            lfident.node == rfident.node && is_exp_equal(cx, lfexp, rfexp),
        (&ExprLit(ref l), &ExprLit(ref r)) => l.node == r.node,
        (&ExprPath(ref lqself, ref lsubpath),
                &ExprPath(ref rqself, ref rsubpath)) =>
            both(lqself, rqself, is_qself_equal) &&
                is_path_equal(lsubpath, rsubpath),
        (&ExprTup(ref ltup), &ExprTup(ref rtup)) =>
            is_exps_equal(cx, ltup, rtup),
        (&ExprVec(ref l), &ExprVec(ref r)) => is_exps_equal(cx, l, r),
        (&ExprCast(ref lx, ref lt), &ExprCast(ref rx, ref rt)) =>
            is_exp_equal(cx, lx, rx) && is_cast_ty_equal(lt, rt),
        _ => false
    }
}

fn is_exps_equal(cx: &LateContext, left : &[P<Expr>], right : &[P<Expr>]) -> bool {
    over(left, right, |l, r| is_exp_equal(cx, l, r))
}

fn is_path_equal(left : &Path, right : &Path) -> bool {
    // The == of idents doesn't work with different contexts,
    // we have to be explicit about hygiene
    left.global == right.global && over(&left.segments, &right.segments,
        |l, r| l.identifier.name == r.identifier.name
              && l.identifier.ctxt == r.identifier.ctxt
               && l.parameters == r.parameters)
}

fn is_qself_equal(left : &QSelf, right : &QSelf) -> bool {
    left.ty.node == right.ty.node && left.position == right.position
}

fn over<X, F>(left: &[X], right: &[X], mut eq_fn: F) -> bool
        where F: FnMut(&X, &X) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)|
        eq_fn(x, y))
}

fn both<X, F>(l: &Option<X>, r: &Option<X>, mut eq_fn : F) -> bool
        where F: FnMut(&X, &X) -> bool {
    l.as_ref().map_or_else(|| r.is_none(), |x| r.as_ref().map_or(false,
        |y| eq_fn(x, y)))
}

fn is_cmp_or_bit(op : &BinOp) -> bool {
    match op.node {
        BiEq | BiLt | BiLe | BiGt | BiGe | BiNe | BiAnd | BiOr |
        BiBitXor | BiBitAnd | BiBitOr => true,
        _ => false
    }
}

fn is_cast_ty_equal(left: &Ty, right: &Ty) -> bool {
    match (&left.node, &right.node) {
        (&TyVec(ref lvec), &TyVec(ref rvec)) => is_cast_ty_equal(lvec, rvec),
        (&TyPtr(ref lmut), &TyPtr(ref rmut)) =>
            lmut.mutbl == rmut.mutbl &&
            is_cast_ty_equal(&*lmut.ty, &*rmut.ty),
        (&TyRptr(_, ref lrmut), &TyRptr(_, ref rrmut)) =>
            lrmut.mutbl == rrmut.mutbl &&
            is_cast_ty_equal(&*lrmut.ty, &*rrmut.ty),
        (&TyPath(ref lq, ref lpath), &TyPath(ref rq, ref rpath)) =>
            both(lq, rq, is_qself_equal) && is_path_equal(lpath, rpath),
        (&TyInfer, &TyInfer) => true,
        _ => false
    }
}
