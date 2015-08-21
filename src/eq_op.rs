use rustc::lint::*;
use syntax::ast::*;
use syntax::ast_util as ast_util;
use syntax::ptr::P;
use syntax::codemap as code;

use consts::constant;
use utils::span_lint;

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

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprBinary(ref op, ref left, ref right) = e.node {
            if is_cmp_or_bit(op) && is_exp_equal(cx, left, right) {
                span_lint(cx, EQ_OP, e.span, &format!(
                    "equal expressions as operands to {}",
                        ast_util::binop_to_string(op.node)));
            }
        }
    }
}

pub fn is_exp_equal(cx: &Context, left : &Expr, right : &Expr) -> bool {
    match (&left.node, &right.node) {
        (&ExprBinary(ref lop, ref ll, ref lr),
                &ExprBinary(ref rop, ref rl, ref rr)) =>
            lop.node == rop.node &&
            is_exp_equal(cx, ll, rl) && is_exp_equal(cx, lr, rr),
        (&ExprBox(ref lpl, ref lbox), &ExprBox(ref rpl, ref rbox)) =>
            both(lpl, rpl, |l, r| is_exp_equal(cx, l, r)) &&
                is_exp_equal(cx, lbox, rbox),
        (&ExprCall(ref lcallee, ref largs),
         &ExprCall(ref rcallee, ref rargs)) => is_exp_equal(cx, lcallee,
            rcallee) && is_exps_equal(cx, largs, rargs),
        (&ExprCast(ref lc, ref lty), &ExprCast(ref rc, ref rty)) =>
            is_ty_equal(cx, lty, rty) && is_exp_equal(cx, lc, rc),
        (&ExprField(ref lfexp, ref lfident),
                &ExprField(ref rfexp, ref rfident)) =>
            lfident.node == rfident.node && is_exp_equal(cx, lfexp, rfexp),
        (&ExprLit(ref l), &ExprLit(ref r)) => l.node == r.node,
        (&ExprMethodCall(ref lident, ref lcty, ref lmargs),
                &ExprMethodCall(ref rident, ref rcty, ref rmargs)) =>
            lident.node == rident.node && is_tys_equal(cx, lcty, rcty) &&
                is_exps_equal(cx, lmargs, rmargs),
        (&ExprParen(ref lparen), _) => is_exp_equal(cx, lparen, right),
        (_, &ExprParen(ref rparen)) => is_exp_equal(cx, left, rparen),
        (&ExprPath(ref lqself, ref lsubpath),
                &ExprPath(ref rqself, ref rsubpath)) =>
            both(lqself, rqself, |l, r| is_qself_equal(l, r)) &&
                is_path_equal(lsubpath, rsubpath),
        (&ExprTup(ref ltup), &ExprTup(ref rtup)) =>
            is_exps_equal(cx, ltup, rtup),
        (&ExprUnary(lunop, ref l), &ExprUnary(runop, ref r)) =>
            lunop == runop && is_exp_equal(cx, l, r),
        (&ExprVec(ref l), &ExprVec(ref r)) => is_exps_equal(cx, l, r),
        _ => false
    } || match (constant(cx, left), constant(cx, right)) {
        (Some(l), Some(r)) => l == r,
        _ => false
    }
}

fn is_exps_equal(cx: &Context, left : &[P<Expr>], right : &[P<Expr>]) -> bool {
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

fn is_ty_equal(cx: &Context, left : &Ty, right : &Ty) -> bool {
    match (&left.node, &right.node) {
    (&TyVec(ref lvec), &TyVec(ref rvec)) => is_ty_equal(cx, lvec, rvec),
    (&TyFixedLengthVec(ref lfvty, ref lfvexp),
            &TyFixedLengthVec(ref rfvty, ref rfvexp)) =>
        is_ty_equal(cx, lfvty, rfvty) && is_exp_equal(cx, lfvexp, rfvexp),
    (&TyPtr(ref lmut), &TyPtr(ref rmut)) => is_mut_ty_equal(cx, lmut, rmut),
    (&TyRptr(ref ltime, ref lrmut), &TyRptr(ref rtime, ref rrmut)) =>
        both(ltime, rtime, is_lifetime_equal) &&
        is_mut_ty_equal(cx, lrmut, rrmut),
    (&TyBareFn(ref lbare), &TyBareFn(ref rbare)) =>
        is_bare_fn_ty_equal(cx, lbare, rbare),
    (&TyTup(ref ltup), &TyTup(ref rtup)) => is_tys_equal(cx, ltup, rtup),
    (&TyPath(ref lq, ref lpath), &TyPath(ref rq, ref rpath)) =>
        both(lq, rq, is_qself_equal) && is_path_equal(lpath, rpath),
    (&TyObjectSum(ref lsumty, ref lobounds),
            &TyObjectSum(ref rsumty, ref robounds)) =>
        is_ty_equal(cx, lsumty, rsumty) &&
        is_param_bounds_equal(lobounds, robounds),
    (&TyPolyTraitRef(ref ltbounds), &TyPolyTraitRef(ref rtbounds)) =>
        is_param_bounds_equal(ltbounds, rtbounds),
    (&TyParen(ref lty), &TyParen(ref rty)) => is_ty_equal(cx, lty, rty),
    (&TyTypeof(ref lof), &TyTypeof(ref rof)) => is_exp_equal(cx, lof, rof),
    (&TyInfer, &TyInfer) => true,
    _ => false
    }
}

fn is_param_bound_equal(left : &TyParamBound, right : &TyParamBound)
        -> bool {
    match(left, right) {
    (&TraitTyParamBound(ref lpoly, ref lmod),
            &TraitTyParamBound(ref rpoly, ref rmod)) =>
        lmod == rmod && is_poly_traitref_equal(lpoly, rpoly),
    (&RegionTyParamBound(ref ltime), &RegionTyParamBound(ref rtime)) =>
        is_lifetime_equal(ltime, rtime),
    _ => false
    }
}

fn is_poly_traitref_equal(left : &PolyTraitRef, right : &PolyTraitRef)
        -> bool {
    is_lifetimedefs_equal(&left.bound_lifetimes, &right.bound_lifetimes)
        && is_path_equal(&left.trait_ref.path, &right.trait_ref.path)
}

fn is_param_bounds_equal(left : &TyParamBounds, right : &TyParamBounds)
        -> bool {
    over(left, right, is_param_bound_equal)
}

fn is_mut_ty_equal(cx: &Context, left : &MutTy, right : &MutTy) -> bool {
    left.mutbl == right.mutbl && is_ty_equal(cx, &left.ty, &right.ty)
}

fn is_bare_fn_ty_equal(cx: &Context, left : &BareFnTy, right : &BareFnTy) -> bool {
    left.unsafety == right.unsafety && left.abi == right.abi &&
        is_lifetimedefs_equal(&left.lifetimes, &right.lifetimes) &&
            is_fndecl_equal(cx, &left.decl, &right.decl)
}

fn is_fndecl_equal(cx: &Context, left : &P<FnDecl>, right : &P<FnDecl>) -> bool {
    left.variadic == right.variadic &&
        is_args_equal(cx, &left.inputs, &right.inputs) &&
        is_fnret_ty_equal(cx, &left.output, &right.output)
}

fn is_fnret_ty_equal(cx: &Context, left : &FunctionRetTy,
        right : &FunctionRetTy) -> bool {
    match (left, right) {
    (&NoReturn(_), &NoReturn(_)) |
    (&DefaultReturn(_), &DefaultReturn(_)) => true,
    (&Return(ref lty), &Return(ref rty)) => is_ty_equal(cx, lty, rty),
    _ => false
    }
}

fn is_arg_equal(cx: &Context, l: &Arg, r : &Arg) -> bool {
    is_ty_equal(cx, &l.ty, &r.ty) && is_pat_equal(cx, &l.pat, &r.pat)
}

fn is_args_equal(cx: &Context, left : &[Arg], right : &[Arg]) -> bool {
    over(left, right, |l, r| is_arg_equal(cx, l, r))
}

fn is_pat_equal(cx: &Context, left : &Pat, right : &Pat) -> bool {
    match(&left.node, &right.node) {
    (&PatWild(lwild), &PatWild(rwild)) => lwild == rwild,
    (&PatIdent(ref lmode, ref lident, Option::None),
            &PatIdent(ref rmode, ref rident, Option::None)) =>
        lmode == rmode && is_ident_equal(&lident.node, &rident.node),
    (&PatIdent(ref lmode, ref lident, Option::Some(ref lpat)),
            &PatIdent(ref rmode, ref rident, Option::Some(ref rpat))) =>
        lmode == rmode && is_ident_equal(&lident.node, &rident.node) &&
            is_pat_equal(cx, lpat, rpat),
    (&PatEnum(ref lpath, ref lenum), &PatEnum(ref rpath, ref renum)) =>
        is_path_equal(lpath, rpath) && both(lenum, renum, |l, r|
            is_pats_equal(cx, l, r)),
    (&PatStruct(ref lpath, ref lfieldpat, lbool),
            &PatStruct(ref rpath, ref rfieldpat, rbool)) =>
        lbool == rbool && is_path_equal(lpath, rpath) &&
            is_spanned_fieldpats_equal(cx, lfieldpat, rfieldpat),
    (&PatTup(ref ltup), &PatTup(ref rtup)) => is_pats_equal(cx, ltup, rtup),
    (&PatBox(ref lboxed), &PatBox(ref rboxed)) =>
        is_pat_equal(cx, lboxed, rboxed),
    (&PatRegion(ref lpat, ref lmut), &PatRegion(ref rpat, ref rmut)) =>
        is_pat_equal(cx, lpat, rpat) && lmut == rmut,
    (&PatLit(ref llit), &PatLit(ref rlit)) => is_exp_equal(cx, llit, rlit),
    (&PatRange(ref lfrom, ref lto), &PatRange(ref rfrom, ref rto)) =>
        is_exp_equal(cx, lfrom, rfrom) && is_exp_equal(cx, lto, rto),
    (&PatVec(ref lfirst, Option::None, ref llast),
            &PatVec(ref rfirst, Option::None, ref rlast)) =>
        is_pats_equal(cx, lfirst, rfirst) && is_pats_equal(cx, llast, rlast),
    (&PatVec(ref lfirst, Option::Some(ref lpat), ref llast),
            &PatVec(ref rfirst, Option::Some(ref rpat), ref rlast)) =>
        is_pats_equal(cx, lfirst, rfirst) && is_pat_equal(cx, lpat, rpat) &&
            is_pats_equal(cx, llast, rlast),
    // I don't match macros for now, the code is slow enough as is ;-)
    _ => false
    }
}

fn is_spanned_fieldpats_equal(cx: &Context, left : &[code::Spanned<FieldPat>],
        right : &[code::Spanned<FieldPat>]) -> bool {
    over(left, right, |l, r| is_fieldpat_equal(cx, &l.node, &r.node))
}

fn is_fieldpat_equal(cx: &Context, left : &FieldPat, right : &FieldPat) -> bool {
    left.is_shorthand == right.is_shorthand &&
        is_ident_equal(&left.ident, &right.ident) &&
        is_pat_equal(cx, &left.pat, &right.pat)
}

fn is_ident_equal(left : &Ident, right : &Ident) -> bool {
    &left.name == &right.name && left.ctxt == right.ctxt
}

fn is_pats_equal(cx: &Context, left : &[P<Pat>], right : &[P<Pat>]) -> bool {
    over(left, right, |l, r| is_pat_equal(cx, l, r))
}

fn is_lifetimedef_equal(left : &LifetimeDef, right : &LifetimeDef)
        -> bool {
    is_lifetime_equal(&left.lifetime, &right.lifetime) &&
        over(&left.bounds, &right.bounds, is_lifetime_equal)
}

fn is_lifetimedefs_equal(left : &[LifetimeDef], right : &[LifetimeDef])
        -> bool {
    over(left, right, is_lifetimedef_equal)
}

fn is_lifetime_equal(left : &Lifetime, right : &Lifetime) -> bool {
    left.name == right.name
}

fn is_tys_equal(cx: &Context, left : &[P<Ty>], right : &[P<Ty>]) -> bool {
    over(left, right, |l, r| is_ty_equal(cx, l, r))
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
