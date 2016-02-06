use consts::constant;
use rustc::lint::*;
use rustc_front::hir::*;
use syntax::ptr::P;

/// Type used to check whether two ast are the same. This is different from the operator
/// `==` on ast types as this operator would compare true equality with ID and span.
///
/// Note that some expressions kinds are not considered but could be added.
pub struct SpanlessEq<'a, 'tcx: 'a> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'a, 'tcx>,
    /// If is true, never consider as equal expressions containing fonction calls.
    ignore_fn: bool,
}

impl<'a, 'tcx: 'a> SpanlessEq<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        SpanlessEq { cx: cx, ignore_fn: false }
    }

    pub fn ignore_fn(self) -> Self {
        SpanlessEq { cx: self.cx, ignore_fn: true }
    }

    /// Check whether two statements are the same.
    pub fn eq_stmt(&self, left: &Stmt, right: &Stmt) -> bool {
        match (&left.node, &right.node) {
            (&StmtDecl(ref l, _), &StmtDecl(ref r, _)) => {
                if let (&DeclLocal(ref l), &DeclLocal(ref r)) = (&l.node, &r.node) {
                    // TODO: tys
                    l.ty.is_none() && r.ty.is_none() &&
                        both(&l.init, &r.init, |l, r| self.eq_expr(l, r))
                }
                else {
                    false
                }
            }
            (&StmtExpr(ref l, _), &StmtExpr(ref r, _)) => self.eq_expr(l, r),
            (&StmtSemi(ref l, _), &StmtSemi(ref r, _)) => self.eq_expr(l, r),
            _ => false,
        }
    }

    /// Check whether two blocks are the same.
    pub fn eq_block(&self, left: &Block, right: &Block) -> bool {
        over(&left.stmts, &right.stmts, |l, r| self.eq_stmt(l, r)) &&
            both(&left.expr, &right.expr, |l, r| self.eq_expr(l, r))
    }

    // ok, it’s a big function, but mostly one big match with simples cases
    #[allow(cyclomatic_complexity)]
    pub fn eq_expr(&self, left: &Expr, right: &Expr) -> bool {
        if let (Some(l), Some(r)) = (constant(self.cx, left), constant(self.cx, right)) {
            if l == r {
                return true;
            }
        }

        match (&left.node, &right.node) {
            (&ExprAddrOf(ref lmut, ref le), &ExprAddrOf(ref rmut, ref re)) => {
                lmut == rmut && self.eq_expr(le, re)
            }
            (&ExprAgain(li), &ExprAgain(ri)) => {
                both(&li, &ri, |l, r| l.node.name.as_str() == r.node.name.as_str())
            }
            (&ExprAssign(ref ll, ref lr), &ExprAssign(ref rl, ref rr)) => {
                self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            }
            (&ExprAssignOp(ref lo, ref ll, ref lr), &ExprAssignOp(ref ro, ref rl, ref rr)) => {
                lo.node == ro.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            }
            (&ExprBlock(ref l), &ExprBlock(ref r)) => {
                self.eq_block(l, r)
            }
            (&ExprBinary(lop, ref ll, ref lr), &ExprBinary(rop, ref rl, ref rr)) => {
                lop.node == rop.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            }
            (&ExprBreak(li), &ExprBreak(ri)) => {
                both(&li, &ri, |l, r| l.node.name.as_str() == r.node.name.as_str())
            }
            (&ExprBox(ref l), &ExprBox(ref r)) => {
                self.eq_expr(l, r)
            }
            (&ExprCall(ref lfun, ref largs), &ExprCall(ref rfun, ref rargs)) => {
                !self.ignore_fn &&
                    self.eq_expr(lfun, rfun) &&
                    self.eq_exprs(largs, rargs)
            }
            (&ExprCast(ref lx, ref lt), &ExprCast(ref rx, ref rt)) => {
                self.eq_expr(lx, rx) && self.eq_ty(lt, rt)
            }
            (&ExprField(ref lfexp, ref lfident), &ExprField(ref rfexp, ref rfident)) => {
                lfident.node == rfident.node && self.eq_expr(lfexp, rfexp)
            }
            (&ExprIndex(ref la, ref li), &ExprIndex(ref ra, ref ri)) => {
                self.eq_expr(la, ra) && self.eq_expr(li, ri)
            }
            (&ExprIf(ref lc, ref lt, ref le), &ExprIf(ref rc, ref rt, ref re)) => {
                self.eq_expr(lc, rc) &&
                    self.eq_block(lt, rt) &&
                    both(le, re, |l, r| self.eq_expr(l, r))
            }
            (&ExprLit(ref l), &ExprLit(ref r)) => l.node == r.node,
            (&ExprMatch(ref le, ref la, ref ls), &ExprMatch(ref re, ref ra, ref rs)) => {
                ls == rs &&
                    self.eq_expr(le, re) &&
                    over(la, ra, |l, r| {
                        self.eq_expr(&l.body, &r.body) &&
                            both(&l.guard, &r.guard, |l, r| self.eq_expr(l, r)) &&
                            over(&l.pats, &r.pats, |l, r| self.eq_pat(l, r))
                    })
            }
            (&ExprMethodCall(ref lname, ref ltys, ref largs), &ExprMethodCall(ref rname, ref rtys, ref rargs)) => {
                // TODO: tys
                !self.ignore_fn &&
                    lname.node == rname.node &&
                    ltys.is_empty() &&
                    rtys.is_empty() &&
                    self.eq_exprs(largs, rargs)
            }
            (&ExprRange(ref lb, ref le), &ExprRange(ref rb, ref re)) => {
                both(lb, rb, |l, r| self.eq_expr(l, r)) &&
                both(le, re, |l, r| self.eq_expr(l, r))
            }
            (&ExprRepeat(ref le, ref ll), &ExprRepeat(ref re, ref rl)) => {
                self.eq_expr(le, re) && self.eq_expr(ll, rl)
            }
            (&ExprRet(ref l), &ExprRet(ref r)) => {
                both(l, r, |l, r| self.eq_expr(l, r))
            }
            (&ExprPath(ref lqself, ref lsubpath), &ExprPath(ref rqself, ref rsubpath)) => {
                both(lqself, rqself, |l, r| self.eq_qself(l, r)) && self.eq_path(lsubpath, rsubpath)
            }
            (&ExprTup(ref ltup), &ExprTup(ref rtup)) => self.eq_exprs(ltup, rtup),
            (&ExprTupField(ref le, li), &ExprTupField(ref re, ri)) => {
                li.node == ri.node && self.eq_expr(le, re)
            }
            (&ExprUnary(lop, ref le), &ExprUnary(rop, ref re)) => {
                lop == rop && self.eq_expr(le, re)
            }
            (&ExprVec(ref l), &ExprVec(ref r)) => self.eq_exprs(l, r),
            (&ExprWhile(ref lc, ref lb, ref ll), &ExprWhile(ref rc, ref rb, ref rl)) => {
                self.eq_expr(lc, rc) &&
                    self.eq_block(lb, rb) &&
                    both(ll, rl, |l, r| l.name.as_str() == r.name.as_str())
            }
            _ => false,
        }
    }

    fn eq_exprs(&self, left: &[P<Expr>], right: &[P<Expr>]) -> bool {
        over(left, right, |l, r| self.eq_expr(l, r))
    }

    /// Check whether two patterns are the same.
    pub fn eq_pat(&self, left: &Pat, right: &Pat) -> bool {
        match (&left.node, &right.node) {
            (&PatBox(ref l), &PatBox(ref r)) => {
                self.eq_pat(l, r)
            }
            (&PatEnum(ref lp, ref la), &PatEnum(ref rp, ref ra)) => {
                self.eq_path(lp, rp) &&
                    both(la, ra, |l, r| {
                        over(l, r, |l, r| self.eq_pat(l, r))
                    })
            }
            (&PatIdent(ref lb, ref li, ref lp), &PatIdent(ref rb, ref ri, ref rp)) => {
                lb == rb && li.node.name.as_str() == ri.node.name.as_str() &&
                    both(lp, rp, |l, r| self.eq_pat(l, r))
            }
            (&PatLit(ref l), &PatLit(ref r)) => {
                self.eq_expr(l, r)
            }
            (&PatQPath(ref ls, ref lp), &PatQPath(ref rs, ref rp)) => {
                self.eq_qself(ls, rs) && self.eq_path(lp, rp)
            }
            (&PatTup(ref l), &PatTup(ref r)) => {
                over(l, r, |l, r| self.eq_pat(l, r))
            }
            (&PatRange(ref ls, ref le), &PatRange(ref rs, ref re)) => {
                self.eq_expr(ls, rs) &&
                    self.eq_expr(le, re)
            }
            (&PatRegion(ref le, ref lm), &PatRegion(ref re, ref rm)) => {
                lm == rm && self.eq_pat(le, re)
            }
            (&PatVec(ref ls, ref li, ref le), &PatVec(ref rs, ref ri, ref re)) => {
                over(ls, rs, |l, r| self.eq_pat(l, r)) &&
                    over(le, re, |l, r| self.eq_pat(l, r)) &&
                    both(li, ri, |l, r| self.eq_pat(l, r))
            }
            (&PatWild, &PatWild) => true,
            _ => false,
        }
    }

    fn eq_path(&self, left: &Path, right: &Path) -> bool {
        // The == of idents doesn't work with different contexts,
        // we have to be explicit about hygiene
        left.global == right.global &&
        over(&left.segments,
             &right.segments,
             |l, r| l.identifier.name.as_str() == r.identifier.name.as_str() && l.parameters == r.parameters)
    }

    fn eq_qself(&self, left: &QSelf, right: &QSelf) -> bool {
        left.ty.node == right.ty.node && left.position == right.position
    }

    fn eq_ty(&self, left: &Ty, right: &Ty) -> bool {
        match (&left.node, &right.node) {
            (&TyVec(ref lvec), &TyVec(ref rvec)) => self.eq_ty(lvec, rvec),
            (&TyPtr(ref lmut), &TyPtr(ref rmut)) => lmut.mutbl == rmut.mutbl && self.eq_ty(&*lmut.ty, &*rmut.ty),
            (&TyRptr(_, ref lrmut), &TyRptr(_, ref rrmut)) => {
                lrmut.mutbl == rrmut.mutbl && self.eq_ty(&*lrmut.ty, &*rrmut.ty)
            }
            (&TyPath(ref lq, ref lpath), &TyPath(ref rq, ref rpath)) => {
                both(lq, rq, |l, r| self.eq_qself(l, r)) && self.eq_path(lpath, rpath)
            }
            (&TyInfer, &TyInfer) => true,
            _ => false,
        }
    }
}

/// Check if the two `Option`s are both `None` or some equal values as per `eq_fn`.
fn both<X, F>(l: &Option<X>, r: &Option<X>, mut eq_fn: F) -> bool
    where F: FnMut(&X, &X) -> bool
{
    l.as_ref().map_or_else(|| r.is_none(), |x| r.as_ref().map_or(false, |y| eq_fn(x, y)))
}

/// Check if two slices are equal as per `eq_fn`.
fn over<X, F>(left: &[X], right: &[X], mut eq_fn: F) -> bool
    where F: FnMut(&X, &X) -> bool
{
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| eq_fn(x, y))
}
