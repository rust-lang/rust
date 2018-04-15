use consts::{constant_simple, constant_context};
use rustc::lint::*;
use rustc::hir::*;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use syntax::ast::Name;
use syntax::ptr::P;
use utils::differing_macro_contexts;

/// Type used to check whether two ast are the same. This is different from the
/// operator
/// `==` on ast types as this operator would compare true equality with ID and
/// span.
///
/// Note that some expressions kinds are not considered but could be added.
pub struct SpanlessEq<'a, 'tcx: 'a> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'a, 'tcx>,
    /// If is true, never consider as equal expressions containing function
    /// calls.
    ignore_fn: bool,
}

impl<'a, 'tcx: 'a> SpanlessEq<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self {
            cx,
            ignore_fn: false,
        }
    }

    pub fn ignore_fn(self) -> Self {
        Self {
            cx: self.cx,
            ignore_fn: true,
        }
    }

    /// Check whether two statements are the same.
    pub fn eq_stmt(&self, left: &Stmt, right: &Stmt) -> bool {
        match (&left.node, &right.node) {
            (&StmtDecl(ref l, _), &StmtDecl(ref r, _)) => {
                if let (&DeclLocal(ref l), &DeclLocal(ref r)) = (&l.node, &r.node) {
                    both(&l.ty, &r.ty, |l, r| self.eq_ty(l, r)) && both(&l.init, &r.init, |l, r| self.eq_expr(l, r))
                } else {
                    false
                }
            },
            (&StmtExpr(ref l, _), &StmtExpr(ref r, _)) | (&StmtSemi(ref l, _), &StmtSemi(ref r, _)) => {
                self.eq_expr(l, r)
            },
            _ => false,
        }
    }

    /// Check whether two blocks are the same.
    pub fn eq_block(&self, left: &Block, right: &Block) -> bool {
        over(&left.stmts, &right.stmts, |l, r| self.eq_stmt(l, r))
            && both(&left.expr, &right.expr, |l, r| self.eq_expr(l, r))
    }

    pub fn eq_expr(&self, left: &Expr, right: &Expr) -> bool {
        if self.ignore_fn && differing_macro_contexts(left.span, right.span) {
            return false;
        }

        if let (Some(l), Some(r)) = (constant_simple(self.cx, left), constant_simple(self.cx, right)) {
            if l == r {
                return true;
            }
        }

        match (&left.node, &right.node) {
            (&ExprAddrOf(l_mut, ref le), &ExprAddrOf(r_mut, ref re)) => l_mut == r_mut && self.eq_expr(le, re),
            (&ExprAgain(li), &ExprAgain(ri)) => {
                both(&li.label, &ri.label, |l, r| l.name.as_str() == r.name.as_str())
            },
            (&ExprAssign(ref ll, ref lr), &ExprAssign(ref rl, ref rr)) => self.eq_expr(ll, rl) && self.eq_expr(lr, rr),
            (&ExprAssignOp(ref lo, ref ll, ref lr), &ExprAssignOp(ref ro, ref rl, ref rr)) => {
                lo.node == ro.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            },
            (&ExprBlock(ref l), &ExprBlock(ref r)) => self.eq_block(l, r),
            (&ExprBinary(l_op, ref ll, ref lr), &ExprBinary(r_op, ref rl, ref rr)) => {
                l_op.node == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    || swap_binop(l_op.node, ll, lr).map_or(false, |(l_op, ll, lr)| {
                        l_op == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    })
            },
            (&ExprBreak(li, ref le), &ExprBreak(ri, ref re)) => {
                both(&li.label, &ri.label, |l, r| l.name.as_str() == r.name.as_str())
                    && both(le, re, |l, r| self.eq_expr(l, r))
            },
            (&ExprBox(ref l), &ExprBox(ref r)) => self.eq_expr(l, r),
            (&ExprCall(ref l_fun, ref l_args), &ExprCall(ref r_fun, ref r_args)) => {
                !self.ignore_fn && self.eq_expr(l_fun, r_fun) && self.eq_exprs(l_args, r_args)
            },
            (&ExprCast(ref lx, ref lt), &ExprCast(ref rx, ref rt)) |
            (&ExprType(ref lx, ref lt), &ExprType(ref rx, ref rt)) => self.eq_expr(lx, rx) && self.eq_ty(lt, rt),
            (&ExprField(ref l_f_exp, ref l_f_ident), &ExprField(ref r_f_exp, ref r_f_ident)) => {
                l_f_ident.node == r_f_ident.node && self.eq_expr(l_f_exp, r_f_exp)
            },
            (&ExprIndex(ref la, ref li), &ExprIndex(ref ra, ref ri)) => self.eq_expr(la, ra) && self.eq_expr(li, ri),
            (&ExprIf(ref lc, ref lt, ref le), &ExprIf(ref rc, ref rt, ref re)) => {
                self.eq_expr(lc, rc) && self.eq_expr(&**lt, &**rt) && both(le, re, |l, r| self.eq_expr(l, r))
            },
            (&ExprLit(ref l), &ExprLit(ref r)) => l.node == r.node,
            (&ExprLoop(ref lb, ref ll, ref lls), &ExprLoop(ref rb, ref rl, ref rls)) => {
                lls == rls && self.eq_block(lb, rb) && both(ll, rl, |l, r| l.name.as_str() == r.name.as_str())
            },
            (&ExprMatch(ref le, ref la, ref ls), &ExprMatch(ref re, ref ra, ref rs)) => {
                ls == rs && self.eq_expr(le, re) && over(la, ra, |l, r| {
                    self.eq_expr(&l.body, &r.body) && both(&l.guard, &r.guard, |l, r| self.eq_expr(l, r))
                        && over(&l.pats, &r.pats, |l, r| self.eq_pat(l, r))
                })
            },
            (&ExprMethodCall(ref l_path, _, ref l_args), &ExprMethodCall(ref r_path, _, ref r_args)) => {
                !self.ignore_fn && l_path == r_path && self.eq_exprs(l_args, r_args)
            },
            (&ExprRepeat(ref le, ll_id), &ExprRepeat(ref re, rl_id)) => {
                let mut celcx = constant_context(self.cx, self.cx.tcx.body_tables(ll_id));
                let ll = celcx.expr(&self.cx.tcx.hir.body(ll_id).value);
                let mut celcx = constant_context(self.cx, self.cx.tcx.body_tables(rl_id));
                let rl = celcx.expr(&self.cx.tcx.hir.body(rl_id).value);

                self.eq_expr(le, re) && ll == rl
            },
            (&ExprRet(ref l), &ExprRet(ref r)) => both(l, r, |l, r| self.eq_expr(l, r)),
            (&ExprPath(ref l), &ExprPath(ref r)) => self.eq_qpath(l, r),
            (&ExprStruct(ref l_path, ref lf, ref lo), &ExprStruct(ref r_path, ref rf, ref ro)) => {
                self.eq_qpath(l_path, r_path) && both(lo, ro, |l, r| self.eq_expr(l, r))
                    && over(lf, rf, |l, r| self.eq_field(l, r))
            },
            (&ExprTup(ref l_tup), &ExprTup(ref r_tup)) => self.eq_exprs(l_tup, r_tup),
            (&ExprUnary(l_op, ref le), &ExprUnary(r_op, ref re)) => l_op == r_op && self.eq_expr(le, re),
            (&ExprArray(ref l), &ExprArray(ref r)) => self.eq_exprs(l, r),
            (&ExprWhile(ref lc, ref lb, ref ll), &ExprWhile(ref rc, ref rb, ref rl)) => {
                self.eq_expr(lc, rc) && self.eq_block(lb, rb) && both(ll, rl, |l, r| l.name.as_str() == r.name.as_str())
            },
            _ => false,
        }
    }

    fn eq_exprs(&self, left: &P<[Expr]>, right: &P<[Expr]>) -> bool {
        over(left, right, |l, r| self.eq_expr(l, r))
    }

    fn eq_field(&self, left: &Field, right: &Field) -> bool {
        left.name.node == right.name.node && self.eq_expr(&left.expr, &right.expr)
    }

    fn eq_lifetime(&self, left: &Lifetime, right: &Lifetime) -> bool {
        left.name == right.name
    }

    /// Check whether two patterns are the same.
    pub fn eq_pat(&self, left: &Pat, right: &Pat) -> bool {
        match (&left.node, &right.node) {
            (&PatKind::Box(ref l), &PatKind::Box(ref r)) => self.eq_pat(l, r),
            (&PatKind::TupleStruct(ref lp, ref la, ls), &PatKind::TupleStruct(ref rp, ref ra, rs)) => {
                self.eq_qpath(lp, rp) && over(la, ra, |l, r| self.eq_pat(l, r)) && ls == rs
            },
            (&PatKind::Binding(ref lb, _, ref li, ref lp), &PatKind::Binding(ref rb, _, ref ri, ref rp)) => {
                lb == rb && li.node.as_str() == ri.node.as_str() && both(lp, rp, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Path(ref l), &PatKind::Path(ref r)) => self.eq_qpath(l, r),
            (&PatKind::Lit(ref l), &PatKind::Lit(ref r)) => self.eq_expr(l, r),
            (&PatKind::Tuple(ref l, ls), &PatKind::Tuple(ref r, rs)) => {
                ls == rs && over(l, r, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Range(ref ls, ref le, ref li), &PatKind::Range(ref rs, ref re, ref ri)) => {
                self.eq_expr(ls, rs) && self.eq_expr(le, re) && (*li == *ri)
            },
            (&PatKind::Ref(ref le, ref lm), &PatKind::Ref(ref re, ref rm)) => lm == rm && self.eq_pat(le, re),
            (&PatKind::Slice(ref ls, ref li, ref le), &PatKind::Slice(ref rs, ref ri, ref re)) => {
                over(ls, rs, |l, r| self.eq_pat(l, r)) && over(le, re, |l, r| self.eq_pat(l, r))
                    && both(li, ri, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Wild, &PatKind::Wild) => true,
            _ => false,
        }
    }

    fn eq_qpath(&self, left: &QPath, right: &QPath) -> bool {
        match (left, right) {
            (&QPath::Resolved(ref lty, ref lpath), &QPath::Resolved(ref rty, ref rpath)) => {
                both(lty, rty, |l, r| self.eq_ty(l, r)) && self.eq_path(lpath, rpath)
            },
            (&QPath::TypeRelative(ref lty, ref lseg), &QPath::TypeRelative(ref rty, ref rseg)) => {
                self.eq_ty(lty, rty) && self.eq_path_segment(lseg, rseg)
            },
            _ => false,
        }
    }

    fn eq_path(&self, left: &Path, right: &Path) -> bool {
        left.is_global() == right.is_global()
            && over(&left.segments, &right.segments, |l, r| self.eq_path_segment(l, r))
    }

    fn eq_path_parameters(&self, left: &PathParameters, right: &PathParameters) -> bool {
        if !(left.parenthesized || right.parenthesized) {
            over(&left.lifetimes, &right.lifetimes, |l, r| self.eq_lifetime(l, r))
                && over(&left.types, &right.types, |l, r| self.eq_ty(l, r))
                && over(&left.bindings, &right.bindings, |l, r| self.eq_type_binding(l, r))
        } else if left.parenthesized && right.parenthesized {
            over(left.inputs(), right.inputs(), |l, r| self.eq_ty(l, r))
                && both(
                    &Some(&left.bindings[0].ty),
                    &Some(&right.bindings[0].ty),
                    |l, r| self.eq_ty(l, r),
                )
        } else {
            false
        }
    }

    fn eq_path_segment(&self, left: &PathSegment, right: &PathSegment) -> bool {
        // The == of idents doesn't work with different contexts,
        // we have to be explicit about hygiene
        if left.name.as_str() != right.name.as_str() {
            return false;
        }
        match (&left.parameters, &right.parameters) {
            (&None, &None) => true,
            (&Some(ref l), &Some(ref r)) => self.eq_path_parameters(l, r),
            _ => false,
        }
    }

    fn eq_ty(&self, left: &Ty, right: &Ty) -> bool {
        match (&left.node, &right.node) {
            (&TySlice(ref l_vec), &TySlice(ref r_vec)) => self.eq_ty(l_vec, r_vec),
            (&TyArray(ref lt, ll_id), &TyArray(ref rt, rl_id)) => {
                self.eq_ty(lt, rt)
                    && self.eq_expr(&self.cx.tcx.hir.body(ll_id).value, &self.cx.tcx.hir.body(rl_id).value)
            },
            (&TyPtr(ref l_mut), &TyPtr(ref r_mut)) => l_mut.mutbl == r_mut.mutbl && self.eq_ty(&*l_mut.ty, &*r_mut.ty),
            (&TyRptr(_, ref l_rmut), &TyRptr(_, ref r_rmut)) => {
                l_rmut.mutbl == r_rmut.mutbl && self.eq_ty(&*l_rmut.ty, &*r_rmut.ty)
            },
            (&TyPath(ref l), &TyPath(ref r)) => self.eq_qpath(l, r),
            (&TyTup(ref l), &TyTup(ref r)) => over(l, r, |l, r| self.eq_ty(l, r)),
            (&TyInfer, &TyInfer) => true,
            _ => false,
        }
    }

    fn eq_type_binding(&self, left: &TypeBinding, right: &TypeBinding) -> bool {
        left.name == right.name && self.eq_ty(&left.ty, &right.ty)
    }
}

fn swap_binop<'a>(binop: BinOp_, lhs: &'a Expr, rhs: &'a Expr) -> Option<(BinOp_, &'a Expr, &'a Expr)> {
    match binop {
        BiAdd | BiMul | BiBitXor | BiBitAnd | BiEq | BiNe | BiBitOr => Some((binop, rhs, lhs)),
        BiLt => Some((BiGt, rhs, lhs)),
        BiLe => Some((BiGe, rhs, lhs)),
        BiGe => Some((BiLe, rhs, lhs)),
        BiGt => Some((BiLt, rhs, lhs)),
        BiShl | BiShr | BiRem | BiSub | BiDiv | BiAnd | BiOr => None,
    }
}

/// Check if the two `Option`s are both `None` or some equal values as per
/// `eq_fn`.
fn both<X, F>(l: &Option<X>, r: &Option<X>, mut eq_fn: F) -> bool
where
    F: FnMut(&X, &X) -> bool,
{
    l.as_ref()
        .map_or_else(|| r.is_none(), |x| r.as_ref().map_or(false, |y| eq_fn(x, y)))
}

/// Check if two slices are equal as per `eq_fn`.
fn over<X, F>(left: &[X], right: &[X], mut eq_fn: F) -> bool
where
    F: FnMut(&X, &X) -> bool,
{
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| eq_fn(x, y))
}


/// Type used to hash an ast element. This is different from the `Hash` trait
/// on ast types as this
/// trait would consider IDs and spans.
///
/// All expressions kind are hashed, but some might have a weaker hash.
pub struct SpanlessHash<'a, 'tcx: 'a> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'a, 'tcx>,
    s: DefaultHasher,
}

impl<'a, 'tcx: 'a> SpanlessHash<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self {
            cx,
            s: DefaultHasher::new(),
        }
    }

    pub fn finish(&self) -> u64 {
        self.s.finish()
    }

    pub fn hash_block(&mut self, b: &Block) {
        for s in &b.stmts {
            self.hash_stmt(s);
        }

        if let Some(ref e) = b.expr {
            self.hash_expr(e);
        }

        b.rules.hash(&mut self.s);
    }

    #[allow(many_single_char_names)]
    pub fn hash_expr(&mut self, e: &Expr) {
        if let Some(e) = constant_simple(self.cx, e) {
            return e.hash(&mut self.s);
        }

        match e.node {
            ExprAddrOf(m, ref e) => {
                let c: fn(_, _) -> _ = ExprAddrOf;
                c.hash(&mut self.s);
                m.hash(&mut self.s);
                self.hash_expr(e);
            },
            ExprAgain(i) => {
                let c: fn(_) -> _ = ExprAgain;
                c.hash(&mut self.s);
                if let Some(i) = i.label {
                    self.hash_name(&i.name);
                }
            },
            ExprYield(ref e) => {
                let c: fn(_) -> _ = ExprYield;
                c.hash(&mut self.s);
                self.hash_expr(e);
            },
            ExprAssign(ref l, ref r) => {
                let c: fn(_, _) -> _ = ExprAssign;
                c.hash(&mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprAssignOp(ref o, ref l, ref r) => {
                let c: fn(_, _, _) -> _ = ExprAssignOp;
                c.hash(&mut self.s);
                o.hash(&mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprBlock(ref b) => {
                let c: fn(_) -> _ = ExprBlock;
                c.hash(&mut self.s);
                self.hash_block(b);
            },
            ExprBinary(op, ref l, ref r) => {
                let c: fn(_, _, _) -> _ = ExprBinary;
                c.hash(&mut self.s);
                op.node.hash(&mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprBreak(i, ref j) => {
                let c: fn(_, _) -> _ = ExprBreak;
                c.hash(&mut self.s);
                if let Some(i) = i.label {
                    self.hash_name(&i.name);
                }
                if let Some(ref j) = *j {
                    self.hash_expr(&*j);
                }
            },
            ExprBox(ref e) => {
                let c: fn(_) -> _ = ExprBox;
                c.hash(&mut self.s);
                self.hash_expr(e);
            },
            ExprCall(ref fun, ref args) => {
                let c: fn(_, _) -> _ = ExprCall;
                c.hash(&mut self.s);
                self.hash_expr(fun);
                self.hash_exprs(args);
            },
            ExprCast(ref e, ref _ty) => {
                let c: fn(_, _) -> _ = ExprCast;
                c.hash(&mut self.s);
                self.hash_expr(e);
                // TODO: _ty
            },
            ExprClosure(cap, _, eid, _, _) => {
                let c: fn(_, _, _, _, _) -> _ = ExprClosure;
                c.hash(&mut self.s);
                cap.hash(&mut self.s);
                self.hash_expr(&self.cx.tcx.hir.body(eid).value);
            },
            ExprField(ref e, ref f) => {
                let c: fn(_, _) -> _ = ExprField;
                c.hash(&mut self.s);
                self.hash_expr(e);
                self.hash_name(&f.node);
            },
            ExprIndex(ref a, ref i) => {
                let c: fn(_, _) -> _ = ExprIndex;
                c.hash(&mut self.s);
                self.hash_expr(a);
                self.hash_expr(i);
            },
            ExprInlineAsm(..) => {
                let c: fn(_, _, _) -> _ = ExprInlineAsm;
                c.hash(&mut self.s);
            },
            ExprIf(ref cond, ref t, ref e) => {
                let c: fn(_, _, _) -> _ = ExprIf;
                c.hash(&mut self.s);
                self.hash_expr(cond);
                self.hash_expr(&**t);
                if let Some(ref e) = *e {
                    self.hash_expr(e);
                }
            },
            ExprLit(ref l) => {
                let c: fn(_) -> _ = ExprLit;
                c.hash(&mut self.s);
                l.hash(&mut self.s);
            },
            ExprLoop(ref b, ref i, _) => {
                let c: fn(_, _, _) -> _ = ExprLoop;
                c.hash(&mut self.s);
                self.hash_block(b);
                if let Some(i) = *i {
                    self.hash_name(&i.name);
                }
            },
            ExprMatch(ref e, ref arms, ref s) => {
                let c: fn(_, _, _) -> _ = ExprMatch;
                c.hash(&mut self.s);
                self.hash_expr(e);

                for arm in arms {
                    // TODO: arm.pat?
                    if let Some(ref e) = arm.guard {
                        self.hash_expr(e);
                    }
                    self.hash_expr(&arm.body);
                }

                s.hash(&mut self.s);
            },
            ExprMethodCall(ref path, ref _tys, ref args) => {
                let c: fn(_, _, _) -> _ = ExprMethodCall;
                c.hash(&mut self.s);
                self.hash_name(&path.name);
                self.hash_exprs(args);
            },
            ExprRepeat(ref e, l_id) => {
                let c: fn(_, _) -> _ = ExprRepeat;
                c.hash(&mut self.s);
                self.hash_expr(e);
                self.hash_expr(&self.cx.tcx.hir.body(l_id).value);
            },
            ExprRet(ref e) => {
                let c: fn(_) -> _ = ExprRet;
                c.hash(&mut self.s);
                if let Some(ref e) = *e {
                    self.hash_expr(e);
                }
            },
            ExprPath(ref qpath) => {
                let c: fn(_) -> _ = ExprPath;
                c.hash(&mut self.s);
                self.hash_qpath(qpath);
            },
            ExprStruct(ref path, ref fields, ref expr) => {
                let c: fn(_, _, _) -> _ = ExprStruct;
                c.hash(&mut self.s);

                self.hash_qpath(path);

                for f in fields {
                    self.hash_name(&f.name.node);
                    self.hash_expr(&f.expr);
                }

                if let Some(ref e) = *expr {
                    self.hash_expr(e);
                }
            },
            ExprTup(ref tup) => {
                let c: fn(_) -> _ = ExprTup;
                c.hash(&mut self.s);
                self.hash_exprs(tup);
            },
            ExprType(ref e, ref _ty) => {
                let c: fn(_, _) -> _ = ExprType;
                c.hash(&mut self.s);
                self.hash_expr(e);
                // TODO: _ty
            },
            ExprUnary(lop, ref le) => {
                let c: fn(_, _) -> _ = ExprUnary;
                c.hash(&mut self.s);

                lop.hash(&mut self.s);
                self.hash_expr(le);
            },
            ExprArray(ref v) => {
                let c: fn(_) -> _ = ExprArray;
                c.hash(&mut self.s);

                self.hash_exprs(v);
            },
            ExprWhile(ref cond, ref b, l) => {
                let c: fn(_, _, _) -> _ = ExprWhile;
                c.hash(&mut self.s);

                self.hash_expr(cond);
                self.hash_block(b);
                if let Some(l) = l {
                    self.hash_name(&l.name);
                }
            },
        }
    }

    pub fn hash_exprs(&mut self, e: &P<[Expr]>) {
        for e in e {
            self.hash_expr(e);
        }
    }

    pub fn hash_name(&mut self, n: &Name) {
        n.as_str().hash(&mut self.s);
    }

    pub fn hash_qpath(&mut self, p: &QPath) {
        match *p {
            QPath::Resolved(_, ref path) => {
                self.hash_path(path);
            },
            QPath::TypeRelative(_, ref path) => {
                self.hash_name(&path.name);
            },
        }
        // self.cx.tables.qpath_def(p, id).hash(&mut self.s);
    }

    pub fn hash_path(&mut self, p: &Path) {
        p.is_global().hash(&mut self.s);
        for p in &p.segments {
            self.hash_name(&p.name);
        }
    }

    pub fn hash_stmt(&mut self, b: &Stmt) {
        match b.node {
            StmtDecl(ref decl, _) => {
                let c: fn(_, _) -> _ = StmtDecl;
                c.hash(&mut self.s);

                if let DeclLocal(ref local) = decl.node {
                    if let Some(ref init) = local.init {
                        self.hash_expr(init);
                    }
                }
            },
            StmtExpr(ref expr, _) => {
                let c: fn(_, _) -> _ = StmtExpr;
                c.hash(&mut self.s);
                self.hash_expr(expr);
            },
            StmtSemi(ref expr, _) => {
                let c: fn(_, _) -> _ = StmtSemi;
                c.hash(&mut self.s);
                self.hash_expr(expr);
            },
        }
    }
}
