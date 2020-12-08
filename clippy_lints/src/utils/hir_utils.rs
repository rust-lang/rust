use crate::consts::{constant_context, constant_simple};
use crate::utils::differing_macro_contexts;
use rustc_ast::ast::InlineAsmTemplatePiece;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::{
    BinOpKind, Block, BlockCheckMode, BodyId, BorrowKind, CaptureBy, Expr, ExprKind, Field, FieldPat, FnRetTy,
    GenericArg, GenericArgs, Guard, InlineAsmOperand, Lifetime, LifetimeName, ParamName, Pat, PatKind, Path,
    PathSegment, QPath, Stmt, StmtKind, Ty, TyKind, TypeBinding,
};
use rustc_lint::LateContext;
use rustc_middle::ich::StableHashingContextProvider;
use rustc_middle::ty::TypeckResults;
use rustc_span::Symbol;
use std::hash::Hash;

/// Type used to check whether two ast are the same. This is different from the
/// operator
/// `==` on ast types as this operator would compare true equality with ID and
/// span.
///
/// Note that some expressions kinds are not considered but could be added.
pub struct SpanlessEq<'a, 'tcx> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'tcx>,
    maybe_typeck_results: Option<&'tcx TypeckResults<'tcx>>,
    allow_side_effects: bool,
}

impl<'a, 'tcx> SpanlessEq<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results(),
            allow_side_effects: true,
        }
    }

    /// Consider expressions containing potential side effects as not equal.
    pub fn deny_side_effects(self) -> Self {
        Self {
            allow_side_effects: false,
            ..self
        }
    }

    /// Checks whether two statements are the same.
    pub fn eq_stmt(&mut self, left: &Stmt<'_>, right: &Stmt<'_>) -> bool {
        match (&left.kind, &right.kind) {
            (&StmtKind::Local(ref l), &StmtKind::Local(ref r)) => {
                self.eq_pat(&l.pat, &r.pat)
                    && both(&l.ty, &r.ty, |l, r| self.eq_ty(l, r))
                    && both(&l.init, &r.init, |l, r| self.eq_expr(l, r))
            },
            (&StmtKind::Expr(ref l), &StmtKind::Expr(ref r)) | (&StmtKind::Semi(ref l), &StmtKind::Semi(ref r)) => {
                self.eq_expr(l, r)
            },
            _ => false,
        }
    }

    /// Checks whether two blocks are the same.
    pub fn eq_block(&mut self, left: &Block<'_>, right: &Block<'_>) -> bool {
        over(&left.stmts, &right.stmts, |l, r| self.eq_stmt(l, r))
            && both(&left.expr, &right.expr, |l, r| self.eq_expr(l, r))
    }

    #[allow(clippy::similar_names)]
    pub fn eq_expr(&mut self, left: &Expr<'_>, right: &Expr<'_>) -> bool {
        if !self.allow_side_effects && differing_macro_contexts(left.span, right.span) {
            return false;
        }

        if let Some(typeck_results) = self.maybe_typeck_results {
            if let (Some(l), Some(r)) = (
                constant_simple(self.cx, typeck_results, left),
                constant_simple(self.cx, typeck_results, right),
            ) {
                if l == r {
                    return true;
                }
            }
        }

        match (&reduce_exprkind(&left.kind), &reduce_exprkind(&right.kind)) {
            (&ExprKind::AddrOf(lb, l_mut, ref le), &ExprKind::AddrOf(rb, r_mut, ref re)) => {
                lb == rb && l_mut == r_mut && self.eq_expr(le, re)
            },
            (&ExprKind::Continue(li), &ExprKind::Continue(ri)) => {
                both(&li.label, &ri.label, |l, r| l.ident.as_str() == r.ident.as_str())
            },
            (&ExprKind::Assign(ref ll, ref lr, _), &ExprKind::Assign(ref rl, ref rr, _)) => {
                self.allow_side_effects && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            },
            (&ExprKind::AssignOp(ref lo, ref ll, ref lr), &ExprKind::AssignOp(ref ro, ref rl, ref rr)) => {
                self.allow_side_effects && lo.node == ro.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            },
            (&ExprKind::Block(ref l, _), &ExprKind::Block(ref r, _)) => self.eq_block(l, r),
            (&ExprKind::Binary(l_op, ref ll, ref lr), &ExprKind::Binary(r_op, ref rl, ref rr)) => {
                l_op.node == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    || swap_binop(l_op.node, ll, lr).map_or(false, |(l_op, ll, lr)| {
                        l_op == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    })
            },
            (&ExprKind::Break(li, ref le), &ExprKind::Break(ri, ref re)) => {
                both(&li.label, &ri.label, |l, r| l.ident.as_str() == r.ident.as_str())
                    && both(le, re, |l, r| self.eq_expr(l, r))
            },
            (&ExprKind::Box(ref l), &ExprKind::Box(ref r)) => self.eq_expr(l, r),
            (&ExprKind::Call(l_fun, l_args), &ExprKind::Call(r_fun, r_args)) => {
                self.allow_side_effects && self.eq_expr(l_fun, r_fun) && self.eq_exprs(l_args, r_args)
            },
            (&ExprKind::Cast(ref lx, ref lt), &ExprKind::Cast(ref rx, ref rt))
            | (&ExprKind::Type(ref lx, ref lt), &ExprKind::Type(ref rx, ref rt)) => {
                self.eq_expr(lx, rx) && self.eq_ty(lt, rt)
            },
            (&ExprKind::Field(ref l_f_exp, ref l_f_ident), &ExprKind::Field(ref r_f_exp, ref r_f_ident)) => {
                l_f_ident.name == r_f_ident.name && self.eq_expr(l_f_exp, r_f_exp)
            },
            (&ExprKind::Index(ref la, ref li), &ExprKind::Index(ref ra, ref ri)) => {
                self.eq_expr(la, ra) && self.eq_expr(li, ri)
            },
            (&ExprKind::Lit(ref l), &ExprKind::Lit(ref r)) => l.node == r.node,
            (&ExprKind::Loop(ref lb, ref ll, ref lls), &ExprKind::Loop(ref rb, ref rl, ref rls)) => {
                lls == rls && self.eq_block(lb, rb) && both(ll, rl, |l, r| l.ident.as_str() == r.ident.as_str())
            },
            (&ExprKind::Match(ref le, ref la, ref ls), &ExprKind::Match(ref re, ref ra, ref rs)) => {
                ls == rs
                    && self.eq_expr(le, re)
                    && over(la, ra, |l, r| {
                        self.eq_expr(&l.body, &r.body)
                            && both(&l.guard, &r.guard, |l, r| self.eq_guard(l, r))
                            && self.eq_pat(&l.pat, &r.pat)
                    })
            },
            (&ExprKind::MethodCall(l_path, _, l_args, _), &ExprKind::MethodCall(r_path, _, r_args, _)) => {
                self.allow_side_effects && self.eq_path_segment(l_path, r_path) && self.eq_exprs(l_args, r_args)
            },
            (&ExprKind::Repeat(ref le, ref ll_id), &ExprKind::Repeat(ref re, ref rl_id)) => {
                let mut celcx = constant_context(self.cx, self.cx.tcx.typeck_body(ll_id.body));
                let ll = celcx.expr(&self.cx.tcx.hir().body(ll_id.body).value);
                let mut celcx = constant_context(self.cx, self.cx.tcx.typeck_body(rl_id.body));
                let rl = celcx.expr(&self.cx.tcx.hir().body(rl_id.body).value);

                self.eq_expr(le, re) && ll == rl
            },
            (&ExprKind::Ret(ref l), &ExprKind::Ret(ref r)) => both(l, r, |l, r| self.eq_expr(l, r)),
            (&ExprKind::Path(ref l), &ExprKind::Path(ref r)) => self.eq_qpath(l, r),
            (&ExprKind::Struct(ref l_path, ref lf, ref lo), &ExprKind::Struct(ref r_path, ref rf, ref ro)) => {
                self.eq_qpath(l_path, r_path)
                    && both(lo, ro, |l, r| self.eq_expr(l, r))
                    && over(lf, rf, |l, r| self.eq_field(l, r))
            },
            (&ExprKind::Tup(l_tup), &ExprKind::Tup(r_tup)) => self.eq_exprs(l_tup, r_tup),
            (&ExprKind::Unary(l_op, ref le), &ExprKind::Unary(r_op, ref re)) => l_op == r_op && self.eq_expr(le, re),
            (&ExprKind::Array(l), &ExprKind::Array(r)) => self.eq_exprs(l, r),
            (&ExprKind::DropTemps(ref le), &ExprKind::DropTemps(ref re)) => self.eq_expr(le, re),
            _ => false,
        }
    }

    fn eq_exprs(&mut self, left: &[Expr<'_>], right: &[Expr<'_>]) -> bool {
        over(left, right, |l, r| self.eq_expr(l, r))
    }

    fn eq_field(&mut self, left: &Field<'_>, right: &Field<'_>) -> bool {
        left.ident.name == right.ident.name && self.eq_expr(&left.expr, &right.expr)
    }

    fn eq_guard(&mut self, left: &Guard<'_>, right: &Guard<'_>) -> bool {
        match (left, right) {
            (Guard::If(l), Guard::If(r)) => self.eq_expr(l, r),
        }
    }

    fn eq_generic_arg(&mut self, left: &GenericArg<'_>, right: &GenericArg<'_>) -> bool {
        match (left, right) {
            (GenericArg::Lifetime(l_lt), GenericArg::Lifetime(r_lt)) => Self::eq_lifetime(l_lt, r_lt),
            (GenericArg::Type(l_ty), GenericArg::Type(r_ty)) => self.eq_ty(l_ty, r_ty),
            _ => false,
        }
    }

    fn eq_lifetime(left: &Lifetime, right: &Lifetime) -> bool {
        left.name == right.name
    }

    pub fn eq_fieldpat(&mut self, left: &FieldPat<'_>, right: &FieldPat<'_>) -> bool {
        let (FieldPat { ident: li, pat: lp, .. }, FieldPat { ident: ri, pat: rp, .. }) = (&left, &right);
        li.name.as_str() == ri.name.as_str() && self.eq_pat(lp, rp)
    }

    /// Checks whether two patterns are the same.
    pub fn eq_pat(&mut self, left: &Pat<'_>, right: &Pat<'_>) -> bool {
        match (&left.kind, &right.kind) {
            (&PatKind::Box(ref l), &PatKind::Box(ref r)) => self.eq_pat(l, r),
            (&PatKind::Struct(ref lp, ref la, ..), &PatKind::Struct(ref rp, ref ra, ..)) => {
                self.eq_qpath(lp, rp) && over(la, ra, |l, r| self.eq_fieldpat(l, r))
            },
            (&PatKind::TupleStruct(ref lp, ref la, ls), &PatKind::TupleStruct(ref rp, ref ra, rs)) => {
                self.eq_qpath(lp, rp) && over(la, ra, |l, r| self.eq_pat(l, r)) && ls == rs
            },
            (&PatKind::Binding(ref lb, .., ref li, ref lp), &PatKind::Binding(ref rb, .., ref ri, ref rp)) => {
                lb == rb && li.name.as_str() == ri.name.as_str() && both(lp, rp, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Path(ref l), &PatKind::Path(ref r)) => self.eq_qpath(l, r),
            (&PatKind::Lit(ref l), &PatKind::Lit(ref r)) => self.eq_expr(l, r),
            (&PatKind::Tuple(ref l, ls), &PatKind::Tuple(ref r, rs)) => {
                ls == rs && over(l, r, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Range(ref ls, ref le, li), &PatKind::Range(ref rs, ref re, ri)) => {
                both(ls, rs, |a, b| self.eq_expr(a, b)) && both(le, re, |a, b| self.eq_expr(a, b)) && (li == ri)
            },
            (&PatKind::Ref(ref le, ref lm), &PatKind::Ref(ref re, ref rm)) => lm == rm && self.eq_pat(le, re),
            (&PatKind::Slice(ref ls, ref li, ref le), &PatKind::Slice(ref rs, ref ri, ref re)) => {
                over(ls, rs, |l, r| self.eq_pat(l, r))
                    && over(le, re, |l, r| self.eq_pat(l, r))
                    && both(li, ri, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Wild, &PatKind::Wild) => true,
            _ => false,
        }
    }

    #[allow(clippy::similar_names)]
    fn eq_qpath(&mut self, left: &QPath<'_>, right: &QPath<'_>) -> bool {
        match (left, right) {
            (&QPath::Resolved(ref lty, ref lpath), &QPath::Resolved(ref rty, ref rpath)) => {
                both(lty, rty, |l, r| self.eq_ty(l, r)) && self.eq_path(lpath, rpath)
            },
            (&QPath::TypeRelative(ref lty, ref lseg), &QPath::TypeRelative(ref rty, ref rseg)) => {
                self.eq_ty(lty, rty) && self.eq_path_segment(lseg, rseg)
            },
            (&QPath::LangItem(llang_item, _), &QPath::LangItem(rlang_item, _)) => llang_item == rlang_item,
            _ => false,
        }
    }

    fn eq_path(&mut self, left: &Path<'_>, right: &Path<'_>) -> bool {
        left.is_global() == right.is_global()
            && over(&left.segments, &right.segments, |l, r| self.eq_path_segment(l, r))
    }

    fn eq_path_parameters(&mut self, left: &GenericArgs<'_>, right: &GenericArgs<'_>) -> bool {
        if !(left.parenthesized || right.parenthesized) {
            over(&left.args, &right.args, |l, r| self.eq_generic_arg(l, r)) // FIXME(flip1995): may not work
                && over(&left.bindings, &right.bindings, |l, r| self.eq_type_binding(l, r))
        } else if left.parenthesized && right.parenthesized {
            over(left.inputs(), right.inputs(), |l, r| self.eq_ty(l, r))
                && both(&Some(&left.bindings[0].ty()), &Some(&right.bindings[0].ty()), |l, r| {
                    self.eq_ty(l, r)
                })
        } else {
            false
        }
    }

    pub fn eq_path_segments(&mut self, left: &[PathSegment<'_>], right: &[PathSegment<'_>]) -> bool {
        left.len() == right.len() && left.iter().zip(right).all(|(l, r)| self.eq_path_segment(l, r))
    }

    pub fn eq_path_segment(&mut self, left: &PathSegment<'_>, right: &PathSegment<'_>) -> bool {
        // The == of idents doesn't work with different contexts,
        // we have to be explicit about hygiene
        left.ident.as_str() == right.ident.as_str()
            && both(&left.args, &right.args, |l, r| self.eq_path_parameters(l, r))
    }

    pub fn eq_ty(&mut self, left: &Ty<'_>, right: &Ty<'_>) -> bool {
        self.eq_ty_kind(&left.kind, &right.kind)
    }

    #[allow(clippy::similar_names)]
    pub fn eq_ty_kind(&mut self, left: &TyKind<'_>, right: &TyKind<'_>) -> bool {
        match (left, right) {
            (&TyKind::Slice(ref l_vec), &TyKind::Slice(ref r_vec)) => self.eq_ty(l_vec, r_vec),
            (&TyKind::Array(ref lt, ref ll_id), &TyKind::Array(ref rt, ref rl_id)) => {
                let old_maybe_typeck_results = self.maybe_typeck_results;

                let mut celcx = constant_context(self.cx, self.cx.tcx.typeck_body(ll_id.body));
                self.maybe_typeck_results = Some(self.cx.tcx.typeck_body(ll_id.body));
                let ll = celcx.expr(&self.cx.tcx.hir().body(ll_id.body).value);

                let mut celcx = constant_context(self.cx, self.cx.tcx.typeck_body(rl_id.body));
                self.maybe_typeck_results = Some(self.cx.tcx.typeck_body(rl_id.body));
                let rl = celcx.expr(&self.cx.tcx.hir().body(rl_id.body).value);

                let eq_ty = self.eq_ty(lt, rt);
                self.maybe_typeck_results = old_maybe_typeck_results;
                eq_ty && ll == rl
            },
            (&TyKind::Ptr(ref l_mut), &TyKind::Ptr(ref r_mut)) => {
                l_mut.mutbl == r_mut.mutbl && self.eq_ty(&*l_mut.ty, &*r_mut.ty)
            },
            (&TyKind::Rptr(_, ref l_rmut), &TyKind::Rptr(_, ref r_rmut)) => {
                l_rmut.mutbl == r_rmut.mutbl && self.eq_ty(&*l_rmut.ty, &*r_rmut.ty)
            },
            (&TyKind::Path(ref l), &TyKind::Path(ref r)) => self.eq_qpath(l, r),
            (&TyKind::Tup(ref l), &TyKind::Tup(ref r)) => over(l, r, |l, r| self.eq_ty(l, r)),
            (&TyKind::Infer, &TyKind::Infer) => true,
            _ => false,
        }
    }

    fn eq_type_binding(&mut self, left: &TypeBinding<'_>, right: &TypeBinding<'_>) -> bool {
        left.ident.name == right.ident.name && self.eq_ty(&left.ty(), &right.ty())
    }
}

/// Some simple reductions like `{ return }` => `return`
fn reduce_exprkind<'hir>(kind: &'hir ExprKind<'hir>) -> &ExprKind<'hir> {
    if let ExprKind::Block(block, _) = kind {
        match (block.stmts, block.expr) {
            // `{}` => `()`
            ([], None) => &ExprKind::Tup(&[]),
            ([], Some(expr)) => match expr.kind {
                // `{ return .. }` => `return ..`
                ExprKind::Ret(..) => &expr.kind,
                _ => kind,
            },
            ([stmt], None) => match stmt.kind {
                StmtKind::Expr(expr) | StmtKind::Semi(expr) => match expr.kind {
                    // `{ return ..; }` => `return ..`
                    ExprKind::Ret(..) => &expr.kind,
                    _ => kind,
                },
                _ => kind,
            },
            _ => kind,
        }
    } else {
        kind
    }
}

fn swap_binop<'a>(
    binop: BinOpKind,
    lhs: &'a Expr<'a>,
    rhs: &'a Expr<'a>,
) -> Option<(BinOpKind, &'a Expr<'a>, &'a Expr<'a>)> {
    match binop {
        BinOpKind::Add | BinOpKind::Eq | BinOpKind::Ne | BinOpKind::BitAnd | BinOpKind::BitXor | BinOpKind::BitOr => {
            Some((binop, rhs, lhs))
        },
        BinOpKind::Lt => Some((BinOpKind::Gt, rhs, lhs)),
        BinOpKind::Le => Some((BinOpKind::Ge, rhs, lhs)),
        BinOpKind::Ge => Some((BinOpKind::Le, rhs, lhs)),
        BinOpKind::Gt => Some((BinOpKind::Lt, rhs, lhs)),
        BinOpKind::Mul // Not always commutative, e.g. with matrices. See issue #5698
        | BinOpKind::Shl
        | BinOpKind::Shr
        | BinOpKind::Rem
        | BinOpKind::Sub
        | BinOpKind::Div
        | BinOpKind::And
        | BinOpKind::Or => None,
    }
}

/// Checks if the two `Option`s are both `None` or some equal values as per
/// `eq_fn`.
pub fn both<X>(l: &Option<X>, r: &Option<X>, mut eq_fn: impl FnMut(&X, &X) -> bool) -> bool {
    l.as_ref()
        .map_or_else(|| r.is_none(), |x| r.as_ref().map_or(false, |y| eq_fn(x, y)))
}

/// Checks if two slices are equal as per `eq_fn`.
pub fn over<X>(left: &[X], right: &[X], mut eq_fn: impl FnMut(&X, &X) -> bool) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| eq_fn(x, y))
}

/// Checks if two expressions evaluate to the same value, and don't contain any side effects.
pub fn eq_expr_value(cx: &LateContext<'_>, left: &Expr<'_>, right: &Expr<'_>) -> bool {
    SpanlessEq::new(cx).deny_side_effects().eq_expr(left, right)
}

/// Type used to hash an ast element. This is different from the `Hash` trait
/// on ast types as this
/// trait would consider IDs and spans.
///
/// All expressions kind are hashed, but some might have a weaker hash.
pub struct SpanlessHash<'a, 'tcx> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'tcx>,
    maybe_typeck_results: Option<&'tcx TypeckResults<'tcx>>,
    s: StableHasher,
}

impl<'a, 'tcx> SpanlessHash<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results(),
            s: StableHasher::new(),
        }
    }

    pub fn finish(self) -> u64 {
        self.s.finish()
    }

    pub fn hash_block(&mut self, b: &Block<'_>) {
        for s in b.stmts {
            self.hash_stmt(s);
        }

        if let Some(ref e) = b.expr {
            self.hash_expr(e);
        }

        match b.rules {
            BlockCheckMode::DefaultBlock => 0,
            BlockCheckMode::UnsafeBlock(_) => 1,
            BlockCheckMode::PushUnsafeBlock(_) => 2,
            BlockCheckMode::PopUnsafeBlock(_) => 3,
        }
        .hash(&mut self.s);
    }

    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    pub fn hash_expr(&mut self, e: &Expr<'_>) {
        let simple_const = self
            .maybe_typeck_results
            .and_then(|typeck_results| constant_simple(self.cx, typeck_results, e));

        // const hashing may result in the same hash as some unrelated node, so add a sort of
        // discriminant depending on which path we're choosing next
        simple_const.is_some().hash(&mut self.s);

        if let Some(e) = simple_const {
            return e.hash(&mut self.s);
        }

        std::mem::discriminant(&e.kind).hash(&mut self.s);

        match e.kind {
            ExprKind::AddrOf(kind, m, ref e) => {
                match kind {
                    BorrowKind::Ref => 0,
                    BorrowKind::Raw => 1,
                }
                .hash(&mut self.s);
                m.hash(&mut self.s);
                self.hash_expr(e);
            },
            ExprKind::Continue(i) => {
                if let Some(i) = i.label {
                    self.hash_name(i.ident.name);
                }
            },
            ExprKind::Assign(ref l, ref r, _) => {
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::AssignOp(ref o, ref l, ref r) => {
                o.node
                    .hash_stable(&mut self.cx.tcx.get_stable_hashing_context(), &mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::Block(ref b, _) => {
                self.hash_block(b);
            },
            ExprKind::Binary(op, ref l, ref r) => {
                op.node
                    .hash_stable(&mut self.cx.tcx.get_stable_hashing_context(), &mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::Break(i, ref j) => {
                if let Some(i) = i.label {
                    self.hash_name(i.ident.name);
                }
                if let Some(ref j) = *j {
                    self.hash_expr(&*j);
                }
            },
            ExprKind::Box(ref e) | ExprKind::DropTemps(ref e) | ExprKind::Yield(ref e, _) => {
                self.hash_expr(e);
            },
            ExprKind::Call(ref fun, args) => {
                self.hash_expr(fun);
                self.hash_exprs(args);
            },
            ExprKind::Cast(ref e, ref ty) | ExprKind::Type(ref e, ref ty) => {
                self.hash_expr(e);
                self.hash_ty(ty);
            },
            ExprKind::Closure(cap, _, eid, _, _) => {
                match cap {
                    CaptureBy::Value => 0,
                    CaptureBy::Ref => 1,
                }
                .hash(&mut self.s);
                // closures inherit TypeckResults
                self.hash_expr(&self.cx.tcx.hir().body(eid).value);
            },
            ExprKind::Field(ref e, ref f) => {
                self.hash_expr(e);
                self.hash_name(f.name);
            },
            ExprKind::Index(ref a, ref i) => {
                self.hash_expr(a);
                self.hash_expr(i);
            },
            ExprKind::InlineAsm(ref asm) => {
                for piece in asm.template {
                    match piece {
                        InlineAsmTemplatePiece::String(s) => s.hash(&mut self.s),
                        InlineAsmTemplatePiece::Placeholder {
                            operand_idx,
                            modifier,
                            span: _,
                        } => {
                            operand_idx.hash(&mut self.s);
                            modifier.hash(&mut self.s);
                        },
                    }
                }
                asm.options.hash(&mut self.s);
                for (op, _op_sp) in asm.operands {
                    match op {
                        InlineAsmOperand::In { reg, expr } => {
                            reg.hash(&mut self.s);
                            self.hash_expr(expr);
                        },
                        InlineAsmOperand::Out { reg, late, expr } => {
                            reg.hash(&mut self.s);
                            late.hash(&mut self.s);
                            if let Some(expr) = expr {
                                self.hash_expr(expr);
                            }
                        },
                        InlineAsmOperand::InOut { reg, late, expr } => {
                            reg.hash(&mut self.s);
                            late.hash(&mut self.s);
                            self.hash_expr(expr);
                        },
                        InlineAsmOperand::SplitInOut {
                            reg,
                            late,
                            in_expr,
                            out_expr,
                        } => {
                            reg.hash(&mut self.s);
                            late.hash(&mut self.s);
                            self.hash_expr(in_expr);
                            if let Some(out_expr) = out_expr {
                                self.hash_expr(out_expr);
                            }
                        },
                        InlineAsmOperand::Const { expr } | InlineAsmOperand::Sym { expr } => self.hash_expr(expr),
                    }
                }
            },
            ExprKind::LlvmInlineAsm(..) | ExprKind::Err => {},
            ExprKind::Lit(ref l) => {
                l.node.hash(&mut self.s);
            },
            ExprKind::Loop(ref b, ref i, _) => {
                self.hash_block(b);
                if let Some(i) = *i {
                    self.hash_name(i.ident.name);
                }
            },
            ExprKind::Match(ref e, arms, ref s) => {
                self.hash_expr(e);

                for arm in arms {
                    // TODO: arm.pat?
                    if let Some(ref e) = arm.guard {
                        self.hash_guard(e);
                    }
                    self.hash_expr(&arm.body);
                }

                s.hash(&mut self.s);
            },
            ExprKind::MethodCall(ref path, ref _tys, args, ref _fn_span) => {
                self.hash_name(path.ident.name);
                self.hash_exprs(args);
            },
            ExprKind::ConstBlock(ref l_id) => {
                self.hash_body(l_id.body);
            },
            ExprKind::Repeat(ref e, ref l_id) => {
                self.hash_expr(e);
                self.hash_body(l_id.body);
            },
            ExprKind::Ret(ref e) => {
                if let Some(ref e) = *e {
                    self.hash_expr(e);
                }
            },
            ExprKind::Path(ref qpath) => {
                self.hash_qpath(qpath);
            },
            ExprKind::Struct(ref path, fields, ref expr) => {
                self.hash_qpath(path);

                for f in fields {
                    self.hash_name(f.ident.name);
                    self.hash_expr(&f.expr);
                }

                if let Some(ref e) = *expr {
                    self.hash_expr(e);
                }
            },
            ExprKind::Tup(tup) => {
                self.hash_exprs(tup);
            },
            ExprKind::Array(v) => {
                self.hash_exprs(v);
            },
            ExprKind::Unary(lop, ref le) => {
                lop.hash_stable(&mut self.cx.tcx.get_stable_hashing_context(), &mut self.s);
                self.hash_expr(le);
            },
        }
    }

    pub fn hash_exprs(&mut self, e: &[Expr<'_>]) {
        for e in e {
            self.hash_expr(e);
        }
    }

    pub fn hash_name(&mut self, n: Symbol) {
        n.as_str().hash(&mut self.s);
    }

    pub fn hash_qpath(&mut self, p: &QPath<'_>) {
        match *p {
            QPath::Resolved(_, ref path) => {
                self.hash_path(path);
            },
            QPath::TypeRelative(_, ref path) => {
                self.hash_name(path.ident.name);
            },
            QPath::LangItem(lang_item, ..) => {
                lang_item.hash_stable(&mut self.cx.tcx.get_stable_hashing_context(), &mut self.s);
            },
        }
        // self.maybe_typeck_results.unwrap().qpath_res(p, id).hash(&mut self.s);
    }

    pub fn hash_path(&mut self, p: &Path<'_>) {
        p.is_global().hash(&mut self.s);
        for p in p.segments {
            self.hash_name(p.ident.name);
        }
    }

    pub fn hash_stmt(&mut self, b: &Stmt<'_>) {
        std::mem::discriminant(&b.kind).hash(&mut self.s);

        match &b.kind {
            StmtKind::Local(local) => {
                if let Some(ref init) = local.init {
                    self.hash_expr(init);
                }
            },
            StmtKind::Item(..) => {},
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.hash_expr(expr);
            },
        }
    }

    pub fn hash_guard(&mut self, g: &Guard<'_>) {
        match g {
            Guard::If(ref expr) => {
                self.hash_expr(expr);
            },
        }
    }

    pub fn hash_lifetime(&mut self, lifetime: &Lifetime) {
        std::mem::discriminant(&lifetime.name).hash(&mut self.s);
        if let LifetimeName::Param(ref name) = lifetime.name {
            std::mem::discriminant(name).hash(&mut self.s);
            match name {
                ParamName::Plain(ref ident) => {
                    ident.name.hash(&mut self.s);
                },
                ParamName::Fresh(ref size) => {
                    size.hash(&mut self.s);
                },
                ParamName::Error => {},
            }
        }
    }

    pub fn hash_ty(&mut self, ty: &Ty<'_>) {
        self.hash_tykind(&ty.kind);
    }

    pub fn hash_tykind(&mut self, ty: &TyKind<'_>) {
        std::mem::discriminant(ty).hash(&mut self.s);
        match ty {
            TyKind::Slice(ty) => {
                self.hash_ty(ty);
            },
            TyKind::Array(ty, anon_const) => {
                self.hash_ty(ty);
                self.hash_body(anon_const.body);
            },
            TyKind::Ptr(mut_ty) => {
                self.hash_ty(&mut_ty.ty);
                mut_ty.mutbl.hash(&mut self.s);
            },
            TyKind::Rptr(lifetime, mut_ty) => {
                self.hash_lifetime(lifetime);
                self.hash_ty(&mut_ty.ty);
                mut_ty.mutbl.hash(&mut self.s);
            },
            TyKind::BareFn(bfn) => {
                bfn.unsafety.hash(&mut self.s);
                bfn.abi.hash(&mut self.s);
                for arg in bfn.decl.inputs {
                    self.hash_ty(&arg);
                }
                match bfn.decl.output {
                    FnRetTy::DefaultReturn(_) => {
                        ().hash(&mut self.s);
                    },
                    FnRetTy::Return(ref ty) => {
                        self.hash_ty(ty);
                    },
                }
                bfn.decl.c_variadic.hash(&mut self.s);
            },
            TyKind::Tup(ty_list) => {
                for ty in *ty_list {
                    self.hash_ty(ty);
                }
            },
            TyKind::Path(qpath) => match qpath {
                QPath::Resolved(ref maybe_ty, ref path) => {
                    if let Some(ref ty) = maybe_ty {
                        self.hash_ty(ty);
                    }
                    for segment in path.segments {
                        segment.ident.name.hash(&mut self.s);
                        self.hash_generic_args(segment.generic_args().args);
                    }
                },
                QPath::TypeRelative(ref ty, ref segment) => {
                    self.hash_ty(ty);
                    segment.ident.name.hash(&mut self.s);
                },
                QPath::LangItem(lang_item, ..) => {
                    lang_item.hash(&mut self.s);
                },
            },
            TyKind::OpaqueDef(_, arg_list) => {
                self.hash_generic_args(arg_list);
            },
            TyKind::TraitObject(_, lifetime) => {
                self.hash_lifetime(lifetime);
            },
            TyKind::Typeof(anon_const) => {
                self.hash_body(anon_const.body);
            },
            TyKind::Err | TyKind::Infer | TyKind::Never => {},
        }
    }

    pub fn hash_body(&mut self, body_id: BodyId) {
        // swap out TypeckResults when hashing a body
        let old_maybe_typeck_results = self.maybe_typeck_results.replace(self.cx.tcx.typeck_body(body_id));
        self.hash_expr(&self.cx.tcx.hir().body(body_id).value);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn hash_generic_args(&mut self, arg_list: &[GenericArg<'_>]) {
        for arg in arg_list {
            match arg {
                GenericArg::Lifetime(ref l) => self.hash_lifetime(l),
                GenericArg::Type(ref ty) => self.hash_ty(&ty),
                GenericArg::Const(ref ca) => self.hash_body(ca.value.body),
            }
        }
    }
}
