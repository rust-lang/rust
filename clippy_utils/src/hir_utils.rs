use crate::consts::constant_simple;
use crate::macros::macro_backtrace;
use crate::source::snippet_opt;
use rustc_ast::ast::InlineAsmTemplatePiece;
use rustc_data_structures::fx::FxHasher;
use rustc_hir::def::Res;
use rustc_hir::HirIdMap;
use rustc_hir::{
    ArrayLen, BinOpKind, BindingAnnotation, Block, BodyId, Closure, Expr, ExprField, ExprKind, FnRetTy, GenericArg,
    GenericArgs, Guard, HirId, InlineAsmOperand, Let, Lifetime, LifetimeName, ParamName, Pat, PatField, PatKind, Path,
    PathSegment, QPath, Stmt, StmtKind, Ty, TyKind, TypeBinding,
};
use rustc_lexer::{tokenize, TokenKind};
use rustc_lint::LateContext;
use rustc_middle::ty::TypeckResults;
use rustc_span::{sym, Symbol};
use std::hash::{Hash, Hasher};

/// Callback that is called when two expressions are not equal in the sense of `SpanlessEq`, but
/// other conditions would make them equal.
type SpanlessEqCallback<'a> = dyn FnMut(&Expr<'_>, &Expr<'_>) -> bool + 'a;

/// Type used to check whether two ast are the same. This is different from the
/// operator `==` on ast types as this operator would compare true equality with
/// ID and span.
///
/// Note that some expressions kinds are not considered but could be added.
pub struct SpanlessEq<'a, 'tcx> {
    /// Context used to evaluate constant expressions.
    cx: &'a LateContext<'tcx>,
    maybe_typeck_results: Option<(&'tcx TypeckResults<'tcx>, &'tcx TypeckResults<'tcx>)>,
    allow_side_effects: bool,
    expr_fallback: Option<Box<SpanlessEqCallback<'a>>>,
}

impl<'a, 'tcx> SpanlessEq<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results().map(|x| (x, x)),
            allow_side_effects: true,
            expr_fallback: None,
        }
    }

    /// Consider expressions containing potential side effects as not equal.
    #[must_use]
    pub fn deny_side_effects(self) -> Self {
        Self {
            allow_side_effects: false,
            ..self
        }
    }

    #[must_use]
    pub fn expr_fallback(self, expr_fallback: impl FnMut(&Expr<'_>, &Expr<'_>) -> bool + 'a) -> Self {
        Self {
            expr_fallback: Some(Box::new(expr_fallback)),
            ..self
        }
    }

    /// Use this method to wrap comparisons that may involve inter-expression context.
    /// See `self.locals`.
    pub fn inter_expr(&mut self) -> HirEqInterExpr<'_, 'a, 'tcx> {
        HirEqInterExpr {
            inner: self,
            locals: HirIdMap::default(),
        }
    }

    pub fn eq_block(&mut self, left: &Block<'_>, right: &Block<'_>) -> bool {
        self.inter_expr().eq_block(left, right)
    }

    pub fn eq_expr(&mut self, left: &Expr<'_>, right: &Expr<'_>) -> bool {
        self.inter_expr().eq_expr(left, right)
    }

    pub fn eq_path(&mut self, left: &Path<'_>, right: &Path<'_>) -> bool {
        self.inter_expr().eq_path(left, right)
    }

    pub fn eq_path_segment(&mut self, left: &PathSegment<'_>, right: &PathSegment<'_>) -> bool {
        self.inter_expr().eq_path_segment(left, right)
    }

    pub fn eq_path_segments(&mut self, left: &[PathSegment<'_>], right: &[PathSegment<'_>]) -> bool {
        self.inter_expr().eq_path_segments(left, right)
    }
}

pub struct HirEqInterExpr<'a, 'b, 'tcx> {
    inner: &'a mut SpanlessEq<'b, 'tcx>,

    // When binding are declared, the binding ID in the left expression is mapped to the one on the
    // right. For example, when comparing `{ let x = 1; x + 2 }` and `{ let y = 1; y + 2 }`,
    // these blocks are considered equal since `x` is mapped to `y`.
    pub locals: HirIdMap<HirId>,
}

impl HirEqInterExpr<'_, '_, '_> {
    pub fn eq_stmt(&mut self, left: &Stmt<'_>, right: &Stmt<'_>) -> bool {
        match (&left.kind, &right.kind) {
            (&StmtKind::Local(l), &StmtKind::Local(r)) => {
                // This additional check ensures that the type of the locals are equivalent even if the init
                // expression or type have some inferred parts.
                if let Some((typeck_lhs, typeck_rhs)) = self.inner.maybe_typeck_results {
                    let l_ty = typeck_lhs.pat_ty(l.pat);
                    let r_ty = typeck_rhs.pat_ty(r.pat);
                    if l_ty != r_ty {
                        return false;
                    }
                }

                // eq_pat adds the HirIds to the locals map. We therefor call it last to make sure that
                // these only get added if the init and type is equal.
                both(&l.init, &r.init, |l, r| self.eq_expr(l, r))
                    && both(&l.ty, &r.ty, |l, r| self.eq_ty(l, r))
                    && both(&l.els, &r.els, |l, r| self.eq_block(l, r))
                    && self.eq_pat(l.pat, r.pat)
            },
            (&StmtKind::Expr(l), &StmtKind::Expr(r)) | (&StmtKind::Semi(l), &StmtKind::Semi(r)) => self.eq_expr(l, r),
            _ => false,
        }
    }

    /// Checks whether two blocks are the same.
    fn eq_block(&mut self, left: &Block<'_>, right: &Block<'_>) -> bool {
        match (left.stmts, left.expr, right.stmts, right.expr) {
            ([], None, [], None) => {
                // For empty blocks, check to see if the tokens are equal. This will catch the case where a macro
                // expanded to nothing, or the cfg attribute was used.
                let (left, right) = match (
                    snippet_opt(self.inner.cx, left.span),
                    snippet_opt(self.inner.cx, right.span),
                ) {
                    (Some(left), Some(right)) => (left, right),
                    _ => return true,
                };
                let mut left_pos = 0;
                let left = tokenize(&left)
                    .map(|t| {
                        let end = left_pos + t.len as usize;
                        let s = &left[left_pos..end];
                        left_pos = end;
                        (t, s)
                    })
                    .filter(|(t, _)| {
                        !matches!(
                            t.kind,
                            TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace
                        )
                    })
                    .map(|(_, s)| s);
                let mut right_pos = 0;
                let right = tokenize(&right)
                    .map(|t| {
                        let end = right_pos + t.len as usize;
                        let s = &right[right_pos..end];
                        right_pos = end;
                        (t, s)
                    })
                    .filter(|(t, _)| {
                        !matches!(
                            t.kind,
                            TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace
                        )
                    })
                    .map(|(_, s)| s);
                left.eq(right)
            },
            _ => {
                over(left.stmts, right.stmts, |l, r| self.eq_stmt(l, r))
                    && both(&left.expr, &right.expr, |l, r| self.eq_expr(l, r))
            },
        }
    }

    fn should_ignore(&mut self, expr: &Expr<'_>) -> bool {
        macro_backtrace(expr.span).last().map_or(false, |macro_call| {
            matches!(
                &self.inner.cx.tcx.get_diagnostic_name(macro_call.def_id),
                Some(sym::todo_macro | sym::unimplemented_macro)
            )
        })
    }

    pub fn eq_array_length(&mut self, left: ArrayLen, right: ArrayLen) -> bool {
        match (left, right) {
            (ArrayLen::Infer(..), ArrayLen::Infer(..)) => true,
            (ArrayLen::Body(l_ct), ArrayLen::Body(r_ct)) => self.eq_body(l_ct.body, r_ct.body),
            (_, _) => false,
        }
    }

    pub fn eq_body(&mut self, left: BodyId, right: BodyId) -> bool {
        // swap out TypeckResults when hashing a body
        let old_maybe_typeck_results = self.inner.maybe_typeck_results.replace((
            self.inner.cx.tcx.typeck_body(left),
            self.inner.cx.tcx.typeck_body(right),
        ));
        let res = self.eq_expr(
            &self.inner.cx.tcx.hir().body(left).value,
            &self.inner.cx.tcx.hir().body(right).value,
        );
        self.inner.maybe_typeck_results = old_maybe_typeck_results;
        res
    }

    #[expect(clippy::similar_names)]
    pub fn eq_expr(&mut self, left: &Expr<'_>, right: &Expr<'_>) -> bool {
        if !self.inner.allow_side_effects && left.span.ctxt() != right.span.ctxt() {
            return false;
        }

        if let Some((typeck_lhs, typeck_rhs)) = self.inner.maybe_typeck_results {
            if let (Some(l), Some(r)) = (
                constant_simple(self.inner.cx, typeck_lhs, left),
                constant_simple(self.inner.cx, typeck_rhs, right),
            ) {
                if l == r {
                    return true;
                }
            }
        }

        let is_eq = match (
            reduce_exprkind(self.inner.cx, &left.kind),
            reduce_exprkind(self.inner.cx, &right.kind),
        ) {
            (&ExprKind::AddrOf(lb, l_mut, le), &ExprKind::AddrOf(rb, r_mut, re)) => {
                lb == rb && l_mut == r_mut && self.eq_expr(le, re)
            },
            (&ExprKind::Continue(li), &ExprKind::Continue(ri)) => {
                both(&li.label, &ri.label, |l, r| l.ident.name == r.ident.name)
            },
            (&ExprKind::Assign(ll, lr, _), &ExprKind::Assign(rl, rr, _)) => {
                self.inner.allow_side_effects && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            },
            (&ExprKind::AssignOp(ref lo, ll, lr), &ExprKind::AssignOp(ref ro, rl, rr)) => {
                self.inner.allow_side_effects && lo.node == ro.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
            },
            (&ExprKind::Block(l, _), &ExprKind::Block(r, _)) => self.eq_block(l, r),
            (&ExprKind::Binary(l_op, ll, lr), &ExprKind::Binary(r_op, rl, rr)) => {
                l_op.node == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    || swap_binop(l_op.node, ll, lr).map_or(false, |(l_op, ll, lr)| {
                        l_op == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr)
                    })
            },
            (&ExprKind::Break(li, ref le), &ExprKind::Break(ri, ref re)) => {
                both(&li.label, &ri.label, |l, r| l.ident.name == r.ident.name)
                    && both(le, re, |l, r| self.eq_expr(l, r))
            },
            (&ExprKind::Box(l), &ExprKind::Box(r)) => self.eq_expr(l, r),
            (&ExprKind::Call(l_fun, l_args), &ExprKind::Call(r_fun, r_args)) => {
                self.inner.allow_side_effects && self.eq_expr(l_fun, r_fun) && self.eq_exprs(l_args, r_args)
            },
            (&ExprKind::Cast(lx, lt), &ExprKind::Cast(rx, rt)) | (&ExprKind::Type(lx, lt), &ExprKind::Type(rx, rt)) => {
                self.eq_expr(lx, rx) && self.eq_ty(lt, rt)
            },
            (&ExprKind::Field(l_f_exp, ref l_f_ident), &ExprKind::Field(r_f_exp, ref r_f_ident)) => {
                l_f_ident.name == r_f_ident.name && self.eq_expr(l_f_exp, r_f_exp)
            },
            (&ExprKind::Index(la, li), &ExprKind::Index(ra, ri)) => self.eq_expr(la, ra) && self.eq_expr(li, ri),
            (&ExprKind::If(lc, lt, ref le), &ExprKind::If(rc, rt, ref re)) => {
                self.eq_expr(lc, rc) && self.eq_expr(lt, rt) && both(le, re, |l, r| self.eq_expr(l, r))
            },
            (&ExprKind::Let(l), &ExprKind::Let(r)) => {
                self.eq_pat(l.pat, r.pat) && both(&l.ty, &r.ty, |l, r| self.eq_ty(l, r)) && self.eq_expr(l.init, r.init)
            },
            (&ExprKind::Lit(ref l), &ExprKind::Lit(ref r)) => l.node == r.node,
            (&ExprKind::Loop(lb, ref ll, ref lls, _), &ExprKind::Loop(rb, ref rl, ref rls, _)) => {
                lls == rls && self.eq_block(lb, rb) && both(ll, rl, |l, r| l.ident.name == r.ident.name)
            },
            (&ExprKind::Match(le, la, ref ls), &ExprKind::Match(re, ra, ref rs)) => {
                ls == rs
                    && self.eq_expr(le, re)
                    && over(la, ra, |l, r| {
                        self.eq_pat(l.pat, r.pat)
                            && both(&l.guard, &r.guard, |l, r| self.eq_guard(l, r))
                            && self.eq_expr(l.body, r.body)
                    })
            },
            (
                &ExprKind::MethodCall(l_path, l_receiver, l_args, _),
                &ExprKind::MethodCall(r_path, r_receiver, r_args, _),
            ) => {
                self.inner.allow_side_effects
                    && self.eq_path_segment(l_path, r_path)
                    && self.eq_expr(l_receiver, r_receiver)
                    && self.eq_exprs(l_args, r_args)
            },
            (&ExprKind::Repeat(le, ll), &ExprKind::Repeat(re, rl)) => {
                self.eq_expr(le, re) && self.eq_array_length(ll, rl)
            },
            (&ExprKind::Ret(ref l), &ExprKind::Ret(ref r)) => both(l, r, |l, r| self.eq_expr(l, r)),
            (&ExprKind::Path(ref l), &ExprKind::Path(ref r)) => self.eq_qpath(l, r),
            (&ExprKind::Struct(l_path, lf, ref lo), &ExprKind::Struct(r_path, rf, ref ro)) => {
                self.eq_qpath(l_path, r_path)
                    && both(lo, ro, |l, r| self.eq_expr(l, r))
                    && over(lf, rf, |l, r| self.eq_expr_field(l, r))
            },
            (&ExprKind::Tup(l_tup), &ExprKind::Tup(r_tup)) => self.eq_exprs(l_tup, r_tup),
            (&ExprKind::Unary(l_op, le), &ExprKind::Unary(r_op, re)) => l_op == r_op && self.eq_expr(le, re),
            (&ExprKind::Array(l), &ExprKind::Array(r)) => self.eq_exprs(l, r),
            (&ExprKind::DropTemps(le), &ExprKind::DropTemps(re)) => self.eq_expr(le, re),
            _ => false,
        };
        (is_eq && (!self.should_ignore(left) || !self.should_ignore(right)))
            || self.inner.expr_fallback.as_mut().map_or(false, |f| f(left, right))
    }

    fn eq_exprs(&mut self, left: &[Expr<'_>], right: &[Expr<'_>]) -> bool {
        over(left, right, |l, r| self.eq_expr(l, r))
    }

    fn eq_expr_field(&mut self, left: &ExprField<'_>, right: &ExprField<'_>) -> bool {
        left.ident.name == right.ident.name && self.eq_expr(left.expr, right.expr)
    }

    fn eq_guard(&mut self, left: &Guard<'_>, right: &Guard<'_>) -> bool {
        match (left, right) {
            (Guard::If(l), Guard::If(r)) => self.eq_expr(l, r),
            (Guard::IfLet(l), Guard::IfLet(r)) => {
                self.eq_pat(l.pat, r.pat) && both(&l.ty, &r.ty, |l, r| self.eq_ty(l, r)) && self.eq_expr(l.init, r.init)
            },
            _ => false,
        }
    }

    fn eq_generic_arg(&mut self, left: &GenericArg<'_>, right: &GenericArg<'_>) -> bool {
        match (left, right) {
            (GenericArg::Const(l), GenericArg::Const(r)) => self.eq_body(l.value.body, r.value.body),
            (GenericArg::Lifetime(l_lt), GenericArg::Lifetime(r_lt)) => Self::eq_lifetime(l_lt, r_lt),
            (GenericArg::Type(l_ty), GenericArg::Type(r_ty)) => self.eq_ty(l_ty, r_ty),
            (GenericArg::Infer(l_inf), GenericArg::Infer(r_inf)) => self.eq_ty(&l_inf.to_ty(), &r_inf.to_ty()),
            _ => false,
        }
    }

    fn eq_lifetime(left: &Lifetime, right: &Lifetime) -> bool {
        left.name == right.name
    }

    fn eq_pat_field(&mut self, left: &PatField<'_>, right: &PatField<'_>) -> bool {
        let (PatField { ident: li, pat: lp, .. }, PatField { ident: ri, pat: rp, .. }) = (&left, &right);
        li.name == ri.name && self.eq_pat(lp, rp)
    }

    /// Checks whether two patterns are the same.
    fn eq_pat(&mut self, left: &Pat<'_>, right: &Pat<'_>) -> bool {
        match (&left.kind, &right.kind) {
            (&PatKind::Box(l), &PatKind::Box(r)) => self.eq_pat(l, r),
            (&PatKind::Struct(ref lp, la, ..), &PatKind::Struct(ref rp, ra, ..)) => {
                self.eq_qpath(lp, rp) && over(la, ra, |l, r| self.eq_pat_field(l, r))
            },
            (&PatKind::TupleStruct(ref lp, la, ls), &PatKind::TupleStruct(ref rp, ra, rs)) => {
                self.eq_qpath(lp, rp) && over(la, ra, |l, r| self.eq_pat(l, r)) && ls == rs
            },
            (&PatKind::Binding(lb, li, _, ref lp), &PatKind::Binding(rb, ri, _, ref rp)) => {
                let eq = lb == rb && both(lp, rp, |l, r| self.eq_pat(l, r));
                if eq {
                    self.locals.insert(li, ri);
                }
                eq
            },
            (&PatKind::Path(ref l), &PatKind::Path(ref r)) => self.eq_qpath(l, r),
            (&PatKind::Lit(l), &PatKind::Lit(r)) => self.eq_expr(l, r),
            (&PatKind::Tuple(l, ls), &PatKind::Tuple(r, rs)) => ls == rs && over(l, r, |l, r| self.eq_pat(l, r)),
            (&PatKind::Range(ref ls, ref le, li), &PatKind::Range(ref rs, ref re, ri)) => {
                both(ls, rs, |a, b| self.eq_expr(a, b)) && both(le, re, |a, b| self.eq_expr(a, b)) && (li == ri)
            },
            (&PatKind::Ref(le, ref lm), &PatKind::Ref(re, ref rm)) => lm == rm && self.eq_pat(le, re),
            (&PatKind::Slice(ls, ref li, le), &PatKind::Slice(rs, ref ri, re)) => {
                over(ls, rs, |l, r| self.eq_pat(l, r))
                    && over(le, re, |l, r| self.eq_pat(l, r))
                    && both(li, ri, |l, r| self.eq_pat(l, r))
            },
            (&PatKind::Wild, &PatKind::Wild) => true,
            _ => false,
        }
    }

    #[expect(clippy::similar_names)]
    fn eq_qpath(&mut self, left: &QPath<'_>, right: &QPath<'_>) -> bool {
        match (left, right) {
            (&QPath::Resolved(ref lty, lpath), &QPath::Resolved(ref rty, rpath)) => {
                both(lty, rty, |l, r| self.eq_ty(l, r)) && self.eq_path(lpath, rpath)
            },
            (&QPath::TypeRelative(lty, lseg), &QPath::TypeRelative(rty, rseg)) => {
                self.eq_ty(lty, rty) && self.eq_path_segment(lseg, rseg)
            },
            (&QPath::LangItem(llang_item, ..), &QPath::LangItem(rlang_item, ..)) => llang_item == rlang_item,
            _ => false,
        }
    }

    pub fn eq_path(&mut self, left: &Path<'_>, right: &Path<'_>) -> bool {
        match (left.res, right.res) {
            (Res::Local(l), Res::Local(r)) => l == r || self.locals.get(&l) == Some(&r),
            (Res::Local(_), _) | (_, Res::Local(_)) => false,
            _ => over(left.segments, right.segments, |l, r| self.eq_path_segment(l, r)),
        }
    }

    fn eq_path_parameters(&mut self, left: &GenericArgs<'_>, right: &GenericArgs<'_>) -> bool {
        if !(left.parenthesized || right.parenthesized) {
            over(left.args, right.args, |l, r| self.eq_generic_arg(l, r)) // FIXME(flip1995): may not work
                && over(left.bindings, right.bindings, |l, r| self.eq_type_binding(l, r))
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
        left.ident.name == right.ident.name && both(&left.args, &right.args, |l, r| self.eq_path_parameters(l, r))
    }

    pub fn eq_ty(&mut self, left: &Ty<'_>, right: &Ty<'_>) -> bool {
        match (&left.kind, &right.kind) {
            (&TyKind::Slice(l_vec), &TyKind::Slice(r_vec)) => self.eq_ty(l_vec, r_vec),
            (&TyKind::Array(lt, ll), &TyKind::Array(rt, rl)) => self.eq_ty(lt, rt) && self.eq_array_length(ll, rl),
            (&TyKind::Ptr(ref l_mut), &TyKind::Ptr(ref r_mut)) => {
                l_mut.mutbl == r_mut.mutbl && self.eq_ty(l_mut.ty, r_mut.ty)
            },
            (&TyKind::Rptr(_, ref l_rmut), &TyKind::Rptr(_, ref r_rmut)) => {
                l_rmut.mutbl == r_rmut.mutbl && self.eq_ty(l_rmut.ty, r_rmut.ty)
            },
            (&TyKind::Path(ref l), &TyKind::Path(ref r)) => self.eq_qpath(l, r),
            (&TyKind::Tup(l), &TyKind::Tup(r)) => over(l, r, |l, r| self.eq_ty(l, r)),
            (&TyKind::Infer, &TyKind::Infer) => true,
            _ => false,
        }
    }

    fn eq_type_binding(&mut self, left: &TypeBinding<'_>, right: &TypeBinding<'_>) -> bool {
        left.ident.name == right.ident.name && self.eq_ty(left.ty(), right.ty())
    }
}

/// Some simple reductions like `{ return }` => `return`
fn reduce_exprkind<'hir>(cx: &LateContext<'_>, kind: &'hir ExprKind<'hir>) -> &'hir ExprKind<'hir> {
    if let ExprKind::Block(block, _) = kind {
        match (block.stmts, block.expr) {
            // From an `if let` expression without an `else` block. The arm for the implicit wild pattern is an empty
            // block with an empty span.
            ([], None) if block.span.is_empty() => &ExprKind::Tup(&[]),
            // `{}` => `()`
            ([], None) => match snippet_opt(cx, block.span) {
                // Don't reduce if there are any tokens contained in the braces
                Some(snip)
                    if tokenize(&snip)
                        .map(|t| t.kind)
                        .filter(|t| {
                            !matches!(
                                t,
                                TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace
                            )
                        })
                        .ne([TokenKind::OpenBrace, TokenKind::CloseBrace].iter().copied()) =>
                {
                    kind
                },
                _ => &ExprKind::Tup(&[]),
            },
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

/// Counts how many elements of the slices are equal as per `eq_fn`.
pub fn count_eq<X: Sized>(
    left: &mut dyn Iterator<Item = X>,
    right: &mut dyn Iterator<Item = X>,
    mut eq_fn: impl FnMut(&X, &X) -> bool,
) -> usize {
    left.zip(right).take_while(|(l, r)| eq_fn(l, r)).count()
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
    s: FxHasher,
}

impl<'a, 'tcx> SpanlessHash<'a, 'tcx> {
    pub fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results(),
            s: FxHasher::default(),
        }
    }

    pub fn finish(self) -> u64 {
        self.s.finish()
    }

    pub fn hash_block(&mut self, b: &Block<'_>) {
        for s in b.stmts {
            self.hash_stmt(s);
        }

        if let Some(e) = b.expr {
            self.hash_expr(e);
        }

        std::mem::discriminant(&b.rules).hash(&mut self.s);
    }

    #[expect(clippy::too_many_lines)]
    pub fn hash_expr(&mut self, e: &Expr<'_>) {
        let simple_const = self
            .maybe_typeck_results
            .and_then(|typeck_results| constant_simple(self.cx, typeck_results, e));

        // const hashing may result in the same hash as some unrelated node, so add a sort of
        // discriminant depending on which path we're choosing next
        simple_const.hash(&mut self.s);
        if simple_const.is_some() {
            return;
        }

        std::mem::discriminant(&e.kind).hash(&mut self.s);

        match e.kind {
            ExprKind::AddrOf(kind, m, e) => {
                std::mem::discriminant(&kind).hash(&mut self.s);
                m.hash(&mut self.s);
                self.hash_expr(e);
            },
            ExprKind::Continue(i) => {
                if let Some(i) = i.label {
                    self.hash_name(i.ident.name);
                }
            },
            ExprKind::Assign(l, r, _) => {
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::AssignOp(ref o, l, r) => {
                std::mem::discriminant(&o.node).hash(&mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::Block(b, _) => {
                self.hash_block(b);
            },
            ExprKind::Binary(op, l, r) => {
                std::mem::discriminant(&op.node).hash(&mut self.s);
                self.hash_expr(l);
                self.hash_expr(r);
            },
            ExprKind::Break(i, ref j) => {
                if let Some(i) = i.label {
                    self.hash_name(i.ident.name);
                }
                if let Some(j) = *j {
                    self.hash_expr(j);
                }
            },
            ExprKind::Box(e) | ExprKind::DropTemps(e) | ExprKind::Yield(e, _) => {
                self.hash_expr(e);
            },
            ExprKind::Call(fun, args) => {
                self.hash_expr(fun);
                self.hash_exprs(args);
            },
            ExprKind::Cast(e, ty) | ExprKind::Type(e, ty) => {
                self.hash_expr(e);
                self.hash_ty(ty);
            },
            ExprKind::Closure(&Closure {
                capture_clause, body, ..
            }) => {
                std::mem::discriminant(&capture_clause).hash(&mut self.s);
                // closures inherit TypeckResults
                self.hash_expr(&self.cx.tcx.hir().body(body).value);
            },
            ExprKind::Field(e, ref f) => {
                self.hash_expr(e);
                self.hash_name(f.name);
            },
            ExprKind::Index(a, i) => {
                self.hash_expr(a);
                self.hash_expr(i);
            },
            ExprKind::InlineAsm(asm) => {
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
                        InlineAsmOperand::Const { anon_const } | InlineAsmOperand::SymFn { anon_const } => {
                            self.hash_body(anon_const.body);
                        },
                        InlineAsmOperand::SymStatic { path, def_id: _ } => self.hash_qpath(path),
                    }
                }
            },
            ExprKind::Let(Let { pat, init, ty, .. }) => {
                self.hash_expr(init);
                if let Some(ty) = ty {
                    self.hash_ty(ty);
                }
                self.hash_pat(pat);
            },
            ExprKind::Err => {},
            ExprKind::Lit(ref l) => {
                l.node.hash(&mut self.s);
            },
            ExprKind::Loop(b, ref i, ..) => {
                self.hash_block(b);
                if let Some(i) = *i {
                    self.hash_name(i.ident.name);
                }
            },
            ExprKind::If(cond, then, ref else_opt) => {
                self.hash_expr(cond);
                self.hash_expr(then);
                if let Some(e) = *else_opt {
                    self.hash_expr(e);
                }
            },
            ExprKind::Match(e, arms, ref s) => {
                self.hash_expr(e);

                for arm in arms {
                    self.hash_pat(arm.pat);
                    if let Some(ref e) = arm.guard {
                        self.hash_guard(e);
                    }
                    self.hash_expr(arm.body);
                }

                s.hash(&mut self.s);
            },
            ExprKind::MethodCall(path, receiver, args, ref _fn_span) => {
                self.hash_name(path.ident.name);
                self.hash_expr(receiver);
                self.hash_exprs(args);
            },
            ExprKind::ConstBlock(ref l_id) => {
                self.hash_body(l_id.body);
            },
            ExprKind::Repeat(e, len) => {
                self.hash_expr(e);
                self.hash_array_length(len);
            },
            ExprKind::Ret(ref e) => {
                if let Some(e) = *e {
                    self.hash_expr(e);
                }
            },
            ExprKind::Path(ref qpath) => {
                self.hash_qpath(qpath);
            },
            ExprKind::Struct(path, fields, ref expr) => {
                self.hash_qpath(path);

                for f in fields {
                    self.hash_name(f.ident.name);
                    self.hash_expr(f.expr);
                }

                if let Some(e) = *expr {
                    self.hash_expr(e);
                }
            },
            ExprKind::Tup(tup) => {
                self.hash_exprs(tup);
            },
            ExprKind::Array(v) => {
                self.hash_exprs(v);
            },
            ExprKind::Unary(lop, le) => {
                std::mem::discriminant(&lop).hash(&mut self.s);
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
        n.hash(&mut self.s);
    }

    pub fn hash_qpath(&mut self, p: &QPath<'_>) {
        match *p {
            QPath::Resolved(_, path) => {
                self.hash_path(path);
            },
            QPath::TypeRelative(_, path) => {
                self.hash_name(path.ident.name);
            },
            QPath::LangItem(lang_item, ..) => {
                std::mem::discriminant(&lang_item).hash(&mut self.s);
            },
        }
        // self.maybe_typeck_results.unwrap().qpath_res(p, id).hash(&mut self.s);
    }

    pub fn hash_pat(&mut self, pat: &Pat<'_>) {
        std::mem::discriminant(&pat.kind).hash(&mut self.s);
        match pat.kind {
            PatKind::Binding(BindingAnnotation(by_ref, mutability), _, _, pat) => {
                std::mem::discriminant(&by_ref).hash(&mut self.s);
                std::mem::discriminant(&mutability).hash(&mut self.s);
                if let Some(pat) = pat {
                    self.hash_pat(pat);
                }
            },
            PatKind::Box(pat) => self.hash_pat(pat),
            PatKind::Lit(expr) => self.hash_expr(expr),
            PatKind::Or(pats) => {
                for pat in pats {
                    self.hash_pat(pat);
                }
            },
            PatKind::Path(ref qpath) => self.hash_qpath(qpath),
            PatKind::Range(s, e, i) => {
                if let Some(s) = s {
                    self.hash_expr(s);
                }
                if let Some(e) = e {
                    self.hash_expr(e);
                }
                std::mem::discriminant(&i).hash(&mut self.s);
            },
            PatKind::Ref(pat, mu) => {
                self.hash_pat(pat);
                std::mem::discriminant(&mu).hash(&mut self.s);
            },
            PatKind::Slice(l, m, r) => {
                for pat in l {
                    self.hash_pat(pat);
                }
                if let Some(pat) = m {
                    self.hash_pat(pat);
                }
                for pat in r {
                    self.hash_pat(pat);
                }
            },
            PatKind::Struct(ref qpath, fields, e) => {
                self.hash_qpath(qpath);
                for f in fields {
                    self.hash_name(f.ident.name);
                    self.hash_pat(f.pat);
                }
                e.hash(&mut self.s);
            },
            PatKind::Tuple(pats, e) => {
                for pat in pats {
                    self.hash_pat(pat);
                }
                e.hash(&mut self.s);
            },
            PatKind::TupleStruct(ref qpath, pats, e) => {
                self.hash_qpath(qpath);
                for pat in pats {
                    self.hash_pat(pat);
                }
                e.hash(&mut self.s);
            },
            PatKind::Wild => {},
        }
    }

    pub fn hash_path(&mut self, path: &Path<'_>) {
        match path.res {
            // constant hash since equality is dependant on inter-expression context
            // e.g. The expressions `if let Some(x) = foo() {}` and `if let Some(y) = foo() {}` are considered equal
            // even though the binding names are different and they have different `HirId`s.
            Res::Local(_) => 1_usize.hash(&mut self.s),
            _ => {
                for seg in path.segments {
                    self.hash_name(seg.ident.name);
                    self.hash_generic_args(seg.args().args);
                }
            },
        }
    }

    pub fn hash_stmt(&mut self, b: &Stmt<'_>) {
        std::mem::discriminant(&b.kind).hash(&mut self.s);

        match &b.kind {
            StmtKind::Local(local) => {
                self.hash_pat(local.pat);
                if let Some(init) = local.init {
                    self.hash_expr(init);
                }
                if let Some(els) = local.els {
                    self.hash_block(els);
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
            Guard::If(expr) | Guard::IfLet(Let { init: expr, .. }) => {
                self.hash_expr(expr);
            },
        }
    }

    pub fn hash_lifetime(&mut self, lifetime: &Lifetime) {
        std::mem::discriminant(&lifetime.name).hash(&mut self.s);
        if let LifetimeName::Param(param_id, ref name) = lifetime.name {
            std::mem::discriminant(name).hash(&mut self.s);
            param_id.hash(&mut self.s);
            match name {
                ParamName::Plain(ref ident) => {
                    ident.name.hash(&mut self.s);
                },
                ParamName::Fresh | ParamName::Error => {},
            }
        }
    }

    pub fn hash_ty(&mut self, ty: &Ty<'_>) {
        std::mem::discriminant(&ty.kind).hash(&mut self.s);
        self.hash_tykind(&ty.kind);
    }

    pub fn hash_tykind(&mut self, ty: &TyKind<'_>) {
        match ty {
            TyKind::Slice(ty) => {
                self.hash_ty(ty);
            },
            &TyKind::Array(ty, len) => {
                self.hash_ty(ty);
                self.hash_array_length(len);
            },
            TyKind::Ptr(ref mut_ty) => {
                self.hash_ty(mut_ty.ty);
                mut_ty.mutbl.hash(&mut self.s);
            },
            TyKind::Rptr(lifetime, ref mut_ty) => {
                self.hash_lifetime(*lifetime);
                self.hash_ty(mut_ty.ty);
                mut_ty.mutbl.hash(&mut self.s);
            },
            TyKind::BareFn(bfn) => {
                bfn.unsafety.hash(&mut self.s);
                bfn.abi.hash(&mut self.s);
                for arg in bfn.decl.inputs {
                    self.hash_ty(arg);
                }
                std::mem::discriminant(&bfn.decl.output).hash(&mut self.s);
                match bfn.decl.output {
                    FnRetTy::DefaultReturn(_) => {},
                    FnRetTy::Return(ty) => {
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
            TyKind::Path(ref qpath) => self.hash_qpath(qpath),
            TyKind::OpaqueDef(_, arg_list, in_trait) => {
                self.hash_generic_args(arg_list);
                in_trait.hash(&mut self.s);
            },
            TyKind::TraitObject(_, lifetime, _) => {
                self.hash_lifetime(*lifetime);
            },
            TyKind::Typeof(anon_const) => {
                self.hash_body(anon_const.body);
            },
            TyKind::Err | TyKind::Infer | TyKind::Never => {},
        }
    }

    pub fn hash_array_length(&mut self, length: ArrayLen) {
        match length {
            ArrayLen::Infer(..) => {},
            ArrayLen::Body(anon_const) => self.hash_body(anon_const.body),
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
            match *arg {
                GenericArg::Lifetime(l) => self.hash_lifetime(l),
                GenericArg::Type(ref ty) => self.hash_ty(ty),
                GenericArg::Const(ref ca) => self.hash_body(ca.value.body),
                GenericArg::Infer(ref inf) => self.hash_ty(&inf.to_ty()),
            }
        }
    }
}

pub fn hash_stmt(cx: &LateContext<'_>, s: &Stmt<'_>) -> u64 {
    let mut h = SpanlessHash::new(cx);
    h.hash_stmt(s);
    h.finish()
}

pub fn hash_expr(cx: &LateContext<'_>, e: &Expr<'_>) -> u64 {
    let mut h = SpanlessHash::new(cx);
    h.hash_expr(e);
    h.finish()
}
