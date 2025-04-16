use std::assert_matches::assert_matches;
use std::ops::ControlFlow;
use std::sync::Arc;

use rustc_ast::ptr::P as AstP;
use rustc_ast::*;
use rustc_ast_pretty::pprust::expr_to_string;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::def::{DefKind, Res};
use rustc_middle::span_bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::errors::report_lit_error;
use rustc_span::source_map::{Spanned, respan};
use rustc_span::{DUMMY_SP, DesugaringKind, Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};
use visit::{Visitor, walk_expr};

use super::errors::{
    AsyncCoroutinesNotSupported, AwaitOnlyInAsyncFnAndBlocks, ClosureCannotBeStatic,
    CoroutineTooManyParameters, FunctionalRecordUpdateDestructuringAssignment,
    InclusiveRangeWithNoEnd, MatchArmWithNoBody, NeverPatternWithBody, NeverPatternWithGuard,
    UnderscoreExprLhsAssign,
};
use super::{
    GenericArgsMode, ImplTraitContext, LoweringContext, ParamMode, ResolverAstLoweringExt,
};
use crate::errors::{InvalidLegacyConstGenericArg, UseConstGenericArg, YieldInClosure};
use crate::{AllowReturnTypeNotation, FnDeclKind, ImplTraitPosition, fluent_generated};

struct WillCreateDefIdsVisitor {}

impl<'v> rustc_ast::visit::Visitor<'v> for WillCreateDefIdsVisitor {
    type Result = ControlFlow<Span>;

    fn visit_anon_const(&mut self, c: &'v AnonConst) -> Self::Result {
        ControlFlow::Break(c.value.span)
    }

    fn visit_item(&mut self, item: &'v Item) -> Self::Result {
        ControlFlow::Break(item.span)
    }

    fn visit_expr(&mut self, ex: &'v Expr) -> Self::Result {
        match ex.kind {
            ExprKind::Gen(..) | ExprKind::ConstBlock(..) | ExprKind::Closure(..) => {
                ControlFlow::Break(ex.span)
            }
            _ => walk_expr(self, ex),
        }
    }
}

impl<'hir> LoweringContext<'_, 'hir> {
    fn lower_exprs(&mut self, exprs: &[AstP<Expr>]) -> &'hir [hir::Expr<'hir>] {
        self.arena.alloc_from_iter(exprs.iter().map(|x| self.lower_expr_mut(x)))
    }

    pub(super) fn lower_expr(&mut self, e: &Expr) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.lower_expr_mut(e))
    }

    pub(super) fn lower_expr_mut(&mut self, e: &Expr) -> hir::Expr<'hir> {
        ensure_sufficient_stack(|| {
            match &e.kind {
                // Parenthesis expression does not have a HirId and is handled specially.
                ExprKind::Paren(ex) => {
                    let mut ex = self.lower_expr_mut(ex);
                    // Include parens in span, but only if it is a super-span.
                    if e.span.contains(ex.span) {
                        ex.span = self.lower_span(e.span);
                    }
                    // Merge attributes into the inner expression.
                    if !e.attrs.is_empty() {
                        let old_attrs = self.attrs.get(&ex.hir_id.local_id).copied().unwrap_or(&[]);
                        let attrs = &*self.arena.alloc_from_iter(
                            self.lower_attrs_vec(&e.attrs, e.span)
                                .into_iter()
                                .chain(old_attrs.iter().cloned()),
                        );
                        if attrs.is_empty() {
                            return ex;
                        }

                        self.attrs.insert(ex.hir_id.local_id, attrs);
                    }
                    return ex;
                }
                // Desugar `ExprForLoop`
                // from: `[opt_ident]: for await? <pat> in <iter> <body>`
                //
                // This also needs special handling because the HirId of the returned `hir::Expr` will not
                // correspond to the `e.id`, so `lower_expr_for` handles attribute lowering itself.
                ExprKind::ForLoop { pat, iter, body, label, kind } => {
                    return self.lower_expr_for(e, pat, iter, body, *label, *kind);
                }
                _ => (),
            }

            let expr_hir_id = self.lower_node_id(e.id);
            self.lower_attrs(expr_hir_id, &e.attrs, e.span);

            let kind = match &e.kind {
                ExprKind::Array(exprs) => hir::ExprKind::Array(self.lower_exprs(exprs)),
                ExprKind::ConstBlock(c) => hir::ExprKind::ConstBlock(self.lower_const_block(c)),
                ExprKind::Repeat(expr, count) => {
                    let expr = self.lower_expr(expr);
                    let count = self.lower_array_length_to_const_arg(count);
                    hir::ExprKind::Repeat(expr, count)
                }
                ExprKind::Tup(elts) => hir::ExprKind::Tup(self.lower_exprs(elts)),
                ExprKind::Call(f, args) => {
                    if let Some(legacy_args) = self.resolver.legacy_const_generic_args(f) {
                        self.lower_legacy_const_generics((**f).clone(), args.clone(), &legacy_args)
                    } else {
                        let f = self.lower_expr(f);
                        hir::ExprKind::Call(f, self.lower_exprs(args))
                    }
                }
                ExprKind::MethodCall(box MethodCall { seg, receiver, args, span }) => {
                    let hir_seg = self.arena.alloc(self.lower_path_segment(
                        e.span,
                        seg,
                        ParamMode::Optional,
                        GenericArgsMode::Err,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                        // Method calls can't have bound modifiers
                        None,
                    ));
                    let receiver = self.lower_expr(receiver);
                    let args =
                        self.arena.alloc_from_iter(args.iter().map(|x| self.lower_expr_mut(x)));
                    hir::ExprKind::MethodCall(hir_seg, receiver, args, self.lower_span(*span))
                }
                ExprKind::Binary(binop, lhs, rhs) => {
                    let binop = self.lower_binop(*binop);
                    let lhs = self.lower_expr(lhs);
                    let rhs = self.lower_expr(rhs);
                    hir::ExprKind::Binary(binop, lhs, rhs)
                }
                ExprKind::Unary(op, ohs) => {
                    let op = self.lower_unop(*op);
                    let ohs = self.lower_expr(ohs);
                    hir::ExprKind::Unary(op, ohs)
                }
                ExprKind::Lit(token_lit) => hir::ExprKind::Lit(self.lower_lit(token_lit, e.span)),
                ExprKind::IncludedBytes(bytes) => {
                    let lit = self.arena.alloc(respan(
                        self.lower_span(e.span),
                        LitKind::ByteStr(Arc::clone(bytes), StrStyle::Cooked),
                    ));
                    hir::ExprKind::Lit(lit)
                }
                ExprKind::Cast(expr, ty) => {
                    let expr = self.lower_expr(expr);
                    let ty =
                        self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Cast));
                    hir::ExprKind::Cast(expr, ty)
                }
                ExprKind::Type(expr, ty) => {
                    let expr = self.lower_expr(expr);
                    let ty =
                        self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Cast));
                    hir::ExprKind::Type(expr, ty)
                }
                ExprKind::AddrOf(k, m, ohs) => {
                    let ohs = self.lower_expr(ohs);
                    hir::ExprKind::AddrOf(*k, *m, ohs)
                }
                ExprKind::Let(pat, scrutinee, span, recovered) => {
                    hir::ExprKind::Let(self.arena.alloc(hir::LetExpr {
                        span: self.lower_span(*span),
                        pat: self.lower_pat(pat),
                        ty: None,
                        init: self.lower_expr(scrutinee),
                        recovered: *recovered,
                    }))
                }
                ExprKind::If(cond, then, else_opt) => {
                    self.lower_expr_if(cond, then, else_opt.as_deref())
                }
                ExprKind::While(cond, body, opt_label) => {
                    self.with_loop_scope(expr_hir_id, |this| {
                        let span =
                            this.mark_span_with_reason(DesugaringKind::WhileLoop, e.span, None);
                        let opt_label = this.lower_label(*opt_label, e.id, expr_hir_id);
                        this.lower_expr_while_in_loop_scope(span, cond, body, opt_label)
                    })
                }
                ExprKind::Loop(body, opt_label, span) => {
                    self.with_loop_scope(expr_hir_id, |this| {
                        let opt_label = this.lower_label(*opt_label, e.id, expr_hir_id);
                        hir::ExprKind::Loop(
                            this.lower_block(body, false),
                            opt_label,
                            hir::LoopSource::Loop,
                            this.lower_span(*span),
                        )
                    })
                }
                ExprKind::TryBlock(body) => self.lower_expr_try_block(body),
                ExprKind::Match(expr, arms, kind) => hir::ExprKind::Match(
                    self.lower_expr(expr),
                    self.arena.alloc_from_iter(arms.iter().map(|x| self.lower_arm(x))),
                    match kind {
                        MatchKind::Prefix => hir::MatchSource::Normal,
                        MatchKind::Postfix => hir::MatchSource::Postfix,
                    },
                ),
                ExprKind::Await(expr, await_kw_span) => self.lower_expr_await(*await_kw_span, expr),
                ExprKind::Use(expr, use_kw_span) => self.lower_expr_use(*use_kw_span, expr),
                ExprKind::Closure(box Closure {
                    binder,
                    capture_clause,
                    constness,
                    coroutine_kind,
                    movability,
                    fn_decl,
                    body,
                    fn_decl_span,
                    fn_arg_span,
                }) => match coroutine_kind {
                    Some(coroutine_kind) => self.lower_expr_coroutine_closure(
                        binder,
                        *capture_clause,
                        e.id,
                        expr_hir_id,
                        *coroutine_kind,
                        fn_decl,
                        body,
                        *fn_decl_span,
                        *fn_arg_span,
                    ),
                    None => self.lower_expr_closure(
                        binder,
                        *capture_clause,
                        e.id,
                        expr_hir_id,
                        *constness,
                        *movability,
                        fn_decl,
                        body,
                        *fn_decl_span,
                        *fn_arg_span,
                    ),
                },
                ExprKind::Gen(capture_clause, block, genblock_kind, decl_span) => {
                    let desugaring_kind = match genblock_kind {
                        GenBlockKind::Async => hir::CoroutineDesugaring::Async,
                        GenBlockKind::Gen => hir::CoroutineDesugaring::Gen,
                        GenBlockKind::AsyncGen => hir::CoroutineDesugaring::AsyncGen,
                    };
                    self.make_desugared_coroutine_expr(
                        *capture_clause,
                        e.id,
                        None,
                        *decl_span,
                        e.span,
                        desugaring_kind,
                        hir::CoroutineSource::Block,
                        |this| this.with_new_scopes(e.span, |this| this.lower_block_expr(block)),
                    )
                }
                ExprKind::Block(blk, opt_label) => {
                    // Different from loops, label of block resolves to block id rather than
                    // expr node id.
                    let block_hir_id = self.lower_node_id(blk.id);
                    let opt_label = self.lower_label(*opt_label, blk.id, block_hir_id);
                    let hir_block = self.arena.alloc(self.lower_block_noalloc(
                        block_hir_id,
                        blk,
                        opt_label.is_some(),
                    ));
                    hir::ExprKind::Block(hir_block, opt_label)
                }
                ExprKind::Assign(el, er, span) => self.lower_expr_assign(el, er, *span, e.span),
                ExprKind::AssignOp(op, el, er) => hir::ExprKind::AssignOp(
                    self.lower_assign_op(*op),
                    self.lower_expr(el),
                    self.lower_expr(er),
                ),
                ExprKind::Field(el, ident) => {
                    hir::ExprKind::Field(self.lower_expr(el), self.lower_ident(*ident))
                }
                ExprKind::Index(el, er, brackets_span) => {
                    hir::ExprKind::Index(self.lower_expr(el), self.lower_expr(er), *brackets_span)
                }
                ExprKind::Range(e1, e2, lims) => {
                    self.lower_expr_range(e.span, e1.as_deref(), e2.as_deref(), *lims)
                }
                ExprKind::Underscore => {
                    let guar = self.dcx().emit_err(UnderscoreExprLhsAssign { span: e.span });
                    hir::ExprKind::Err(guar)
                }
                ExprKind::Path(qself, path) => {
                    let qpath = self.lower_qpath(
                        e.id,
                        qself,
                        path,
                        ParamMode::Optional,
                        AllowReturnTypeNotation::No,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                        None,
                    );
                    hir::ExprKind::Path(qpath)
                }
                ExprKind::Break(opt_label, opt_expr) => {
                    let opt_expr = opt_expr.as_ref().map(|x| self.lower_expr(x));
                    hir::ExprKind::Break(self.lower_jump_destination(e.id, *opt_label), opt_expr)
                }
                ExprKind::Continue(opt_label) => {
                    hir::ExprKind::Continue(self.lower_jump_destination(e.id, *opt_label))
                }
                ExprKind::Ret(e) => {
                    let expr = e.as_ref().map(|x| self.lower_expr(x));
                    self.checked_return(expr)
                }
                ExprKind::Yeet(sub_expr) => self.lower_expr_yeet(e.span, sub_expr.as_deref()),
                ExprKind::Become(sub_expr) => {
                    let sub_expr = self.lower_expr(sub_expr);
                    hir::ExprKind::Become(sub_expr)
                }
                ExprKind::InlineAsm(asm) => {
                    hir::ExprKind::InlineAsm(self.lower_inline_asm(e.span, asm))
                }
                ExprKind::FormatArgs(fmt) => self.lower_format_args(e.span, fmt),
                ExprKind::OffsetOf(container, fields) => hir::ExprKind::OffsetOf(
                    self.lower_ty(
                        container,
                        ImplTraitContext::Disallowed(ImplTraitPosition::OffsetOf),
                    ),
                    self.arena.alloc_from_iter(fields.iter().map(|&ident| self.lower_ident(ident))),
                ),
                ExprKind::Struct(se) => {
                    let rest = match &se.rest {
                        StructRest::Base(e) => hir::StructTailExpr::Base(self.lower_expr(e)),
                        StructRest::Rest(sp) => hir::StructTailExpr::DefaultFields(*sp),
                        StructRest::None => hir::StructTailExpr::None,
                    };
                    hir::ExprKind::Struct(
                        self.arena.alloc(self.lower_qpath(
                            e.id,
                            &se.qself,
                            &se.path,
                            ParamMode::Optional,
                            AllowReturnTypeNotation::No,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                            None,
                        )),
                        self.arena
                            .alloc_from_iter(se.fields.iter().map(|x| self.lower_expr_field(x))),
                        rest,
                    )
                }
                ExprKind::Yield(kind) => self.lower_expr_yield(e.span, kind.expr().map(|x| &**x)),
                ExprKind::Err(guar) => hir::ExprKind::Err(*guar),

                ExprKind::UnsafeBinderCast(kind, expr, ty) => hir::ExprKind::UnsafeBinderCast(
                    *kind,
                    self.lower_expr(expr),
                    ty.as_ref().map(|ty| {
                        self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Cast))
                    }),
                ),

                ExprKind::Dummy => {
                    span_bug!(e.span, "lowered ExprKind::Dummy")
                }

                ExprKind::Try(sub_expr) => self.lower_expr_try(e.span, sub_expr),

                ExprKind::Paren(_) | ExprKind::ForLoop { .. } => {
                    unreachable!("already handled")
                }

                ExprKind::MacCall(_) => panic!("{:?} shouldn't exist here", e.span),
            };

            hir::Expr { hir_id: expr_hir_id, kind, span: self.lower_span(e.span) }
        })
    }

    /// Create an `ExprKind::Ret` that is optionally wrapped by a call to check
    /// a contract ensures clause, if it exists.
    fn checked_return(&mut self, opt_expr: Option<&'hir hir::Expr<'hir>>) -> hir::ExprKind<'hir> {
        let checked_ret =
            if let Some((check_span, check_ident, check_hir_id)) = self.contract_ensures {
                let expr = opt_expr.unwrap_or_else(|| self.expr_unit(check_span));
                Some(self.inject_ensures_check(expr, check_span, check_ident, check_hir_id))
            } else {
                opt_expr
            };
        hir::ExprKind::Ret(checked_ret)
    }

    /// Wraps an expression with a call to the ensures check before it gets returned.
    pub(crate) fn inject_ensures_check(
        &mut self,
        expr: &'hir hir::Expr<'hir>,
        span: Span,
        cond_ident: Ident,
        cond_hir_id: HirId,
    ) -> &'hir hir::Expr<'hir> {
        let cond_fn = self.expr_ident(span, cond_ident, cond_hir_id);
        let call_expr = self.expr_call_lang_item_fn_mut(
            span,
            hir::LangItem::ContractCheckEnsures,
            arena_vec![self; *cond_fn, *expr],
        );
        self.arena.alloc(call_expr)
    }

    pub(crate) fn lower_const_block(&mut self, c: &AnonConst) -> hir::ConstBlock {
        self.with_new_scopes(c.value.span, |this| {
            let def_id = this.local_def_id(c.id);
            hir::ConstBlock {
                def_id,
                hir_id: this.lower_node_id(c.id),
                body: this.lower_const_body(c.value.span, Some(&c.value)),
            }
        })
    }

    pub(crate) fn lower_lit(
        &mut self,
        token_lit: &token::Lit,
        span: Span,
    ) -> &'hir Spanned<LitKind> {
        let lit_kind = match LitKind::from_token_lit(*token_lit) {
            Ok(lit_kind) => lit_kind,
            Err(err) => {
                let guar = report_lit_error(&self.tcx.sess.psess, err, *token_lit, span);
                LitKind::Err(guar)
            }
        };
        self.arena.alloc(respan(self.lower_span(span), lit_kind))
    }

    fn lower_unop(&mut self, u: UnOp) -> hir::UnOp {
        match u {
            UnOp::Deref => hir::UnOp::Deref,
            UnOp::Not => hir::UnOp::Not,
            UnOp::Neg => hir::UnOp::Neg,
        }
    }

    fn lower_binop(&mut self, b: BinOp) -> BinOp {
        Spanned { node: b.node, span: self.lower_span(b.span) }
    }

    fn lower_assign_op(&mut self, a: AssignOp) -> AssignOp {
        Spanned { node: a.node, span: self.lower_span(a.span) }
    }

    fn lower_legacy_const_generics(
        &mut self,
        mut f: Expr,
        args: ThinVec<AstP<Expr>>,
        legacy_args_idx: &[usize],
    ) -> hir::ExprKind<'hir> {
        let ExprKind::Path(None, path) = &mut f.kind else {
            unreachable!();
        };

        let mut error = None;
        let mut invalid_expr_error = |tcx: TyCtxt<'_>, span| {
            // Avoid emitting the error multiple times.
            if error.is_none() {
                let mut const_args = vec![];
                let mut other_args = vec![];
                for (idx, arg) in args.iter().enumerate() {
                    if legacy_args_idx.contains(&idx) {
                        const_args.push(format!("{{ {} }}", expr_to_string(arg)));
                    } else {
                        other_args.push(expr_to_string(arg));
                    }
                }
                let suggestion = UseConstGenericArg {
                    end_of_fn: f.span.shrink_to_hi(),
                    const_args: const_args.join(", "),
                    other_args: other_args.join(", "),
                    call_args: args[0].span.to(args.last().unwrap().span),
                };
                error = Some(tcx.dcx().emit_err(InvalidLegacyConstGenericArg { span, suggestion }));
            }
            error.unwrap()
        };

        // Split the arguments into const generics and normal arguments
        let mut real_args = vec![];
        let mut generic_args = ThinVec::new();
        for (idx, arg) in args.iter().cloned().enumerate() {
            if legacy_args_idx.contains(&idx) {
                let parent_def_id = self.current_hir_id_owner.def_id;
                let node_id = self.next_node_id();
                self.create_def(parent_def_id, node_id, None, DefKind::AnonConst, f.span);
                let mut visitor = WillCreateDefIdsVisitor {};
                let const_value = if let ControlFlow::Break(span) = visitor.visit_expr(&arg) {
                    AstP(Expr {
                        id: self.next_node_id(),
                        kind: ExprKind::Err(invalid_expr_error(self.tcx, span)),
                        span: f.span,
                        attrs: [].into(),
                        tokens: None,
                    })
                } else {
                    arg
                };

                let anon_const = AnonConst { id: node_id, value: const_value };
                generic_args.push(AngleBracketedArg::Arg(GenericArg::Const(anon_const)));
            } else {
                real_args.push(arg);
            }
        }

        // Add generic args to the last element of the path.
        let last_segment = path.segments.last_mut().unwrap();
        assert!(last_segment.args.is_none());
        last_segment.args = Some(AstP(GenericArgs::AngleBracketed(AngleBracketedArgs {
            span: DUMMY_SP,
            args: generic_args,
        })));

        // Now lower everything as normal.
        let f = self.lower_expr(&f);
        hir::ExprKind::Call(f, self.lower_exprs(&real_args))
    }

    fn lower_expr_if(
        &mut self,
        cond: &Expr,
        then: &Block,
        else_opt: Option<&Expr>,
    ) -> hir::ExprKind<'hir> {
        let lowered_cond = self.lower_cond(cond);
        let then_expr = self.lower_block_expr(then);
        if let Some(rslt) = else_opt {
            hir::ExprKind::If(
                lowered_cond,
                self.arena.alloc(then_expr),
                Some(self.lower_expr(rslt)),
            )
        } else {
            hir::ExprKind::If(lowered_cond, self.arena.alloc(then_expr), None)
        }
    }

    // Lowers a condition (i.e. `cond` in `if cond` or `while cond`), wrapping it in a terminating scope
    // so that temporaries created in the condition don't live beyond it.
    fn lower_cond(&mut self, cond: &Expr) -> &'hir hir::Expr<'hir> {
        fn has_let_expr(expr: &Expr) -> bool {
            match &expr.kind {
                ExprKind::Binary(_, lhs, rhs) => has_let_expr(lhs) || has_let_expr(rhs),
                ExprKind::Let(..) => true,
                _ => false,
            }
        }

        // We have to take special care for `let` exprs in the condition, e.g. in
        // `if let pat = val` or `if foo && let pat = val`, as we _do_ want `val` to live beyond the
        // condition in this case.
        //
        // In order to maintain the drop behavior for the non `let` parts of the condition,
        // we still wrap them in terminating scopes, e.g. `if foo && let pat = val` essentially
        // gets transformed into `if { let _t = foo; _t } && let pat = val`
        match &cond.kind {
            ExprKind::Binary(op @ Spanned { node: ast::BinOpKind::And, .. }, lhs, rhs)
                if has_let_expr(cond) =>
            {
                let op = self.lower_binop(*op);
                let lhs = self.lower_cond(lhs);
                let rhs = self.lower_cond(rhs);

                self.arena.alloc(self.expr(cond.span, hir::ExprKind::Binary(op, lhs, rhs)))
            }
            ExprKind::Let(..) => self.lower_expr(cond),
            _ => {
                let cond = self.lower_expr(cond);
                let reason = DesugaringKind::CondTemporary;
                let span_block = self.mark_span_with_reason(reason, cond.span, None);
                self.expr_drop_temps(span_block, cond)
            }
        }
    }

    // We desugar: `'label: while $cond $body` into:
    //
    // ```
    // 'label: loop {
    //   if { let _t = $cond; _t } {
    //     $body
    //   }
    //   else {
    //     break;
    //   }
    // }
    // ```
    //
    // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
    // to preserve drop semantics since `while $cond { ... }` does not
    // let temporaries live outside of `cond`.
    fn lower_expr_while_in_loop_scope(
        &mut self,
        span: Span,
        cond: &Expr,
        body: &Block,
        opt_label: Option<Label>,
    ) -> hir::ExprKind<'hir> {
        let lowered_cond = self.with_loop_condition_scope(|t| t.lower_cond(cond));
        let then = self.lower_block_expr(body);
        let expr_break = self.expr_break(span);
        let stmt_break = self.stmt_expr(span, expr_break);
        let else_blk = self.block_all(span, arena_vec![self; stmt_break], None);
        let else_expr = self.arena.alloc(self.expr_block(else_blk));
        let if_kind = hir::ExprKind::If(lowered_cond, self.arena.alloc(then), Some(else_expr));
        let if_expr = self.expr(span, if_kind);
        let block = self.block_expr(self.arena.alloc(if_expr));
        let span = self.lower_span(span.with_hi(cond.span.hi()));
        hir::ExprKind::Loop(block, opt_label, hir::LoopSource::While, span)
    }

    /// Desugar `try { <stmts>; <expr> }` into `{ <stmts>; ::std::ops::Try::from_output(<expr>) }`,
    /// `try { <stmts>; }` into `{ <stmts>; ::std::ops::Try::from_output(()) }`
    /// and save the block id to use it as a break target for desugaring of the `?` operator.
    fn lower_expr_try_block(&mut self, body: &Block) -> hir::ExprKind<'hir> {
        let body_hir_id = self.lower_node_id(body.id);
        self.with_catch_scope(body_hir_id, |this| {
            let mut block = this.lower_block_noalloc(body_hir_id, body, true);

            // Final expression of the block (if present) or `()` with span at the end of block
            let (try_span, tail_expr) = if let Some(expr) = block.expr.take() {
                (
                    this.mark_span_with_reason(
                        DesugaringKind::TryBlock,
                        expr.span,
                        Some(Arc::clone(&this.allow_try_trait)),
                    ),
                    expr,
                )
            } else {
                let try_span = this.mark_span_with_reason(
                    DesugaringKind::TryBlock,
                    this.tcx.sess.source_map().end_point(body.span),
                    Some(Arc::clone(&this.allow_try_trait)),
                );

                (try_span, this.expr_unit(try_span))
            };

            let ok_wrapped_span =
                this.mark_span_with_reason(DesugaringKind::TryBlock, tail_expr.span, None);

            // `::std::ops::Try::from_output($tail_expr)`
            block.expr = Some(this.wrap_in_try_constructor(
                hir::LangItem::TryTraitFromOutput,
                try_span,
                tail_expr,
                ok_wrapped_span,
            ));

            hir::ExprKind::Block(this.arena.alloc(block), None)
        })
    }

    fn wrap_in_try_constructor(
        &mut self,
        lang_item: hir::LangItem,
        method_span: Span,
        expr: &'hir hir::Expr<'hir>,
        overall_span: Span,
    ) -> &'hir hir::Expr<'hir> {
        let constructor = self.arena.alloc(self.expr_lang_item_path(method_span, lang_item));
        self.expr_call(overall_span, constructor, std::slice::from_ref(expr))
    }

    fn lower_arm(&mut self, arm: &Arm) -> hir::Arm<'hir> {
        let pat = self.lower_pat(&arm.pat);
        let guard = arm.guard.as_ref().map(|cond| self.lower_expr(cond));
        let hir_id = self.next_id();
        let span = self.lower_span(arm.span);
        self.lower_attrs(hir_id, &arm.attrs, arm.span);
        let is_never_pattern = pat.is_never_pattern();
        // We need to lower the body even if it's unneeded for never pattern in match,
        // ensure that we can get HirId for DefId if need (issue #137708).
        let body = arm.body.as_ref().map(|x| self.lower_expr(x));
        let body = if let Some(body) = body
            && !is_never_pattern
        {
            body
        } else {
            // Either `body.is_none()` or `is_never_pattern` here.
            if !is_never_pattern {
                if self.tcx.features().never_patterns() {
                    // If the feature is off we already emitted the error after parsing.
                    let suggestion = span.shrink_to_hi();
                    self.dcx().emit_err(MatchArmWithNoBody { span, suggestion });
                }
            } else if let Some(body) = &arm.body {
                self.dcx().emit_err(NeverPatternWithBody { span: body.span });
            } else if let Some(g) = &arm.guard {
                self.dcx().emit_err(NeverPatternWithGuard { span: g.span });
            }

            // We add a fake `loop {}` arm body so that it typecks to `!`. The mir lowering of never
            // patterns ensures this loop is not reachable.
            let block = self.arena.alloc(hir::Block {
                stmts: &[],
                expr: None,
                hir_id: self.next_id(),
                rules: hir::BlockCheckMode::DefaultBlock,
                span,
                targeted_by_break: false,
            });
            self.arena.alloc(hir::Expr {
                hir_id: self.next_id(),
                kind: hir::ExprKind::Loop(block, None, hir::LoopSource::Loop, span),
                span,
            })
        };
        hir::Arm { hir_id, pat, guard, body, span }
    }

    /// Lower/desugar a coroutine construct.
    ///
    /// In particular, this creates the correct async resume argument and `_task_context`.
    ///
    /// This results in:
    ///
    /// ```text
    /// static move? |<_task_context?>| -> <return_ty> {
    ///     <body>
    /// }
    /// ```
    pub(super) fn make_desugared_coroutine_expr(
        &mut self,
        capture_clause: CaptureBy,
        closure_node_id: NodeId,
        return_ty: Option<hir::FnRetTy<'hir>>,
        fn_decl_span: Span,
        span: Span,
        desugaring_kind: hir::CoroutineDesugaring,
        coroutine_source: hir::CoroutineSource,
        body: impl FnOnce(&mut Self) -> hir::Expr<'hir>,
    ) -> hir::ExprKind<'hir> {
        let closure_def_id = self.local_def_id(closure_node_id);
        let coroutine_kind = hir::CoroutineKind::Desugared(desugaring_kind, coroutine_source);

        // The `async` desugaring takes a resume argument and maintains a `task_context`,
        // whereas a generator does not.
        let (inputs, params, task_context): (&[_], &[_], _) = match desugaring_kind {
            hir::CoroutineDesugaring::Async | hir::CoroutineDesugaring::AsyncGen => {
                // Resume argument type: `ResumeTy`
                let unstable_span = self.mark_span_with_reason(
                    DesugaringKind::Async,
                    self.lower_span(span),
                    Some(Arc::clone(&self.allow_gen_future)),
                );
                let resume_ty =
                    self.make_lang_item_qpath(hir::LangItem::ResumeTy, unstable_span, None);
                let input_ty = hir::Ty {
                    hir_id: self.next_id(),
                    kind: hir::TyKind::Path(resume_ty),
                    span: unstable_span,
                };
                let inputs = arena_vec![self; input_ty];

                // Lower the argument pattern/ident. The ident is used again in the `.await` lowering.
                let (pat, task_context_hid) = self.pat_ident_binding_mode(
                    span,
                    Ident::with_dummy_span(sym::_task_context),
                    hir::BindingMode::MUT,
                );
                let param = hir::Param {
                    hir_id: self.next_id(),
                    pat,
                    ty_span: self.lower_span(span),
                    span: self.lower_span(span),
                };
                let params = arena_vec![self; param];

                (inputs, params, Some(task_context_hid))
            }
            hir::CoroutineDesugaring::Gen => (&[], &[], None),
        };

        let output =
            return_ty.unwrap_or_else(|| hir::FnRetTy::DefaultReturn(self.lower_span(span)));

        let fn_decl = self.arena.alloc(hir::FnDecl {
            inputs,
            output,
            c_variadic: false,
            implicit_self: hir::ImplicitSelfKind::None,
            lifetime_elision_allowed: false,
        });

        let body = self.lower_body(move |this| {
            this.coroutine_kind = Some(coroutine_kind);

            let old_ctx = this.task_context;
            if task_context.is_some() {
                this.task_context = task_context;
            }
            let res = body(this);
            this.task_context = old_ctx;

            (params, res)
        });

        // `static |<_task_context?>| -> <return_ty> { <body> }`:
        hir::ExprKind::Closure(self.arena.alloc(hir::Closure {
            def_id: closure_def_id,
            binder: hir::ClosureBinder::Default,
            capture_clause,
            bound_generic_params: &[],
            fn_decl,
            body,
            fn_decl_span: self.lower_span(fn_decl_span),
            fn_arg_span: None,
            kind: hir::ClosureKind::Coroutine(coroutine_kind),
            constness: hir::Constness::NotConst,
        }))
    }

    /// Forwards a possible `#[track_caller]` annotation from `outer_hir_id` to
    /// `inner_hir_id` in case the `async_fn_track_caller` feature is enabled.
    pub(super) fn maybe_forward_track_caller(
        &mut self,
        span: Span,
        outer_hir_id: HirId,
        inner_hir_id: HirId,
    ) {
        if self.tcx.features().async_fn_track_caller()
            && let Some(attrs) = self.attrs.get(&outer_hir_id.local_id)
            && attrs.into_iter().any(|attr| attr.has_name(sym::track_caller))
        {
            let unstable_span = self.mark_span_with_reason(
                DesugaringKind::Async,
                span,
                Some(Arc::clone(&self.allow_gen_future)),
            );
            self.lower_attrs(
                inner_hir_id,
                &[Attribute {
                    kind: AttrKind::Normal(ptr::P(NormalAttr::from_ident(Ident::new(
                        sym::track_caller,
                        span,
                    )))),
                    id: self.tcx.sess.psess.attr_id_generator.mk_attr_id(),
                    style: AttrStyle::Outer,
                    span: unstable_span,
                }],
                span,
            );
        }
    }

    /// Desugar `<expr>.await` into:
    /// ```ignore (pseudo-rust)
    /// match ::std::future::IntoFuture::into_future(<expr>) {
    ///     mut __awaitee => loop {
    ///         match unsafe { ::std::future::Future::poll(
    ///             <::std::pin::Pin>::new_unchecked(&mut __awaitee),
    ///             ::std::future::get_context(task_context),
    ///         ) } {
    ///             ::std::task::Poll::Ready(result) => break result,
    ///             ::std::task::Poll::Pending => {}
    ///         }
    ///         task_context = yield ();
    ///     }
    /// }
    /// ```
    fn lower_expr_await(&mut self, await_kw_span: Span, expr: &Expr) -> hir::ExprKind<'hir> {
        let expr = self.arena.alloc(self.lower_expr_mut(expr));
        self.make_lowered_await(await_kw_span, expr, FutureKind::Future)
    }

    /// Takes an expr that has already been lowered and generates a desugared await loop around it
    fn make_lowered_await(
        &mut self,
        await_kw_span: Span,
        expr: &'hir hir::Expr<'hir>,
        await_kind: FutureKind,
    ) -> hir::ExprKind<'hir> {
        let full_span = expr.span.to(await_kw_span);

        let is_async_gen = match self.coroutine_kind {
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => false,
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _)) => true,
            Some(hir::CoroutineKind::Coroutine(_))
            | Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _))
            | None => {
                // Lower to a block `{ EXPR; <error> }` so that the awaited expr
                // is not accidentally orphaned.
                let stmt_id = self.next_id();
                let expr_err = self.expr(
                    expr.span,
                    hir::ExprKind::Err(self.dcx().emit_err(AwaitOnlyInAsyncFnAndBlocks {
                        await_kw_span,
                        item_span: self.current_item,
                    })),
                );
                return hir::ExprKind::Block(
                    self.block_all(
                        expr.span,
                        arena_vec![self; hir::Stmt {
                            hir_id: stmt_id,
                            kind: hir::StmtKind::Semi(expr),
                            span: expr.span,
                        }],
                        Some(self.arena.alloc(expr_err)),
                    ),
                    None,
                );
            }
        };

        let features = match await_kind {
            FutureKind::Future => None,
            FutureKind::AsyncIterator => Some(Arc::clone(&self.allow_for_await)),
        };
        let span = self.mark_span_with_reason(DesugaringKind::Await, await_kw_span, features);
        let gen_future_span = self.mark_span_with_reason(
            DesugaringKind::Await,
            full_span,
            Some(Arc::clone(&self.allow_gen_future)),
        );
        let expr_hir_id = expr.hir_id;

        // Note that the name of this binding must not be changed to something else because
        // debuggers and debugger extensions expect it to be called `__awaitee`. They use
        // this name to identify what is being awaited by a suspended async functions.
        let awaitee_ident = Ident::with_dummy_span(sym::__awaitee);
        let (awaitee_pat, awaitee_pat_hid) =
            self.pat_ident_binding_mode(gen_future_span, awaitee_ident, hir::BindingMode::MUT);

        let task_context_ident = Ident::with_dummy_span(sym::_task_context);

        // unsafe {
        //     ::std::future::Future::poll(
        //         ::std::pin::Pin::new_unchecked(&mut __awaitee),
        //         ::std::future::get_context(task_context),
        //     )
        // }
        let poll_expr = {
            let awaitee = self.expr_ident(span, awaitee_ident, awaitee_pat_hid);
            let ref_mut_awaitee = self.expr_mut_addr_of(span, awaitee);

            let Some(task_context_hid) = self.task_context else {
                unreachable!("use of `await` outside of an async context.");
            };

            let task_context = self.expr_ident_mut(span, task_context_ident, task_context_hid);

            let new_unchecked = self.expr_call_lang_item_fn_mut(
                span,
                hir::LangItem::PinNewUnchecked,
                arena_vec![self; ref_mut_awaitee],
            );
            let get_context = self.expr_call_lang_item_fn_mut(
                gen_future_span,
                hir::LangItem::GetContext,
                arena_vec![self; task_context],
            );
            let call = match await_kind {
                FutureKind::Future => self.expr_call_lang_item_fn(
                    span,
                    hir::LangItem::FuturePoll,
                    arena_vec![self; new_unchecked, get_context],
                ),
                FutureKind::AsyncIterator => self.expr_call_lang_item_fn(
                    span,
                    hir::LangItem::AsyncIteratorPollNext,
                    arena_vec![self; new_unchecked, get_context],
                ),
            };
            self.arena.alloc(self.expr_unsafe(call))
        };

        // `::std::task::Poll::Ready(result) => break result`
        let loop_node_id = self.next_node_id();
        let loop_hir_id = self.lower_node_id(loop_node_id);
        let ready_arm = {
            let x_ident = Ident::with_dummy_span(sym::result);
            let (x_pat, x_pat_hid) = self.pat_ident(gen_future_span, x_ident);
            let x_expr = self.expr_ident(gen_future_span, x_ident, x_pat_hid);
            let ready_field = self.single_pat_field(gen_future_span, x_pat);
            let ready_pat = self.pat_lang_item_variant(span, hir::LangItem::PollReady, ready_field);
            let break_x = self.with_loop_scope(loop_hir_id, move |this| {
                let expr_break =
                    hir::ExprKind::Break(this.lower_loop_destination(None), Some(x_expr));
                this.arena.alloc(this.expr(gen_future_span, expr_break))
            });
            self.arm(ready_pat, break_x)
        };

        // `::std::task::Poll::Pending => {}`
        let pending_arm = {
            let pending_pat = self.pat_lang_item_variant(span, hir::LangItem::PollPending, &[]);
            let empty_block = self.expr_block_empty(span);
            self.arm(pending_pat, empty_block)
        };

        let inner_match_stmt = {
            let match_expr = self.expr_match(
                span,
                poll_expr,
                arena_vec![self; ready_arm, pending_arm],
                hir::MatchSource::AwaitDesugar,
            );
            self.stmt_expr(span, match_expr)
        };

        // Depending on `async` of `async gen`:
        // async     - task_context = yield ();
        // async gen - task_context = yield ASYNC_GEN_PENDING;
        let yield_stmt = {
            let yielded = if is_async_gen {
                self.arena.alloc(self.expr_lang_item_path(span, hir::LangItem::AsyncGenPending))
            } else {
                self.expr_unit(span)
            };

            let yield_expr = self.expr(
                span,
                hir::ExprKind::Yield(yielded, hir::YieldSource::Await { expr: Some(expr_hir_id) }),
            );
            let yield_expr = self.arena.alloc(yield_expr);

            let Some(task_context_hid) = self.task_context else {
                unreachable!("use of `await` outside of an async context.");
            };

            let lhs = self.expr_ident(span, task_context_ident, task_context_hid);
            let assign =
                self.expr(span, hir::ExprKind::Assign(lhs, yield_expr, self.lower_span(span)));
            self.stmt_expr(span, assign)
        };

        let loop_block = self.block_all(span, arena_vec![self; inner_match_stmt, yield_stmt], None);

        // loop { .. }
        let loop_expr = self.arena.alloc(hir::Expr {
            hir_id: loop_hir_id,
            kind: hir::ExprKind::Loop(
                loop_block,
                None,
                hir::LoopSource::Loop,
                self.lower_span(span),
            ),
            span: self.lower_span(span),
        });

        // mut __awaitee => loop { ... }
        let awaitee_arm = self.arm(awaitee_pat, loop_expr);

        // `match ::std::future::IntoFuture::into_future(<expr>) { ... }`
        let into_future_expr = match await_kind {
            FutureKind::Future => self.expr_call_lang_item_fn(
                span,
                hir::LangItem::IntoFutureIntoFuture,
                arena_vec![self; *expr],
            ),
            // Not needed for `for await` because we expect to have already called
            // `IntoAsyncIterator::into_async_iter` on it.
            FutureKind::AsyncIterator => expr,
        };

        // match <into_future_expr> {
        //     mut __awaitee => loop { .. }
        // }
        hir::ExprKind::Match(
            into_future_expr,
            arena_vec![self; awaitee_arm],
            hir::MatchSource::AwaitDesugar,
        )
    }

    fn lower_expr_use(&mut self, use_kw_span: Span, expr: &Expr) -> hir::ExprKind<'hir> {
        hir::ExprKind::Use(self.lower_expr(expr), use_kw_span)
    }

    fn lower_expr_closure(
        &mut self,
        binder: &ClosureBinder,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        closure_hir_id: hir::HirId,
        constness: Const,
        movability: Movability,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
        fn_arg_span: Span,
    ) -> hir::ExprKind<'hir> {
        let closure_def_id = self.local_def_id(closure_id);
        let (binder_clause, generic_params) = self.lower_closure_binder(binder);

        let (body_id, closure_kind) = self.with_new_scopes(fn_decl_span, move |this| {
            let mut coroutine_kind = if this
                .attrs
                .get(&closure_hir_id.local_id)
                .is_some_and(|attrs| attrs.iter().any(|attr| attr.has_name(sym::coroutine)))
            {
                Some(hir::CoroutineKind::Coroutine(Movability::Movable))
            } else {
                None
            };
            // FIXME(contracts): Support contracts on closures?
            let body_id = this.lower_fn_body(decl, None, |this| {
                this.coroutine_kind = coroutine_kind;
                let e = this.lower_expr_mut(body);
                coroutine_kind = this.coroutine_kind;
                e
            });
            let coroutine_option =
                this.closure_movability_for_fn(decl, fn_decl_span, coroutine_kind, movability);
            (body_id, coroutine_option)
        });

        let bound_generic_params = self.lower_lifetime_binder(closure_id, generic_params);
        // Lower outside new scope to preserve `is_in_loop_condition`.
        let fn_decl = self.lower_fn_decl(decl, closure_id, fn_decl_span, FnDeclKind::Closure, None);

        let c = self.arena.alloc(hir::Closure {
            def_id: closure_def_id,
            binder: binder_clause,
            capture_clause,
            bound_generic_params,
            fn_decl,
            body: body_id,
            fn_decl_span: self.lower_span(fn_decl_span),
            fn_arg_span: Some(self.lower_span(fn_arg_span)),
            kind: closure_kind,
            constness: self.lower_constness(constness),
        });

        hir::ExprKind::Closure(c)
    }

    fn closure_movability_for_fn(
        &mut self,
        decl: &FnDecl,
        fn_decl_span: Span,
        coroutine_kind: Option<hir::CoroutineKind>,
        movability: Movability,
    ) -> hir::ClosureKind {
        match coroutine_kind {
            Some(hir::CoroutineKind::Coroutine(_)) => {
                if decl.inputs.len() > 1 {
                    self.dcx().emit_err(CoroutineTooManyParameters { fn_decl_span });
                }
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(movability))
            }
            Some(
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _),
            ) => {
                panic!("non-`async`/`gen` closure body turned `async`/`gen` during lowering");
            }
            None => {
                if movability == Movability::Static {
                    self.dcx().emit_err(ClosureCannotBeStatic { fn_decl_span });
                }
                hir::ClosureKind::Closure
            }
        }
    }

    fn lower_closure_binder<'c>(
        &mut self,
        binder: &'c ClosureBinder,
    ) -> (hir::ClosureBinder, &'c [GenericParam]) {
        let (binder, params) = match binder {
            ClosureBinder::NotPresent => (hir::ClosureBinder::Default, &[][..]),
            ClosureBinder::For { span, generic_params } => {
                let span = self.lower_span(*span);
                (hir::ClosureBinder::For { span }, &**generic_params)
            }
        };

        (binder, params)
    }

    fn lower_expr_coroutine_closure(
        &mut self,
        binder: &ClosureBinder,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        closure_hir_id: HirId,
        coroutine_kind: CoroutineKind,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
        fn_arg_span: Span,
    ) -> hir::ExprKind<'hir> {
        let closure_def_id = self.local_def_id(closure_id);
        let (binder_clause, generic_params) = self.lower_closure_binder(binder);

        assert_matches!(
            coroutine_kind,
            CoroutineKind::Async { .. },
            "only async closures are supported currently"
        );

        let body = self.with_new_scopes(fn_decl_span, |this| {
            let inner_decl =
                FnDecl { inputs: decl.inputs.clone(), output: FnRetTy::Default(fn_decl_span) };

            // Transform `async |x: u8| -> X { ... }` into
            // `|x: u8| || -> X { ... }`.
            let body_id = this.lower_body(|this| {
                let (parameters, expr) = this.lower_coroutine_body_with_moved_arguments(
                    &inner_decl,
                    |this| this.with_new_scopes(fn_decl_span, |this| this.lower_expr_mut(body)),
                    fn_decl_span,
                    body.span,
                    coroutine_kind,
                    hir::CoroutineSource::Closure,
                );

                this.maybe_forward_track_caller(body.span, closure_hir_id, expr.hir_id);

                (parameters, expr)
            });
            body_id
        });

        let bound_generic_params = self.lower_lifetime_binder(closure_id, generic_params);
        // We need to lower the declaration outside the new scope, because we
        // have to conserve the state of being inside a loop condition for the
        // closure argument types.
        let fn_decl =
            self.lower_fn_decl(&decl, closure_id, fn_decl_span, FnDeclKind::Closure, None);

        let c = self.arena.alloc(hir::Closure {
            def_id: closure_def_id,
            binder: binder_clause,
            capture_clause,
            bound_generic_params,
            fn_decl,
            body,
            fn_decl_span: self.lower_span(fn_decl_span),
            fn_arg_span: Some(self.lower_span(fn_arg_span)),
            // Lower this as a `CoroutineClosure`. That will ensure that HIR typeck
            // knows that a `FnDecl` output type like `-> &str` actually means
            // "coroutine that returns &str", rather than directly returning a `&str`.
            kind: hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async),
            constness: hir::Constness::NotConst,
        });
        hir::ExprKind::Closure(c)
    }

    /// Destructure the LHS of complex assignments.
    /// For instance, lower `(a, b) = t` to `{ let (lhs1, lhs2) = t; a = lhs1; b = lhs2; }`.
    fn lower_expr_assign(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        eq_sign_span: Span,
        whole_span: Span,
    ) -> hir::ExprKind<'hir> {
        // Return early in case of an ordinary assignment.
        fn is_ordinary(lower_ctx: &mut LoweringContext<'_, '_>, lhs: &Expr) -> bool {
            match &lhs.kind {
                ExprKind::Array(..)
                | ExprKind::Struct(..)
                | ExprKind::Tup(..)
                | ExprKind::Underscore => false,
                // Check for unit struct constructor.
                ExprKind::Path(..) => lower_ctx.extract_unit_struct_path(lhs).is_none(),
                // Check for tuple struct constructor.
                ExprKind::Call(callee, ..) => lower_ctx.extract_tuple_struct_path(callee).is_none(),
                ExprKind::Paren(e) => {
                    match e.kind {
                        // We special-case `(..)` for consistency with patterns.
                        ExprKind::Range(None, None, RangeLimits::HalfOpen) => false,
                        _ => is_ordinary(lower_ctx, e),
                    }
                }
                _ => true,
            }
        }
        if is_ordinary(self, lhs) {
            return hir::ExprKind::Assign(
                self.lower_expr(lhs),
                self.lower_expr(rhs),
                self.lower_span(eq_sign_span),
            );
        }

        let mut assignments = vec![];

        // The LHS becomes a pattern: `(lhs1, lhs2)`.
        let pat = self.destructure_assign(lhs, eq_sign_span, &mut assignments);
        let rhs = self.lower_expr(rhs);

        // Introduce a `let` for destructuring: `let (lhs1, lhs2) = t`.
        let destructure_let = self.stmt_let_pat(
            None,
            whole_span,
            Some(rhs),
            pat,
            hir::LocalSource::AssignDesugar(self.lower_span(eq_sign_span)),
        );

        // `a = lhs1; b = lhs2;`.
        let stmts = self.arena.alloc_from_iter(std::iter::once(destructure_let).chain(assignments));

        // Wrap everything in a block.
        hir::ExprKind::Block(self.block_all(whole_span, stmts, None), None)
    }

    /// If the given expression is a path to a tuple struct, returns that path.
    /// It is not a complete check, but just tries to reject most paths early
    /// if they are not tuple structs.
    /// Type checking will take care of the full validation later.
    fn extract_tuple_struct_path<'a>(
        &mut self,
        expr: &'a Expr,
    ) -> Option<(&'a Option<AstP<QSelf>>, &'a Path)> {
        if let ExprKind::Path(qself, path) = &expr.kind {
            // Does the path resolve to something disallowed in a tuple struct/variant pattern?
            if let Some(partial_res) = self.resolver.get_partial_res(expr.id) {
                if let Some(res) = partial_res.full_res()
                    && !res.expected_in_tuple_struct_pat()
                {
                    return None;
                }
            }
            return Some((qself, path));
        }
        None
    }

    /// If the given expression is a path to a unit struct, returns that path.
    /// It is not a complete check, but just tries to reject most paths early
    /// if they are not unit structs.
    /// Type checking will take care of the full validation later.
    fn extract_unit_struct_path<'a>(
        &mut self,
        expr: &'a Expr,
    ) -> Option<(&'a Option<AstP<QSelf>>, &'a Path)> {
        if let ExprKind::Path(qself, path) = &expr.kind {
            // Does the path resolve to something disallowed in a unit struct/variant pattern?
            if let Some(partial_res) = self.resolver.get_partial_res(expr.id) {
                if let Some(res) = partial_res.full_res()
                    && !res.expected_in_unit_struct_pat()
                {
                    return None;
                }
            }
            return Some((qself, path));
        }
        None
    }

    /// Convert the LHS of a destructuring assignment to a pattern.
    /// Each sub-assignment is recorded in `assignments`.
    fn destructure_assign(
        &mut self,
        lhs: &Expr,
        eq_sign_span: Span,
        assignments: &mut Vec<hir::Stmt<'hir>>,
    ) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(self.destructure_assign_mut(lhs, eq_sign_span, assignments))
    }

    fn destructure_assign_mut(
        &mut self,
        lhs: &Expr,
        eq_sign_span: Span,
        assignments: &mut Vec<hir::Stmt<'hir>>,
    ) -> hir::Pat<'hir> {
        match &lhs.kind {
            // Underscore pattern.
            ExprKind::Underscore => {
                return self.pat_without_dbm(lhs.span, hir::PatKind::Wild);
            }
            // Slice patterns.
            ExprKind::Array(elements) => {
                let (pats, rest) =
                    self.destructure_sequence(elements, "slice", eq_sign_span, assignments);
                let slice_pat = if let Some((i, span)) = rest {
                    let (before, after) = pats.split_at(i);
                    hir::PatKind::Slice(
                        before,
                        Some(self.arena.alloc(self.pat_without_dbm(span, hir::PatKind::Wild))),
                        after,
                    )
                } else {
                    hir::PatKind::Slice(pats, None, &[])
                };
                return self.pat_without_dbm(lhs.span, slice_pat);
            }
            // Tuple structs.
            ExprKind::Call(callee, args) => {
                if let Some((qself, path)) = self.extract_tuple_struct_path(callee) {
                    let (pats, rest) = self.destructure_sequence(
                        args,
                        "tuple struct or variant",
                        eq_sign_span,
                        assignments,
                    );
                    let qpath = self.lower_qpath(
                        callee.id,
                        qself,
                        path,
                        ParamMode::Optional,
                        AllowReturnTypeNotation::No,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                        None,
                    );
                    // Destructure like a tuple struct.
                    let tuple_struct_pat = hir::PatKind::TupleStruct(
                        qpath,
                        pats,
                        hir::DotDotPos::new(rest.map(|r| r.0)),
                    );
                    return self.pat_without_dbm(lhs.span, tuple_struct_pat);
                }
            }
            // Unit structs and enum variants.
            ExprKind::Path(..) => {
                if let Some((qself, path)) = self.extract_unit_struct_path(lhs) {
                    let qpath = self.lower_qpath(
                        lhs.id,
                        qself,
                        path,
                        ParamMode::Optional,
                        AllowReturnTypeNotation::No,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                        None,
                    );
                    // Destructure like a unit struct.
                    let unit_struct_pat = hir::PatKind::Expr(self.arena.alloc(hir::PatExpr {
                        kind: hir::PatExprKind::Path(qpath),
                        hir_id: self.next_id(),
                        span: self.lower_span(lhs.span),
                    }));
                    return self.pat_without_dbm(lhs.span, unit_struct_pat);
                }
            }
            // Structs.
            ExprKind::Struct(se) => {
                let field_pats = self.arena.alloc_from_iter(se.fields.iter().map(|f| {
                    let pat = self.destructure_assign(&f.expr, eq_sign_span, assignments);
                    hir::PatField {
                        hir_id: self.next_id(),
                        ident: self.lower_ident(f.ident),
                        pat,
                        is_shorthand: f.is_shorthand,
                        span: self.lower_span(f.span),
                    }
                }));
                let qpath = self.lower_qpath(
                    lhs.id,
                    &se.qself,
                    &se.path,
                    ParamMode::Optional,
                    AllowReturnTypeNotation::No,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                    None,
                );
                let fields_omitted = match &se.rest {
                    StructRest::Base(e) => {
                        self.dcx().emit_err(FunctionalRecordUpdateDestructuringAssignment {
                            span: e.span,
                        });
                        true
                    }
                    StructRest::Rest(_) => true,
                    StructRest::None => false,
                };
                let struct_pat = hir::PatKind::Struct(qpath, field_pats, fields_omitted);
                return self.pat_without_dbm(lhs.span, struct_pat);
            }
            // Tuples.
            ExprKind::Tup(elements) => {
                let (pats, rest) =
                    self.destructure_sequence(elements, "tuple", eq_sign_span, assignments);
                let tuple_pat = hir::PatKind::Tuple(pats, hir::DotDotPos::new(rest.map(|r| r.0)));
                return self.pat_without_dbm(lhs.span, tuple_pat);
            }
            ExprKind::Paren(e) => {
                // We special-case `(..)` for consistency with patterns.
                if let ExprKind::Range(None, None, RangeLimits::HalfOpen) = e.kind {
                    let tuple_pat = hir::PatKind::Tuple(&[], hir::DotDotPos::new(Some(0)));
                    return self.pat_without_dbm(lhs.span, tuple_pat);
                } else {
                    return self.destructure_assign_mut(e, eq_sign_span, assignments);
                }
            }
            _ => {}
        }
        // Treat all other cases as normal lvalue.
        let ident = Ident::new(sym::lhs, self.lower_span(lhs.span));
        let (pat, binding) = self.pat_ident_mut(lhs.span, ident);
        let ident = self.expr_ident(lhs.span, ident, binding);
        let assign =
            hir::ExprKind::Assign(self.lower_expr(lhs), ident, self.lower_span(eq_sign_span));
        let expr = self.expr(lhs.span, assign);
        assignments.push(self.stmt_expr(lhs.span, expr));
        pat
    }

    /// Destructure a sequence of expressions occurring on the LHS of an assignment.
    /// Such a sequence occurs in a tuple (struct)/slice.
    /// Return a sequence of corresponding patterns, and the index and the span of `..` if it
    /// exists.
    /// Each sub-assignment is recorded in `assignments`.
    fn destructure_sequence(
        &mut self,
        elements: &[AstP<Expr>],
        ctx: &str,
        eq_sign_span: Span,
        assignments: &mut Vec<hir::Stmt<'hir>>,
    ) -> (&'hir [hir::Pat<'hir>], Option<(usize, Span)>) {
        let mut rest = None;
        let elements =
            self.arena.alloc_from_iter(elements.iter().enumerate().filter_map(|(i, e)| {
                // Check for `..` pattern.
                if let ExprKind::Range(None, None, RangeLimits::HalfOpen) = e.kind {
                    if let Some((_, prev_span)) = rest {
                        self.ban_extra_rest_pat(e.span, prev_span, ctx);
                    } else {
                        rest = Some((i, e.span));
                    }
                    None
                } else {
                    Some(self.destructure_assign_mut(e, eq_sign_span, assignments))
                }
            }));
        (elements, rest)
    }

    /// Desugar `<start>..=<end>` into `std::ops::RangeInclusive::new(<start>, <end>)`.
    fn lower_expr_range_closed(&mut self, span: Span, e1: &Expr, e2: &Expr) -> hir::ExprKind<'hir> {
        let e1 = self.lower_expr_mut(e1);
        let e2 = self.lower_expr_mut(e2);
        let fn_path = hir::QPath::LangItem(hir::LangItem::RangeInclusiveNew, self.lower_span(span));
        let fn_expr = self.arena.alloc(self.expr(span, hir::ExprKind::Path(fn_path)));
        hir::ExprKind::Call(fn_expr, arena_vec![self; e1, e2])
    }

    fn lower_expr_range(
        &mut self,
        span: Span,
        e1: Option<&Expr>,
        e2: Option<&Expr>,
        lims: RangeLimits,
    ) -> hir::ExprKind<'hir> {
        use rustc_ast::RangeLimits::*;

        let lang_item = match (e1, e2, lims) {
            (None, None, HalfOpen) => hir::LangItem::RangeFull,
            (Some(..), None, HalfOpen) => {
                if self.tcx.features().new_range() {
                    hir::LangItem::RangeFromCopy
                } else {
                    hir::LangItem::RangeFrom
                }
            }
            (None, Some(..), HalfOpen) => hir::LangItem::RangeTo,
            (Some(..), Some(..), HalfOpen) => {
                if self.tcx.features().new_range() {
                    hir::LangItem::RangeCopy
                } else {
                    hir::LangItem::Range
                }
            }
            (None, Some(..), Closed) => hir::LangItem::RangeToInclusive,
            (Some(e1), Some(e2), Closed) => {
                if self.tcx.features().new_range() {
                    hir::LangItem::RangeInclusiveCopy
                } else {
                    return self.lower_expr_range_closed(span, e1, e2);
                }
            }
            (start, None, Closed) => {
                self.dcx().emit_err(InclusiveRangeWithNoEnd { span });
                match start {
                    Some(..) => {
                        if self.tcx.features().new_range() {
                            hir::LangItem::RangeFromCopy
                        } else {
                            hir::LangItem::RangeFrom
                        }
                    }
                    None => hir::LangItem::RangeFull,
                }
            }
        };

        let fields = self.arena.alloc_from_iter(
            e1.iter().map(|e| (sym::start, e)).chain(e2.iter().map(|e| (sym::end, e))).map(
                |(s, e)| {
                    let expr = self.lower_expr(e);
                    let ident = Ident::new(s, self.lower_span(e.span));
                    self.expr_field(ident, expr, e.span)
                },
            ),
        );

        hir::ExprKind::Struct(
            self.arena.alloc(hir::QPath::LangItem(lang_item, self.lower_span(span))),
            fields,
            hir::StructTailExpr::None,
        )
    }

    // Record labelled expr's HirId so that we can retrieve it in `lower_jump_destination` without
    // lowering node id again.
    fn lower_label(
        &mut self,
        opt_label: Option<Label>,
        dest_id: NodeId,
        dest_hir_id: hir::HirId,
    ) -> Option<Label> {
        let label = opt_label?;
        self.ident_and_label_to_local_id.insert(dest_id, dest_hir_id.local_id);
        Some(Label { ident: self.lower_ident(label.ident) })
    }

    fn lower_loop_destination(&mut self, destination: Option<(NodeId, Label)>) -> hir::Destination {
        let target_id = match destination {
            Some((id, _)) => {
                if let Some(loop_id) = self.resolver.get_label_res(id) {
                    let local_id = self.ident_and_label_to_local_id[&loop_id];
                    let loop_hir_id = HirId { owner: self.current_hir_id_owner, local_id };
                    Ok(loop_hir_id)
                } else {
                    Err(hir::LoopIdError::UnresolvedLabel)
                }
            }
            None => {
                self.loop_scope.map(|id| Ok(id)).unwrap_or(Err(hir::LoopIdError::OutsideLoopScope))
            }
        };
        let label = destination
            .map(|(_, label)| label)
            .map(|label| Label { ident: self.lower_ident(label.ident) });
        hir::Destination { label, target_id }
    }

    fn lower_jump_destination(&mut self, id: NodeId, opt_label: Option<Label>) -> hir::Destination {
        if self.is_in_loop_condition && opt_label.is_none() {
            hir::Destination {
                label: None,
                target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition),
            }
        } else {
            self.lower_loop_destination(opt_label.map(|label| (id, label)))
        }
    }

    fn with_catch_scope<T>(&mut self, catch_id: hir::HirId, f: impl FnOnce(&mut Self) -> T) -> T {
        let old_scope = self.catch_scope.replace(catch_id);
        let result = f(self);
        self.catch_scope = old_scope;
        result
    }

    fn with_loop_scope<T>(&mut self, loop_id: hir::HirId, f: impl FnOnce(&mut Self) -> T) -> T {
        // We're no longer in the base loop's condition; we're in another loop.
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let old_scope = self.loop_scope.replace(loop_id);
        let result = f(self);
        self.loop_scope = old_scope;

        self.is_in_loop_condition = was_in_loop_condition;

        result
    }

    fn with_loop_condition_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = true;

        let result = f(self);

        self.is_in_loop_condition = was_in_loop_condition;

        result
    }

    fn lower_expr_field(&mut self, f: &ExprField) -> hir::ExprField<'hir> {
        let hir_id = self.lower_node_id(f.id);
        self.lower_attrs(hir_id, &f.attrs, f.span);
        hir::ExprField {
            hir_id,
            ident: self.lower_ident(f.ident),
            expr: self.lower_expr(&f.expr),
            span: self.lower_span(f.span),
            is_shorthand: f.is_shorthand,
        }
    }

    fn lower_expr_yield(&mut self, span: Span, opt_expr: Option<&Expr>) -> hir::ExprKind<'hir> {
        let yielded =
            opt_expr.as_ref().map(|x| self.lower_expr(x)).unwrap_or_else(|| self.expr_unit(span));

        if !self.tcx.features().yield_expr()
            && !self.tcx.features().coroutines()
            && !self.tcx.features().gen_blocks()
        {
            rustc_session::parse::feature_err(
                &self.tcx.sess,
                sym::yield_expr,
                span,
                fluent_generated::ast_lowering_yield,
            )
            .emit();
        }

        let is_async_gen = match self.coroutine_kind {
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)) => false,
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _)) => true,
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => {
                // Lower to a block `{ EXPR; <error> }` so that the awaited expr
                // is not accidentally orphaned.
                let stmt_id = self.next_id();
                let expr_err = self.expr(
                    yielded.span,
                    hir::ExprKind::Err(self.dcx().emit_err(AsyncCoroutinesNotSupported { span })),
                );
                return hir::ExprKind::Block(
                    self.block_all(
                        yielded.span,
                        arena_vec![self; hir::Stmt {
                            hir_id: stmt_id,
                            kind: hir::StmtKind::Semi(yielded),
                            span: yielded.span,
                        }],
                        Some(self.arena.alloc(expr_err)),
                    ),
                    None,
                );
            }
            Some(hir::CoroutineKind::Coroutine(_)) => false,
            None => {
                let suggestion = self.current_item.map(|s| s.shrink_to_lo());
                self.dcx().emit_err(YieldInClosure { span, suggestion });
                self.coroutine_kind = Some(hir::CoroutineKind::Coroutine(Movability::Movable));

                false
            }
        };

        if is_async_gen {
            // `yield $expr` is transformed into `task_context = yield async_gen_ready($expr)`.
            // This ensures that we store our resumed `ResumeContext` correctly, and also that
            // the apparent value of the `yield` expression is `()`.
            let wrapped_yielded = self.expr_call_lang_item_fn(
                span,
                hir::LangItem::AsyncGenReady,
                std::slice::from_ref(yielded),
            );
            let yield_expr = self.arena.alloc(
                self.expr(span, hir::ExprKind::Yield(wrapped_yielded, hir::YieldSource::Yield)),
            );

            let Some(task_context_hid) = self.task_context else {
                unreachable!("use of `await` outside of an async context.");
            };
            let task_context_ident = Ident::with_dummy_span(sym::_task_context);
            let lhs = self.expr_ident(span, task_context_ident, task_context_hid);

            hir::ExprKind::Assign(lhs, yield_expr, self.lower_span(span))
        } else {
            hir::ExprKind::Yield(yielded, hir::YieldSource::Yield)
        }
    }

    /// Desugar `ExprForLoop` from: `[opt_ident]: for <pat> in <head> <body>` into:
    /// ```ignore (pseudo-rust)
    /// {
    ///     let result = match IntoIterator::into_iter(<head>) {
    ///         mut iter => {
    ///             [opt_ident]: loop {
    ///                 match Iterator::next(&mut iter) {
    ///                     None => break,
    ///                     Some(<pat>) => <body>,
    ///                 };
    ///             }
    ///         }
    ///     };
    ///     result
    /// }
    /// ```
    fn lower_expr_for(
        &mut self,
        e: &Expr,
        pat: &Pat,
        head: &Expr,
        body: &Block,
        opt_label: Option<Label>,
        loop_kind: ForLoopKind,
    ) -> hir::Expr<'hir> {
        let head = self.lower_expr_mut(head);
        let pat = self.lower_pat(pat);
        let for_span =
            self.mark_span_with_reason(DesugaringKind::ForLoop, self.lower_span(e.span), None);
        let head_span = self.mark_span_with_reason(DesugaringKind::ForLoop, head.span, None);
        let pat_span = self.mark_span_with_reason(DesugaringKind::ForLoop, pat.span, None);

        let loop_hir_id = self.lower_node_id(e.id);
        let label = self.lower_label(opt_label, e.id, loop_hir_id);

        // `None => break`
        let none_arm = {
            let break_expr =
                self.with_loop_scope(loop_hir_id, |this| this.expr_break_alloc(for_span));
            let pat = self.pat_none(for_span);
            self.arm(pat, break_expr)
        };

        // Some(<pat>) => <body>,
        let some_arm = {
            let some_pat = self.pat_some(pat_span, pat);
            let body_block =
                self.with_loop_scope(loop_hir_id, |this| this.lower_block(body, false));
            let body_expr = self.arena.alloc(self.expr_block(body_block));
            self.arm(some_pat, body_expr)
        };

        // `mut iter`
        let iter = Ident::with_dummy_span(sym::iter);
        let (iter_pat, iter_pat_nid) =
            self.pat_ident_binding_mode(head_span, iter, hir::BindingMode::MUT);

        let match_expr = {
            let iter = self.expr_ident(head_span, iter, iter_pat_nid);
            let next_expr = match loop_kind {
                ForLoopKind::For => {
                    // `Iterator::next(&mut iter)`
                    let ref_mut_iter = self.expr_mut_addr_of(head_span, iter);
                    self.expr_call_lang_item_fn(
                        head_span,
                        hir::LangItem::IteratorNext,
                        arena_vec![self; ref_mut_iter],
                    )
                }
                ForLoopKind::ForAwait => {
                    // we'll generate `unsafe { Pin::new_unchecked(&mut iter) })` and then pass this
                    // to make_lowered_await with `FutureKind::AsyncIterator` which will generator
                    // calls to `poll_next`. In user code, this would probably be a call to
                    // `Pin::as_mut` but here it's easy enough to do `new_unchecked`.

                    // `&mut iter`
                    let iter = self.expr_mut_addr_of(head_span, iter);
                    // `Pin::new_unchecked(...)`
                    let iter = self.arena.alloc(self.expr_call_lang_item_fn_mut(
                        head_span,
                        hir::LangItem::PinNewUnchecked,
                        arena_vec![self; iter],
                    ));
                    // `unsafe { ... }`
                    let iter = self.arena.alloc(self.expr_unsafe(iter));
                    let kind = self.make_lowered_await(head_span, iter, FutureKind::AsyncIterator);
                    self.arena.alloc(hir::Expr { hir_id: self.next_id(), kind, span: head_span })
                }
            };
            let arms = arena_vec![self; none_arm, some_arm];

            // `match $next_expr { ... }`
            self.expr_match(head_span, next_expr, arms, hir::MatchSource::ForLoopDesugar)
        };
        let match_stmt = self.stmt_expr(for_span, match_expr);

        let loop_block = self.block_all(for_span, arena_vec![self; match_stmt], None);

        // `[opt_ident]: loop { ... }`
        let kind = hir::ExprKind::Loop(
            loop_block,
            label,
            hir::LoopSource::ForLoop,
            self.lower_span(for_span.with_hi(head.span.hi())),
        );
        let loop_expr = self.arena.alloc(hir::Expr { hir_id: loop_hir_id, kind, span: for_span });

        // `mut iter => { ... }`
        let iter_arm = self.arm(iter_pat, loop_expr);

        let match_expr = match loop_kind {
            ForLoopKind::For => {
                // `::std::iter::IntoIterator::into_iter(<head>)`
                let into_iter_expr = self.expr_call_lang_item_fn(
                    head_span,
                    hir::LangItem::IntoIterIntoIter,
                    arena_vec![self; head],
                );

                self.arena.alloc(self.expr_match(
                    for_span,
                    into_iter_expr,
                    arena_vec![self; iter_arm],
                    hir::MatchSource::ForLoopDesugar,
                ))
            }
            // `match into_async_iter(<head>) { ref mut iter => match unsafe { Pin::new_unchecked(iter) } { ... } }`
            ForLoopKind::ForAwait => {
                let iter_ident = iter;
                let (async_iter_pat, async_iter_pat_id) =
                    self.pat_ident_binding_mode(head_span, iter_ident, hir::BindingMode::REF_MUT);
                let iter = self.expr_ident_mut(head_span, iter_ident, async_iter_pat_id);
                // `Pin::new_unchecked(...)`
                let iter = self.arena.alloc(self.expr_call_lang_item_fn_mut(
                    head_span,
                    hir::LangItem::PinNewUnchecked,
                    arena_vec![self; iter],
                ));
                // `unsafe { ... }`
                let iter = self.arena.alloc(self.expr_unsafe(iter));
                let inner_match_expr = self.arena.alloc(self.expr_match(
                    for_span,
                    iter,
                    arena_vec![self; iter_arm],
                    hir::MatchSource::ForLoopDesugar,
                ));

                // `::core::async_iter::IntoAsyncIterator::into_async_iter(<head>)`
                let iter = self.expr_call_lang_item_fn(
                    head_span,
                    hir::LangItem::IntoAsyncIterIntoIter,
                    arena_vec![self; head],
                );
                let iter_arm = self.arm(async_iter_pat, inner_match_expr);
                self.arena.alloc(self.expr_match(
                    for_span,
                    iter,
                    arena_vec![self; iter_arm],
                    hir::MatchSource::ForLoopDesugar,
                ))
            }
        };

        // This is effectively `{ let _result = ...; _result }`.
        // The construct was introduced in #21984 and is necessary to make sure that
        // temporaries in the `head` expression are dropped and do not leak to the
        // surrounding scope of the `match` since the `match` is not a terminating scope.
        //
        // Also, add the attributes to the outer returned expr node.
        let expr = self.expr_drop_temps_mut(for_span, match_expr);
        self.lower_attrs(expr.hir_id, &e.attrs, e.span);
        expr
    }

    /// Desugar `ExprKind::Try` from: `<expr>?` into:
    /// ```ignore (pseudo-rust)
    /// match Try::branch(<expr>) {
    ///     ControlFlow::Continue(val) => #[allow(unreachable_code)] val,,
    ///     ControlFlow::Break(residual) =>
    ///         #[allow(unreachable_code)]
    ///         // If there is an enclosing `try {...}`:
    ///         break 'catch_target Try::from_residual(residual),
    ///         // Otherwise:
    ///         return Try::from_residual(residual),
    /// }
    /// ```
    fn lower_expr_try(&mut self, span: Span, sub_expr: &Expr) -> hir::ExprKind<'hir> {
        let unstable_span = self.mark_span_with_reason(
            DesugaringKind::QuestionMark,
            span,
            Some(Arc::clone(&self.allow_try_trait)),
        );
        let try_span = self.tcx.sess.source_map().end_point(span);
        let try_span = self.mark_span_with_reason(
            DesugaringKind::QuestionMark,
            try_span,
            Some(Arc::clone(&self.allow_try_trait)),
        );

        // `Try::branch(<expr>)`
        let scrutinee = {
            // expand <expr>
            let sub_expr = self.lower_expr_mut(sub_expr);

            self.expr_call_lang_item_fn(
                unstable_span,
                hir::LangItem::TryTraitBranch,
                arena_vec![self; sub_expr],
            )
        };

        // `#[allow(unreachable_code)]`
        let attr = attr::mk_attr_nested_word(
            &self.tcx.sess.psess.attr_id_generator,
            AttrStyle::Outer,
            Safety::Default,
            sym::allow,
            sym::unreachable_code,
            try_span,
        );
        let attrs: AttrVec = thin_vec![attr];

        // `ControlFlow::Continue(val) => #[allow(unreachable_code)] val,`
        let continue_arm = {
            let val_ident = Ident::with_dummy_span(sym::val);
            let (val_pat, val_pat_nid) = self.pat_ident(span, val_ident);
            let val_expr = self.expr_ident(span, val_ident, val_pat_nid);
            self.lower_attrs(val_expr.hir_id, &attrs, span);
            let continue_pat = self.pat_cf_continue(unstable_span, val_pat);
            self.arm(continue_pat, val_expr)
        };

        // `ControlFlow::Break(residual) =>
        //     #[allow(unreachable_code)]
        //     return Try::from_residual(residual),`
        let break_arm = {
            let residual_ident = Ident::with_dummy_span(sym::residual);
            let (residual_local, residual_local_nid) = self.pat_ident(try_span, residual_ident);
            let residual_expr = self.expr_ident_mut(try_span, residual_ident, residual_local_nid);
            let from_residual_expr = self.wrap_in_try_constructor(
                hir::LangItem::TryTraitFromResidual,
                try_span,
                self.arena.alloc(residual_expr),
                unstable_span,
            );
            let ret_expr = if let Some(catch_id) = self.catch_scope {
                let target_id = Ok(catch_id);
                self.arena.alloc(self.expr(
                    try_span,
                    hir::ExprKind::Break(
                        hir::Destination { label: None, target_id },
                        Some(from_residual_expr),
                    ),
                ))
            } else {
                let ret_expr = self.checked_return(Some(from_residual_expr));
                self.arena.alloc(self.expr(try_span, ret_expr))
            };
            self.lower_attrs(ret_expr.hir_id, &attrs, ret_expr.span);

            let break_pat = self.pat_cf_break(try_span, residual_local);
            self.arm(break_pat, ret_expr)
        };

        hir::ExprKind::Match(
            scrutinee,
            arena_vec![self; break_arm, continue_arm],
            hir::MatchSource::TryDesugar(scrutinee.hir_id),
        )
    }

    /// Desugar `ExprKind::Yeet` from: `do yeet <expr>` into:
    /// ```ignore(illustrative)
    /// // If there is an enclosing `try {...}`:
    /// break 'catch_target FromResidual::from_residual(Yeet(residual));
    /// // Otherwise:
    /// return FromResidual::from_residual(Yeet(residual));
    /// ```
    /// But to simplify this, there's a `from_yeet` lang item function which
    /// handles the combined `FromResidual::from_residual(Yeet(residual))`.
    fn lower_expr_yeet(&mut self, span: Span, sub_expr: Option<&Expr>) -> hir::ExprKind<'hir> {
        // The expression (if present) or `()` otherwise.
        let (yeeted_span, yeeted_expr) = if let Some(sub_expr) = sub_expr {
            (sub_expr.span, self.lower_expr(sub_expr))
        } else {
            (self.mark_span_with_reason(DesugaringKind::YeetExpr, span, None), self.expr_unit(span))
        };

        let unstable_span = self.mark_span_with_reason(
            DesugaringKind::YeetExpr,
            span,
            Some(Arc::clone(&self.allow_try_trait)),
        );

        let from_yeet_expr = self.wrap_in_try_constructor(
            hir::LangItem::TryTraitFromYeet,
            unstable_span,
            yeeted_expr,
            yeeted_span,
        );

        if let Some(catch_id) = self.catch_scope {
            let target_id = Ok(catch_id);
            hir::ExprKind::Break(hir::Destination { label: None, target_id }, Some(from_yeet_expr))
        } else {
            self.checked_return(Some(from_yeet_expr))
        }
    }

    // =========================================================================
    // Helper methods for building HIR.
    // =========================================================================

    /// Wrap the given `expr` in a terminating scope using `hir::ExprKind::DropTemps`.
    ///
    /// In terms of drop order, it has the same effect as wrapping `expr` in
    /// `{ let _t = $expr; _t }` but should provide better compile-time performance.
    ///
    /// The drop order can be important in e.g. `if expr { .. }`.
    pub(super) fn expr_drop_temps(
        &mut self,
        span: Span,
        expr: &'hir hir::Expr<'hir>,
    ) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.expr_drop_temps_mut(span, expr))
    }

    pub(super) fn expr_drop_temps_mut(
        &mut self,
        span: Span,
        expr: &'hir hir::Expr<'hir>,
    ) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::DropTemps(expr))
    }

    pub(super) fn expr_match(
        &mut self,
        span: Span,
        arg: &'hir hir::Expr<'hir>,
        arms: &'hir [hir::Arm<'hir>],
        source: hir::MatchSource,
    ) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::Match(arg, arms, source))
    }

    fn expr_break(&mut self, span: Span) -> hir::Expr<'hir> {
        let expr_break = hir::ExprKind::Break(self.lower_loop_destination(None), None);
        self.expr(span, expr_break)
    }

    fn expr_break_alloc(&mut self, span: Span) -> &'hir hir::Expr<'hir> {
        let expr_break = self.expr_break(span);
        self.arena.alloc(expr_break)
    }

    fn expr_mut_addr_of(&mut self, span: Span, e: &'hir hir::Expr<'hir>) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Mut, e))
    }

    fn expr_unit(&mut self, sp: Span) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.expr(sp, hir::ExprKind::Tup(&[])))
    }

    fn expr_uint(&mut self, sp: Span, ty: ast::UintTy, value: u128) -> hir::Expr<'hir> {
        let lit = self.arena.alloc(hir::Lit {
            span: sp,
            node: ast::LitKind::Int(value.into(), ast::LitIntType::Unsigned(ty)),
        });
        self.expr(sp, hir::ExprKind::Lit(lit))
    }

    pub(super) fn expr_usize(&mut self, sp: Span, value: usize) -> hir::Expr<'hir> {
        self.expr_uint(sp, ast::UintTy::Usize, value as u128)
    }

    pub(super) fn expr_u32(&mut self, sp: Span, value: u32) -> hir::Expr<'hir> {
        self.expr_uint(sp, ast::UintTy::U32, value as u128)
    }

    pub(super) fn expr_u16(&mut self, sp: Span, value: u16) -> hir::Expr<'hir> {
        self.expr_uint(sp, ast::UintTy::U16, value as u128)
    }

    pub(super) fn expr_str(&mut self, sp: Span, value: Symbol) -> hir::Expr<'hir> {
        let lit = self
            .arena
            .alloc(hir::Lit { span: sp, node: ast::LitKind::Str(value, ast::StrStyle::Cooked) });
        self.expr(sp, hir::ExprKind::Lit(lit))
    }

    pub(super) fn expr_call_mut(
        &mut self,
        span: Span,
        e: &'hir hir::Expr<'hir>,
        args: &'hir [hir::Expr<'hir>],
    ) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::Call(e, args))
    }

    pub(super) fn expr_call(
        &mut self,
        span: Span,
        e: &'hir hir::Expr<'hir>,
        args: &'hir [hir::Expr<'hir>],
    ) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.expr_call_mut(span, e, args))
    }

    pub(super) fn expr_call_lang_item_fn_mut(
        &mut self,
        span: Span,
        lang_item: hir::LangItem,
        args: &'hir [hir::Expr<'hir>],
    ) -> hir::Expr<'hir> {
        let path = self.arena.alloc(self.expr_lang_item_path(span, lang_item));
        self.expr_call_mut(span, path, args)
    }

    pub(super) fn expr_call_lang_item_fn(
        &mut self,
        span: Span,
        lang_item: hir::LangItem,
        args: &'hir [hir::Expr<'hir>],
    ) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.expr_call_lang_item_fn_mut(span, lang_item, args))
    }

    fn expr_lang_item_path(&mut self, span: Span, lang_item: hir::LangItem) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::Path(hir::QPath::LangItem(lang_item, self.lower_span(span))))
    }

    /// `<LangItem>::name`
    pub(super) fn expr_lang_item_type_relative(
        &mut self,
        span: Span,
        lang_item: hir::LangItem,
        name: Symbol,
    ) -> hir::Expr<'hir> {
        let qpath = self.make_lang_item_qpath(lang_item, self.lower_span(span), None);
        let path = hir::ExprKind::Path(hir::QPath::TypeRelative(
            self.arena.alloc(self.ty(span, hir::TyKind::Path(qpath))),
            self.arena.alloc(hir::PathSegment::new(
                Ident::new(name, self.lower_span(span)),
                self.next_id(),
                Res::Err,
            )),
        ));
        self.expr(span, path)
    }

    pub(super) fn expr_ident(
        &mut self,
        sp: Span,
        ident: Ident,
        binding: HirId,
    ) -> &'hir hir::Expr<'hir> {
        self.arena.alloc(self.expr_ident_mut(sp, ident, binding))
    }

    pub(super) fn expr_ident_mut(
        &mut self,
        span: Span,
        ident: Ident,
        binding: HirId,
    ) -> hir::Expr<'hir> {
        let hir_id = self.next_id();
        let res = Res::Local(binding);
        let expr_path = hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            self.arena.alloc(hir::Path {
                span: self.lower_span(span),
                res,
                segments: arena_vec![self; hir::PathSegment::new(ident, hir_id, res)],
            }),
        ));

        self.expr(span, expr_path)
    }

    fn expr_unsafe(&mut self, expr: &'hir hir::Expr<'hir>) -> hir::Expr<'hir> {
        let hir_id = self.next_id();
        let span = expr.span;
        self.expr(
            span,
            hir::ExprKind::Block(
                self.arena.alloc(hir::Block {
                    stmts: &[],
                    expr: Some(expr),
                    hir_id,
                    rules: hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::CompilerGenerated),
                    span: self.lower_span(span),
                    targeted_by_break: false,
                }),
                None,
            ),
        )
    }

    fn expr_block_empty(&mut self, span: Span) -> &'hir hir::Expr<'hir> {
        let blk = self.block_all(span, &[], None);
        let expr = self.expr_block(blk);
        self.arena.alloc(expr)
    }

    pub(super) fn expr_block(&mut self, b: &'hir hir::Block<'hir>) -> hir::Expr<'hir> {
        self.expr(b.span, hir::ExprKind::Block(b, None))
    }

    pub(super) fn expr_array_ref(
        &mut self,
        span: Span,
        elements: &'hir [hir::Expr<'hir>],
    ) -> hir::Expr<'hir> {
        let addrof = hir::ExprKind::AddrOf(
            hir::BorrowKind::Ref,
            hir::Mutability::Not,
            self.arena.alloc(self.expr(span, hir::ExprKind::Array(elements))),
        );
        self.expr(span, addrof)
    }

    pub(super) fn expr(&mut self, span: Span, kind: hir::ExprKind<'hir>) -> hir::Expr<'hir> {
        let hir_id = self.next_id();
        hir::Expr { hir_id, kind, span: self.lower_span(span) }
    }

    pub(super) fn expr_field(
        &mut self,
        ident: Ident,
        expr: &'hir hir::Expr<'hir>,
        span: Span,
    ) -> hir::ExprField<'hir> {
        hir::ExprField {
            hir_id: self.next_id(),
            ident,
            span: self.lower_span(span),
            expr,
            is_shorthand: false,
        }
    }

    pub(super) fn arm(
        &mut self,
        pat: &'hir hir::Pat<'hir>,
        expr: &'hir hir::Expr<'hir>,
    ) -> hir::Arm<'hir> {
        hir::Arm {
            hir_id: self.next_id(),
            pat,
            guard: None,
            span: self.lower_span(expr.span),
            body: expr,
        }
    }
}

/// Used by [`LoweringContext::make_lowered_await`] to customize the desugaring based on what kind
/// of future we are awaiting.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FutureKind {
    /// We are awaiting a normal future
    Future,
    /// We are awaiting something that's known to be an AsyncIterator (i.e. we are in the header of
    /// a `for await` loop)
    AsyncIterator,
}
