use super::{LoweringContext, ParamMode, ParenthesizedGenericArgs, ImplTraitContext};
use crate::hir::{self, HirVec};
use crate::hir::def::Res;
use crate::hir::ptr::P;

use rustc_data_structures::thin_vec::ThinVec;

use syntax::attr;
use syntax::ptr::P as AstP;
use syntax::ast::*;
use syntax::source_map::{respan, DesugaringKind, Span, Spanned};
use syntax::symbol::{sym, Symbol};

impl LoweringContext<'_> {
    fn lower_exprs(&mut self, exprs: &[AstP<Expr>]) -> HirVec<hir::Expr> {
        exprs.iter().map(|x| self.lower_expr(x)).collect()
    }

    pub(super) fn lower_expr(&mut self, e: &Expr) -> hir::Expr {
        let kind = match e.node {
            ExprKind::Box(ref inner) => hir::ExprKind::Box(P(self.lower_expr(inner))),
            ExprKind::Array(ref exprs) => hir::ExprKind::Array(self.lower_exprs(exprs)),
            ExprKind::Repeat(ref expr, ref count) => {
                let expr = P(self.lower_expr(expr));
                let count = self.lower_anon_const(count);
                hir::ExprKind::Repeat(expr, count)
            }
            ExprKind::Tup(ref elts) => hir::ExprKind::Tup(self.lower_exprs(elts)),
            ExprKind::Call(ref f, ref args) => {
                let f = P(self.lower_expr(f));
                hir::ExprKind::Call(f, self.lower_exprs(args))
            }
            ExprKind::MethodCall(ref seg, ref args) => {
                let hir_seg = P(self.lower_path_segment(
                    e.span,
                    seg,
                    ParamMode::Optional,
                    0,
                    ParenthesizedGenericArgs::Err,
                    ImplTraitContext::disallowed(),
                    None,
                ));
                let args = self.lower_exprs(args);
                hir::ExprKind::MethodCall(hir_seg, seg.ident.span, args)
            }
            ExprKind::Binary(binop, ref lhs, ref rhs) => {
                let binop = self.lower_binop(binop);
                let lhs = P(self.lower_expr(lhs));
                let rhs = P(self.lower_expr(rhs));
                hir::ExprKind::Binary(binop, lhs, rhs)
            }
            ExprKind::Unary(op, ref ohs) => {
                let op = self.lower_unop(op);
                let ohs = P(self.lower_expr(ohs));
                hir::ExprKind::Unary(op, ohs)
            }
            ExprKind::Lit(ref l) => hir::ExprKind::Lit(respan(l.span, l.node.clone())),
            ExprKind::Cast(ref expr, ref ty) => {
                let expr = P(self.lower_expr(expr));
                hir::ExprKind::Cast(expr, self.lower_ty(ty, ImplTraitContext::disallowed()))
            }
            ExprKind::Type(ref expr, ref ty) => {
                let expr = P(self.lower_expr(expr));
                hir::ExprKind::Type(expr, self.lower_ty(ty, ImplTraitContext::disallowed()))
            }
            ExprKind::AddrOf(m, ref ohs) => {
                let m = self.lower_mutability(m);
                let ohs = P(self.lower_expr(ohs));
                hir::ExprKind::AddrOf(m, ohs)
            }
            ExprKind::Let(ref pats, ref scrutinee) => self.lower_expr_let(e.span, pats, scrutinee),
            ExprKind::If(ref cond, ref then, ref else_opt) => {
                self.lower_expr_if(e.span, cond, then, else_opt.as_deref())
            }
            ExprKind::While(ref cond, ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                this.lower_expr_while_in_loop_scope(e.span, cond, body, opt_label)
            }),
            ExprKind::Loop(ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                hir::ExprKind::Loop(
                    this.lower_block(body, false),
                    this.lower_label(opt_label),
                    hir::LoopSource::Loop,
                )
            }),
            ExprKind::TryBlock(ref body) => self.lower_expr_try_block(body),
            ExprKind::Match(ref expr, ref arms) => hir::ExprKind::Match(
                P(self.lower_expr(expr)),
                arms.iter().map(|x| self.lower_arm(x)).collect(),
                hir::MatchSource::Normal,
            ),
            ExprKind::Async(capture_clause, closure_node_id, ref block) => {
                self.make_async_expr(capture_clause, closure_node_id, None, block.span, |this| {
                    this.with_new_scopes(|this| {
                        let block = this.lower_block(block, false);
                        this.expr_block(block, ThinVec::new())
                    })
                })
            }
            ExprKind::Await(ref expr) => self.lower_expr_await(e.span, expr),
            ExprKind::Closure(
                capture_clause, asyncness, movability, ref decl, ref body, fn_decl_span
            ) => if let IsAsync::Async { closure_id, .. } = asyncness {
                self.lower_expr_async_closure(capture_clause, closure_id, decl, body, fn_decl_span)
            } else {
                self.lower_expr_closure(capture_clause, movability, decl, body, fn_decl_span)
            }
            ExprKind::Block(ref blk, opt_label) => {
                hir::ExprKind::Block(self.lower_block(blk,
                                                      opt_label.is_some()),
                                                      self.lower_label(opt_label))
            }
            ExprKind::Assign(ref el, ref er) => {
                hir::ExprKind::Assign(P(self.lower_expr(el)), P(self.lower_expr(er)))
            }
            ExprKind::AssignOp(op, ref el, ref er) => hir::ExprKind::AssignOp(
                self.lower_binop(op),
                P(self.lower_expr(el)),
                P(self.lower_expr(er)),
            ),
            ExprKind::Field(ref el, ident) => hir::ExprKind::Field(P(self.lower_expr(el)), ident),
            ExprKind::Index(ref el, ref er) => {
                hir::ExprKind::Index(P(self.lower_expr(el)), P(self.lower_expr(er)))
            }
            ExprKind::Range(Some(ref e1), Some(ref e2), RangeLimits::Closed) => {
                self.lower_expr_range_closed(e.span, e1, e2)
            }
            ExprKind::Range(ref e1, ref e2, lims) => {
                self.lower_expr_range(e.span, e1.as_deref(), e2.as_deref(), lims)
            }
            ExprKind::Path(ref qself, ref path) => {
                let qpath = self.lower_qpath(
                    e.id,
                    qself,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );
                hir::ExprKind::Path(qpath)
            }
            ExprKind::Break(opt_label, ref opt_expr) => {
                hir::ExprKind::Break(
                    self.lower_jump_destination(e.id, opt_label),
                    opt_expr.as_ref().map(|x| P(self.lower_expr(x))),
                )
            }
            ExprKind::Continue(opt_label) => {
                hir::ExprKind::Continue(self.lower_jump_destination(e.id, opt_label))
            }
            ExprKind::Ret(ref e) => hir::ExprKind::Ret(e.as_ref().map(|x| P(self.lower_expr(x)))),
            ExprKind::InlineAsm(ref asm) => self.lower_expr_asm(asm),
            ExprKind::Struct(ref path, ref fields, ref maybe_expr) => hir::ExprKind::Struct(
                P(self.lower_qpath(
                    e.id,
                    &None,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                )),
                fields.iter().map(|x| self.lower_field(x)).collect(),
                maybe_expr.as_ref().map(|x| P(self.lower_expr(x))),
            ),
            ExprKind::Paren(ref ex) => {
                let mut ex = self.lower_expr(ex);
                // Include parens in span, but only if it is a super-span.
                if e.span.contains(ex.span) {
                    ex.span = e.span;
                }
                // Merge attributes into the inner expression.
                let mut attrs = e.attrs.clone();
                attrs.extend::<Vec<_>>(ex.attrs.into());
                ex.attrs = attrs;
                return ex;
            }

            ExprKind::Yield(ref opt_expr) => self.lower_expr_yield(e.span, opt_expr.as_deref()),

            ExprKind::Err => hir::ExprKind::Err,

            // Desugar `ExprForLoop`
            // from: `[opt_ident]: for <pat> in <head> <body>`
            ExprKind::ForLoop(ref pat, ref head, ref body, opt_label) => {
                return self.lower_expr_for(e, pat, head, body, opt_label);
            }
            ExprKind::Try(ref sub_expr) => self.lower_expr_try(e.span, sub_expr),
            ExprKind::Mac(_) => panic!("Shouldn't exist here"),
        };

        hir::Expr {
            hir_id: self.lower_node_id(e.id),
            node: kind,
            span: e.span,
            attrs: e.attrs.clone(),
        }
    }

    fn lower_unop(&mut self, u: UnOp) -> hir::UnOp {
        match u {
            UnOp::Deref => hir::UnDeref,
            UnOp::Not => hir::UnNot,
            UnOp::Neg => hir::UnNeg,
        }
    }

    fn lower_binop(&mut self, b: BinOp) -> hir::BinOp {
        Spanned {
            node: match b.node {
                BinOpKind::Add => hir::BinOpKind::Add,
                BinOpKind::Sub => hir::BinOpKind::Sub,
                BinOpKind::Mul => hir::BinOpKind::Mul,
                BinOpKind::Div => hir::BinOpKind::Div,
                BinOpKind::Rem => hir::BinOpKind::Rem,
                BinOpKind::And => hir::BinOpKind::And,
                BinOpKind::Or => hir::BinOpKind::Or,
                BinOpKind::BitXor => hir::BinOpKind::BitXor,
                BinOpKind::BitAnd => hir::BinOpKind::BitAnd,
                BinOpKind::BitOr => hir::BinOpKind::BitOr,
                BinOpKind::Shl => hir::BinOpKind::Shl,
                BinOpKind::Shr => hir::BinOpKind::Shr,
                BinOpKind::Eq => hir::BinOpKind::Eq,
                BinOpKind::Lt => hir::BinOpKind::Lt,
                BinOpKind::Le => hir::BinOpKind::Le,
                BinOpKind::Ne => hir::BinOpKind::Ne,
                BinOpKind::Ge => hir::BinOpKind::Ge,
                BinOpKind::Gt => hir::BinOpKind::Gt,
            },
            span: b.span,
        }
    }

    /// Emit an error and lower `ast::ExprKind::Let(pats, scrutinee)` into:
    /// ```rust
    /// match scrutinee { pats => true, _ => false }
    /// ```
    fn lower_expr_let(
        &mut self,
        span: Span,
        pats: &[AstP<Pat>],
        scrutinee: &Expr
    ) -> hir::ExprKind {
        // If we got here, the `let` expression is not allowed.
        self.sess
            .struct_span_err(span, "`let` expressions are not supported here")
            .note("only supported directly in conditions of `if`- and `while`-expressions")
            .note("as well as when nested within `&&` and parenthesis in those conditions")
            .emit();

        // For better recovery, we emit:
        // ```
        // match scrutinee { pats => true, _ => false }
        // ```
        // While this doesn't fully match the user's intent, it has key advantages:
        // 1. We can avoid using `abort_if_errors`.
        // 2. We can typeck both `pats` and `scrutinee`.
        // 3. `pats` is allowed to be refutable.
        // 4. The return type of the block is `bool` which seems like what the user wanted.
        let scrutinee = self.lower_expr(scrutinee);
        let then_arm = {
            let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
            let expr = self.expr_bool(span, true);
            self.arm(pats, P(expr))
        };
        let else_arm = {
            let pats = hir_vec![self.pat_wild(span)];
            let expr = self.expr_bool(span, false);
            self.arm(pats, P(expr))
        };
        hir::ExprKind::Match(
            P(scrutinee),
            vec![then_arm, else_arm].into(),
            hir::MatchSource::Normal,
        )
    }

    fn lower_expr_if(
        &mut self,
        span: Span,
        cond: &Expr,
        then: &Block,
        else_opt: Option<&Expr>,
    ) -> hir::ExprKind {
        // FIXME(#53667): handle lowering of && and parens.

        // `_ => else_block` where `else_block` is `{}` if there's `None`:
        let else_pat = self.pat_wild(span);
        let (else_expr, contains_else_clause) = match else_opt {
            None => (self.expr_block_empty(span), false),
            Some(els) => (self.lower_expr(els), true),
        };
        let else_arm = self.arm(hir_vec![else_pat], P(else_expr));

        // Handle then + scrutinee:
        let then_blk = self.lower_block(then, false);
        let then_expr = self.expr_block(then_blk, ThinVec::new());
        let (then_pats, scrutinee, desugar) = match cond.node {
            // `<pat> => <then>`:
            ExprKind::Let(ref pats, ref scrutinee) => {
                let scrutinee = self.lower_expr(scrutinee);
                let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                let desugar = hir::MatchSource::IfLetDesugar { contains_else_clause };
                (pats, scrutinee, desugar)
            }
            // `true => <then>`:
            _ => {
                // Lower condition:
                let cond = self.lower_expr(cond);
                let span_block = self.mark_span_with_reason(
                    DesugaringKind::CondTemporary,
                    cond.span,
                    None
                );
                // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                // to preserve drop semantics since `if cond { ... }` does not
                // let temporaries live outside of `cond`.
                let cond = self.expr_drop_temps(span_block, P(cond), ThinVec::new());

                let desugar = hir::MatchSource::IfDesugar { contains_else_clause };
                let pats = hir_vec![self.pat_bool(span, true)];
                (pats, cond, desugar)
            }
        };
        let then_arm = self.arm(then_pats, P(then_expr));

        hir::ExprKind::Match(P(scrutinee), vec![then_arm, else_arm].into(), desugar)
    }

    fn lower_expr_while_in_loop_scope(
        &mut self,
        span: Span,
        cond: &Expr,
        body: &Block,
        opt_label: Option<Label>
    ) -> hir::ExprKind {
        // FIXME(#53667): handle lowering of && and parens.

        // Note that the block AND the condition are evaluated in the loop scope.
        // This is done to allow `break` from inside the condition of the loop.

        // `_ => break`:
        let else_arm = {
            let else_pat = self.pat_wild(span);
            let else_expr = self.expr_break(span, ThinVec::new());
            self.arm(hir_vec![else_pat], else_expr)
        };

        // Handle then + scrutinee:
        let then_blk = self.lower_block(body, false);
        let then_expr = self.expr_block(then_blk, ThinVec::new());
        let (then_pats, scrutinee, desugar, source) = match cond.node {
            ExprKind::Let(ref pats, ref scrutinee) => {
                // to:
                //
                //   [opt_ident]: loop {
                //     match <sub_expr> {
                //       <pat> => <body>,
                //       _ => break
                //     }
                //   }
                let scrutinee = self.with_loop_condition_scope(|t| t.lower_expr(scrutinee));
                let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                let desugar = hir::MatchSource::WhileLetDesugar;
                (pats, scrutinee, desugar, hir::LoopSource::WhileLet)
            }
            _ => {
                // We desugar: `'label: while $cond $body` into:
                //
                // ```
                // 'label: loop {
                //     match DropTemps($cond) {
                //         true => $body,
                //         _ => break,
                //     }
                // }
                // ```

                // Lower condition:
                let cond = self.with_loop_condition_scope(|this| this.lower_expr(cond));
                let span_block = self.mark_span_with_reason(
                    DesugaringKind::CondTemporary,
                    cond.span,
                    None,
                );
                // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                // to preserve drop semantics since `while cond { ... }` does not
                // let temporaries live outside of `cond`.
                let cond = self.expr_drop_temps(span_block, P(cond), ThinVec::new());

                let desugar = hir::MatchSource::WhileDesugar;
                // `true => <then>`:
                let pats = hir_vec![self.pat_bool(span, true)];
                (pats, cond, desugar, hir::LoopSource::While)
            }
        };
        let then_arm = self.arm(then_pats, P(then_expr));

        // `match <scrutinee> { ... }`
        let match_expr = self.expr_match(
            scrutinee.span,
            P(scrutinee),
            hir_vec![then_arm, else_arm],
            desugar,
        );

        // `[opt_ident]: loop { ... }`
        hir::ExprKind::Loop(
            P(self.block_expr(P(match_expr))),
            self.lower_label(opt_label),
            source
        )
    }

    fn lower_expr_try_block(&mut self, body: &Block) -> hir::ExprKind {
        self.with_catch_scope(body.id, |this| {
            let unstable_span = this.mark_span_with_reason(
                DesugaringKind::TryBlock,
                body.span,
                this.allow_try_trait.clone(),
            );
            let mut block = this.lower_block(body, true).into_inner();
            let tail = block.expr.take().map_or_else(
                || this.expr_unit(this.sess.source_map().end_point(unstable_span)),
                |x: P<hir::Expr>| x.into_inner(),
            );
            block.expr = Some(this.wrap_in_try_constructor(sym::from_ok, tail, unstable_span));
            hir::ExprKind::Block(P(block), None)
        })
    }

    fn wrap_in_try_constructor(
        &mut self,
        method: Symbol,
        e: hir::Expr,
        unstable_span: Span,
    ) -> P<hir::Expr> {
        let path = &[sym::ops, sym::Try, method];
        let from_err = P(self.expr_std_path(unstable_span, path, None, ThinVec::new()));
        P(self.expr_call(e.span, from_err, hir_vec![e]))
    }

    fn lower_arm(&mut self, arm: &Arm) -> hir::Arm {
        hir::Arm {
            hir_id: self.next_id(),
            attrs: self.lower_attrs(&arm.attrs),
            pats: arm.pats.iter().map(|x| self.lower_pat(x)).collect(),
            guard: match arm.guard {
                Some(ref x) => Some(hir::Guard::If(P(self.lower_expr(x)))),
                _ => None,
            },
            body: P(self.lower_expr(&arm.body)),
            span: arm.span,
        }
    }

    pub(super) fn make_async_expr(
        &mut self,
        capture_clause: CaptureBy,
        closure_node_id: NodeId,
        ret_ty: Option<AstP<Ty>>,
        span: Span,
        body: impl FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    ) -> hir::ExprKind {
        let capture_clause = self.lower_capture_clause(capture_clause);
        let output = match ret_ty {
            Some(ty) => FunctionRetTy::Ty(ty),
            None => FunctionRetTy::Default(span),
        };
        let ast_decl = FnDecl {
            inputs: vec![],
            output,
            c_variadic: false
        };
        let decl = self.lower_fn_decl(&ast_decl, None, /* impl trait allowed */ false, None);
        let body_id = self.lower_fn_body(&ast_decl, |this| {
            this.generator_kind = Some(hir::GeneratorKind::Async);
            body(this)
        });

        // `static || -> <ret_ty> { body }`:
        let generator_node = hir::ExprKind::Closure(
            capture_clause,
            decl,
            body_id,
            span,
            Some(hir::GeneratorMovability::Static)
        );
        let generator = hir::Expr {
            hir_id: self.lower_node_id(closure_node_id),
            node: generator_node,
            span,
            attrs: ThinVec::new(),
        };

        // `future::from_generator`:
        let unstable_span = self.mark_span_with_reason(
            DesugaringKind::Async,
            span,
            self.allow_gen_future.clone(),
        );
        let gen_future = self.expr_std_path(
            unstable_span,
            &[sym::future, sym::from_generator],
            None,
            ThinVec::new()
        );

        // `future::from_generator(generator)`:
        hir::ExprKind::Call(P(gen_future), hir_vec![generator])
    }

    /// Desugar `<expr>.await` into:
    /// ```rust
    /// {
    ///     let mut pinned = <expr>;
    ///     loop {
    ///         match ::std::future::poll_with_tls_context(unsafe {
    ///             ::std::pin::Pin::new_unchecked(&mut pinned)
    ///         }) {
    ///             ::std::task::Poll::Ready(result) => break result,
    ///             ::std::task::Poll::Pending => {},
    ///         }
    ///         yield ();
    ///     }
    /// }
    /// ```
    fn lower_expr_await(&mut self, await_span: Span, expr: &Expr) -> hir::ExprKind {
        match self.generator_kind {
            Some(hir::GeneratorKind::Async) => {},
            Some(hir::GeneratorKind::Gen) |
            None => {
                let mut err = struct_span_err!(
                    self.sess,
                    await_span,
                    E0728,
                    "`await` is only allowed inside `async` functions and blocks"
                );
                err.span_label(await_span, "only allowed inside `async` functions and blocks");
                if let Some(item_sp) = self.current_item {
                    err.span_label(item_sp, "this is not `async`");
                }
                err.emit();
            }
        }
        let span = self.mark_span_with_reason(
            DesugaringKind::Await,
            await_span,
            None,
        );
        let gen_future_span = self.mark_span_with_reason(
            DesugaringKind::Await,
            await_span,
            self.allow_gen_future.clone(),
        );

        // let mut pinned = <expr>;
        let expr = P(self.lower_expr(expr));
        let pinned_ident = Ident::with_empty_ctxt(sym::pinned);
        let (pinned_pat, pinned_pat_hid) = self.pat_ident_binding_mode(
            span,
            pinned_ident,
            hir::BindingAnnotation::Mutable,
        );
        let pinned_let = self.stmt_let_pat(
            ThinVec::new(),
            span,
            Some(expr),
            pinned_pat,
            hir::LocalSource::AwaitDesugar,
        );

        // ::std::future::poll_with_tls_context(unsafe {
        //     ::std::pin::Pin::new_unchecked(&mut pinned)
        // })`
        let poll_expr = {
            let pinned = P(self.expr_ident(span, pinned_ident, pinned_pat_hid));
            let ref_mut_pinned = self.expr_mut_addr_of(span, pinned);
            let pin_ty_id = self.next_id();
            let new_unchecked_expr_kind = self.expr_call_std_assoc_fn(
                pin_ty_id,
                span,
                &[sym::pin, sym::Pin],
                "new_unchecked",
                hir_vec![ref_mut_pinned],
            );
            let new_unchecked = P(self.expr(span, new_unchecked_expr_kind, ThinVec::new()));
            let unsafe_expr = self.expr_unsafe(new_unchecked);
            P(self.expr_call_std_path(
                gen_future_span,
                &[sym::future, sym::poll_with_tls_context],
                hir_vec![unsafe_expr],
            ))
        };

        // `::std::task::Poll::Ready(result) => break result`
        let loop_node_id = self.sess.next_node_id();
        let loop_hir_id = self.lower_node_id(loop_node_id);
        let ready_arm = {
            let x_ident = Ident::with_empty_ctxt(sym::result);
            let (x_pat, x_pat_hid) = self.pat_ident(span, x_ident);
            let x_expr = P(self.expr_ident(span, x_ident, x_pat_hid));
            let ready_pat = self.pat_std_enum(
                span,
                &[sym::task, sym::Poll, sym::Ready],
                hir_vec![x_pat],
            );
            let break_x = self.with_loop_scope(loop_node_id, |this| {
                let expr_break = hir::ExprKind::Break(
                    this.lower_loop_destination(None),
                    Some(x_expr),
                );
                P(this.expr(await_span, expr_break, ThinVec::new()))
            });
            self.arm(hir_vec![ready_pat], break_x)
        };

        // `::std::task::Poll::Pending => {}`
        let pending_arm = {
            let pending_pat = self.pat_std_enum(
                span,
                &[sym::task, sym::Poll, sym::Pending],
                hir_vec![],
            );
            let empty_block = P(self.expr_block_empty(span));
            self.arm(hir_vec![pending_pat], empty_block)
        };

        let match_stmt = {
            let match_expr = self.expr_match(
                span,
                poll_expr,
                hir_vec![ready_arm, pending_arm],
                hir::MatchSource::AwaitDesugar,
            );
            self.stmt_expr(span, match_expr)
        };

        let yield_stmt = {
            let unit = self.expr_unit(span);
            let yield_expr = self.expr(
                span,
                hir::ExprKind::Yield(P(unit), hir::YieldSource::Await),
                ThinVec::new(),
            );
            self.stmt_expr(span, yield_expr)
        };

        let loop_block = P(self.block_all(
            span,
            hir_vec![match_stmt, yield_stmt],
            None,
        ));

        let loop_expr = P(hir::Expr {
            hir_id: loop_hir_id,
            node: hir::ExprKind::Loop(
                loop_block,
                None,
                hir::LoopSource::Loop,
            ),
            span,
            attrs: ThinVec::new(),
        });

        hir::ExprKind::Block(
            P(self.block_all(span, hir_vec![pinned_let], Some(loop_expr))),
            None,
        )
    }

    fn lower_expr_closure(
        &mut self,
        capture_clause: CaptureBy,
        movability: Movability,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
    ) -> hir::ExprKind {
        // Lower outside new scope to preserve `is_in_loop_condition`.
        let fn_decl = self.lower_fn_decl(decl, None, false, None);

        self.with_new_scopes(|this| {
            let prev = this.current_item;
            this.current_item = Some(fn_decl_span);
            let mut generator_kind = None;
            let body_id = this.lower_fn_body(decl, |this| {
                let e = this.lower_expr(body);
                generator_kind = this.generator_kind;
                e
            });
            let generator_option = this.generator_movability_for_fn(
                &decl,
                fn_decl_span,
                generator_kind,
                movability,
            );
            let capture_clause = this.lower_capture_clause(capture_clause);
            this.current_item = prev;
            hir::ExprKind::Closure(
                capture_clause,
                fn_decl,
                body_id,
                fn_decl_span,
                generator_option,
            )
        })
    }

    fn lower_capture_clause(&mut self, c: CaptureBy) -> hir::CaptureClause {
        match c {
            CaptureBy::Value => hir::CaptureByValue,
            CaptureBy::Ref => hir::CaptureByRef,
        }
    }

    fn generator_movability_for_fn(
        &mut self,
        decl: &FnDecl,
        fn_decl_span: Span,
        generator_kind: Option<hir::GeneratorKind>,
        movability: Movability,
    ) -> Option<hir::GeneratorMovability> {
        match generator_kind {
            Some(hir::GeneratorKind::Gen) =>  {
                if !decl.inputs.is_empty() {
                    span_err!(
                        self.sess,
                        fn_decl_span,
                        E0628,
                        "generators cannot have explicit arguments"
                    );
                    self.sess.abort_if_errors();
                }
                Some(match movability {
                    Movability::Movable => hir::GeneratorMovability::Movable,
                    Movability::Static => hir::GeneratorMovability::Static,
                })
            },
            Some(hir::GeneratorKind::Async) => {
                bug!("non-`async` closure body turned `async` during lowering");
            },
            None => {
                if movability == Movability::Static {
                    span_err!(
                        self.sess,
                        fn_decl_span,
                        E0697,
                        "closures cannot be static"
                    );
                }
                None
            },
        }
    }

    fn lower_expr_async_closure(
        &mut self,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
    ) -> hir::ExprKind {
        let outer_decl = FnDecl {
            inputs: decl.inputs.clone(),
            output: FunctionRetTy::Default(fn_decl_span),
            c_variadic: false,
        };
        // We need to lower the declaration outside the new scope, because we
        // have to conserve the state of being inside a loop condition for the
        // closure argument types.
        let fn_decl = self.lower_fn_decl(&outer_decl, None, false, None);

        self.with_new_scopes(|this| {
            // FIXME(cramertj): allow `async` non-`move` closures with arguments.
            if capture_clause == CaptureBy::Ref && !decl.inputs.is_empty() {
                struct_span_err!(
                    this.sess,
                    fn_decl_span,
                    E0708,
                    "`async` non-`move` closures with arguments are not currently supported",
                )
                .help(
                    "consider using `let` statements to manually capture \
                    variables by reference before entering an `async move` closure"
                )
                .emit();
            }

            // Transform `async |x: u8| -> X { ... }` into
            // `|x: u8| future_from_generator(|| -> X { ... })`.
            let body_id = this.lower_fn_body(&outer_decl, |this| {
                let async_ret_ty = if let FunctionRetTy::Ty(ty) = &decl.output {
                    Some(ty.clone())
                } else {
                    None
                };
                let async_body = this.make_async_expr(
                    capture_clause, closure_id, async_ret_ty, body.span,
                    |this| {
                        this.with_new_scopes(|this| this.lower_expr(body))
                    }
                );
                this.expr(fn_decl_span, async_body, ThinVec::new())
            });
            hir::ExprKind::Closure(
                this.lower_capture_clause(capture_clause),
                fn_decl,
                body_id,
                fn_decl_span,
                None,
            )
        })
    }

    /// Desugar `<start>..=<end>` into `std::ops::RangeInclusive::new(<start>, <end>)`.
    fn lower_expr_range_closed(&mut self, span: Span, e1: &Expr, e2: &Expr) -> hir::ExprKind {
        let id = self.next_id();
        let e1 = self.lower_expr(e1);
        let e2 = self.lower_expr(e2);
        self.expr_call_std_assoc_fn(
            id,
            span,
            &[sym::ops, sym::RangeInclusive],
            "new",
            hir_vec![e1, e2],
        )
    }

    fn lower_expr_range(
        &mut self,
        span: Span,
        e1: Option<&Expr>,
        e2: Option<&Expr>,
        lims: RangeLimits,
    ) -> hir::ExprKind {
        use syntax::ast::RangeLimits::*;

        let path = match (e1, e2, lims) {
            (None, None, HalfOpen) => sym::RangeFull,
            (Some(..), None, HalfOpen) => sym::RangeFrom,
            (None, Some(..), HalfOpen) => sym::RangeTo,
            (Some(..), Some(..), HalfOpen) => sym::Range,
            (None, Some(..), Closed) => sym::RangeToInclusive,
            (Some(..), Some(..), Closed) => unreachable!(),
            (_, None, Closed) => self.diagnostic()
                .span_fatal(span, "inclusive range with no end")
                .raise(),
        };

        let fields = e1.iter()
            .map(|e| ("start", e))
            .chain(e2.iter().map(|e| ("end", e)))
            .map(|(s, e)| {
                let expr = P(self.lower_expr(&e));
                let ident = Ident::new(Symbol::intern(s), e.span);
                self.field(ident, expr, e.span)
            })
            .collect::<P<[hir::Field]>>();

        let is_unit = fields.is_empty();
        let struct_path = [sym::ops, path];
        let struct_path = self.std_path(span, &struct_path, None, is_unit);
        let struct_path = hir::QPath::Resolved(None, P(struct_path));

        if is_unit {
            hir::ExprKind::Path(struct_path)
        } else {
            hir::ExprKind::Struct(P(struct_path), fields, None)
        }
    }

    fn lower_label(&mut self, label: Option<Label>) -> Option<hir::Label> {
        label.map(|label| hir::Label {
            ident: label.ident,
        })
    }

    fn lower_loop_destination(&mut self, destination: Option<(NodeId, Label)>) -> hir::Destination {
        let target_id = match destination {
            Some((id, _)) => {
                if let Some(loop_id) = self.resolver.get_label_res(id) {
                    Ok(self.lower_node_id(loop_id))
                } else {
                    Err(hir::LoopIdError::UnresolvedLabel)
                }
            }
            None => {
                self.loop_scopes
                    .last()
                    .cloned()
                    .map(|id| Ok(self.lower_node_id(id)))
                    .unwrap_or(Err(hir::LoopIdError::OutsideLoopScope))
                    .into()
            }
        };
        hir::Destination {
            label: self.lower_label(destination.map(|(_, label)| label)),
            target_id,
        }
    }

    fn lower_jump_destination(&mut self, id: NodeId, opt_label: Option<Label>) -> hir::Destination {
        if self.is_in_loop_condition && opt_label.is_none() {
            hir::Destination {
                label: None,
                target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition).into(),
            }
        } else {
            self.lower_loop_destination(opt_label.map(|label| (id, label)))
        }
    }

    fn with_catch_scope<T, F>(&mut self, catch_id: NodeId, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let len = self.catch_scopes.len();
        self.catch_scopes.push(catch_id);

        let result = f(self);
        assert_eq!(
            len + 1,
            self.catch_scopes.len(),
            "catch scopes should be added and removed in stack order"
        );

        self.catch_scopes.pop().unwrap();

        result
    }

    fn with_loop_scope<T, F>(&mut self, loop_id: NodeId, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        // We're no longer in the base loop's condition; we're in another loop.
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let len = self.loop_scopes.len();
        self.loop_scopes.push(loop_id);

        let result = f(self);
        assert_eq!(
            len + 1,
            self.loop_scopes.len(),
            "loop scopes should be added and removed in stack order"
        );

        self.loop_scopes.pop().unwrap();

        self.is_in_loop_condition = was_in_loop_condition;

        result
    }

    fn with_loop_condition_scope<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = true;

        let result = f(self);

        self.is_in_loop_condition = was_in_loop_condition;

        result
    }

    fn lower_expr_asm(&mut self, asm: &InlineAsm) -> hir::ExprKind {
        let hir_asm = hir::InlineAsm {
            inputs: asm.inputs.iter().map(|&(ref c, _)| c.clone()).collect(),
            outputs: asm.outputs
                .iter()
                .map(|out| hir::InlineAsmOutput {
                    constraint: out.constraint.clone(),
                    is_rw: out.is_rw,
                    is_indirect: out.is_indirect,
                    span: out.expr.span,
                })
                .collect(),
            asm: asm.asm.clone(),
            asm_str_style: asm.asm_str_style,
            clobbers: asm.clobbers.clone().into(),
            volatile: asm.volatile,
            alignstack: asm.alignstack,
            dialect: asm.dialect,
            ctxt: asm.ctxt,
        };

        let outputs = asm.outputs
            .iter()
            .map(|out| self.lower_expr(&out.expr))
            .collect();

        let inputs = asm.inputs
            .iter()
            .map(|&(_, ref input)| self.lower_expr(input))
            .collect();

        hir::ExprKind::InlineAsm(P(hir_asm), outputs, inputs)
    }

    fn lower_field(&mut self, f: &Field) -> hir::Field {
        hir::Field {
            hir_id: self.next_id(),
            ident: f.ident,
            expr: P(self.lower_expr(&f.expr)),
            span: f.span,
            is_shorthand: f.is_shorthand,
        }
    }

    fn lower_expr_yield(&mut self, span: Span, opt_expr: Option<&Expr>) -> hir::ExprKind {
        match self.generator_kind {
            Some(hir::GeneratorKind::Gen) => {},
            Some(hir::GeneratorKind::Async) => {
                span_err!(
                    self.sess,
                    span,
                    E0727,
                    "`async` generators are not yet supported",
                );
                self.sess.abort_if_errors();
            },
            None => self.generator_kind = Some(hir::GeneratorKind::Gen),
        }

        let expr = opt_expr
            .as_ref()
            .map(|x| self.lower_expr(x))
            .unwrap_or_else(|| self.expr_unit(span));

        hir::ExprKind::Yield(P(expr), hir::YieldSource::Yield)
    }

    /// Desugar `ExprForLoop` from: `[opt_ident]: for <pat> in <head> <body>` into:
    /// ```rust
    /// {
    ///     let result = match ::std::iter::IntoIterator::into_iter(<head>) {
    ///         mut iter => {
    ///             [opt_ident]: loop {
    ///                 let mut __next;
    ///                 match ::std::iter::Iterator::next(&mut iter) {
    ///                     ::std::option::Option::Some(val) => __next = val,
    ///                     ::std::option::Option::None => break
    ///                 };
    ///                 let <pat> = __next;
    ///                 StmtKind::Expr(<body>);
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
    ) -> hir::Expr {
        // expand <head>
        let mut head = self.lower_expr(head);
        let head_sp = head.span;
        let desugared_span = self.mark_span_with_reason(
            DesugaringKind::ForLoop,
            head_sp,
            None,
        );
        head.span = desugared_span;

        let iter = Ident::with_empty_ctxt(sym::iter);

        let next_ident = Ident::with_empty_ctxt(sym::__next);
        let (next_pat, next_pat_hid) = self.pat_ident_binding_mode(
            desugared_span,
            next_ident,
            hir::BindingAnnotation::Mutable,
        );

        // `::std::option::Option::Some(val) => __next = val`
        let pat_arm = {
            let val_ident = Ident::with_empty_ctxt(sym::val);
            let (val_pat, val_pat_hid) = self.pat_ident(pat.span, val_ident);
            let val_expr = P(self.expr_ident(pat.span, val_ident, val_pat_hid));
            let next_expr = P(self.expr_ident(pat.span, next_ident, next_pat_hid));
            let assign = P(self.expr(
                pat.span,
                hir::ExprKind::Assign(next_expr, val_expr),
                ThinVec::new(),
            ));
            let some_pat = self.pat_some(pat.span, val_pat);
            self.arm(hir_vec![some_pat], assign)
        };

        // `::std::option::Option::None => break`
        let break_arm = {
            let break_expr =
                self.with_loop_scope(e.id, |this| this.expr_break(e.span, ThinVec::new()));
            let pat = self.pat_none(e.span);
            self.arm(hir_vec![pat], break_expr)
        };

        // `mut iter`
        let (iter_pat, iter_pat_nid) = self.pat_ident_binding_mode(
            desugared_span,
            iter,
            hir::BindingAnnotation::Mutable
        );

        // `match ::std::iter::Iterator::next(&mut iter) { ... }`
        let match_expr = {
            let iter = P(self.expr_ident(head_sp, iter, iter_pat_nid));
            let ref_mut_iter = self.expr_mut_addr_of(head_sp, iter);
            let next_path = &[sym::iter, sym::Iterator, sym::next];
            let next_expr = P(self.expr_call_std_path(
                head_sp,
                next_path,
                hir_vec![ref_mut_iter],
            ));
            let arms = hir_vec![pat_arm, break_arm];

            self.expr_match(head_sp, next_expr, arms, hir::MatchSource::ForLoopDesugar)
        };
        let match_stmt = self.stmt_expr(head_sp, match_expr);

        let next_expr = P(self.expr_ident(head_sp, next_ident, next_pat_hid));

        // `let mut __next`
        let next_let = self.stmt_let_pat(
            ThinVec::new(),
            desugared_span,
            None,
            next_pat,
            hir::LocalSource::ForLoopDesugar,
        );

        // `let <pat> = __next`
        let pat = self.lower_pat(pat);
        let pat_let = self.stmt_let_pat(
            ThinVec::new(),
            head_sp,
            Some(next_expr),
            pat,
            hir::LocalSource::ForLoopDesugar,
        );

        let body_block = self.with_loop_scope(e.id, |this| this.lower_block(body, false));
        let body_expr = self.expr_block(body_block, ThinVec::new());
        let body_stmt = self.stmt_expr(body.span, body_expr);

        let loop_block = P(self.block_all(
            e.span,
            hir_vec![next_let, match_stmt, pat_let, body_stmt],
            None,
        ));

        // `[opt_ident]: loop { ... }`
        let loop_expr = hir::ExprKind::Loop(
            loop_block,
            self.lower_label(opt_label),
            hir::LoopSource::ForLoop,
        );
        let loop_expr = P(hir::Expr {
            hir_id: self.lower_node_id(e.id),
            node: loop_expr,
            span: e.span,
            attrs: ThinVec::new(),
        });

        // `mut iter => { ... }`
        let iter_arm = self.arm(hir_vec![iter_pat], loop_expr);

        // `match ::std::iter::IntoIterator::into_iter(<head>) { ... }`
        let into_iter_expr = {
            let into_iter_path =
                &[sym::iter, sym::IntoIterator, sym::into_iter];
            P(self.expr_call_std_path(
                head_sp,
                into_iter_path,
                hir_vec![head],
            ))
        };

        let match_expr = P(self.expr_match(
            head_sp,
            into_iter_expr,
            hir_vec![iter_arm],
            hir::MatchSource::ForLoopDesugar,
        ));

        // This is effectively `{ let _result = ...; _result }`.
        // The construct was introduced in #21984 and is necessary to make sure that
        // temporaries in the `head` expression are dropped and do not leak to the
        // surrounding scope of the `match` since the `match` is not a terminating scope.
        //
        // Also, add the attributes to the outer returned expr node.
        self.expr_drop_temps(head_sp, match_expr, e.attrs.clone())
    }

    /// Desugar `ExprKind::Try` from: `<expr>?` into:
    /// ```rust
    /// match Try::into_result(<expr>) {
    ///     Ok(val) => #[allow(unreachable_code)] val,
    ///     Err(err) => #[allow(unreachable_code)]
    ///                 // If there is an enclosing `try {...}`:
    ///                 break 'catch_target Try::from_error(From::from(err)),
    ///                 // Otherwise:
    ///                 return Try::from_error(From::from(err)),
    /// }
    /// ```
    fn lower_expr_try(&mut self, span: Span, sub_expr: &Expr) -> hir::ExprKind {
        let unstable_span = self.mark_span_with_reason(
            DesugaringKind::QuestionMark,
            span,
            self.allow_try_trait.clone(),
        );
        let try_span = self.sess.source_map().end_point(span);
        let try_span = self.mark_span_with_reason(
            DesugaringKind::QuestionMark,
            try_span,
            self.allow_try_trait.clone(),
        );

        // `Try::into_result(<expr>)`
        let scrutinee = {
            // expand <expr>
            let sub_expr = self.lower_expr(sub_expr);

            let path = &[sym::ops, sym::Try, sym::into_result];
            P(self.expr_call_std_path(unstable_span, path, hir_vec![sub_expr]))
        };

        // `#[allow(unreachable_code)]`
        let attr = {
            // `allow(unreachable_code)`
            let allow = {
                let allow_ident = Ident::new(sym::allow, span);
                let uc_ident = Ident::new(sym::unreachable_code, span);
                let uc_nested = attr::mk_nested_word_item(uc_ident);
                attr::mk_list_item(allow_ident, vec![uc_nested])
            };
            attr::mk_attr_outer(allow)
        };
        let attrs = vec![attr];

        // `Ok(val) => #[allow(unreachable_code)] val,`
        let ok_arm = {
            let val_ident = Ident::with_empty_ctxt(sym::val);
            let (val_pat, val_pat_nid) = self.pat_ident(span, val_ident);
            let val_expr = P(self.expr_ident_with_attrs(
                span,
                val_ident,
                val_pat_nid,
                ThinVec::from(attrs.clone()),
            ));
            let ok_pat = self.pat_ok(span, val_pat);

            self.arm(hir_vec![ok_pat], val_expr)
        };

        // `Err(err) => #[allow(unreachable_code)]
        //              return Try::from_error(From::from(err)),`
        let err_arm = {
            let err_ident = Ident::with_empty_ctxt(sym::err);
            let (err_local, err_local_nid) = self.pat_ident(try_span, err_ident);
            let from_expr = {
                let from_path = &[sym::convert, sym::From, sym::from];
                let err_expr = self.expr_ident(try_span, err_ident, err_local_nid);
                self.expr_call_std_path(try_span, from_path, hir_vec![err_expr])
            };
            let from_err_expr =
                self.wrap_in_try_constructor(sym::from_error, from_expr, unstable_span);
            let thin_attrs = ThinVec::from(attrs);
            let catch_scope = self.catch_scopes.last().map(|x| *x);
            let ret_expr = if let Some(catch_node) = catch_scope {
                let target_id = Ok(self.lower_node_id(catch_node));
                P(self.expr(
                    try_span,
                    hir::ExprKind::Break(
                        hir::Destination {
                            label: None,
                            target_id,
                        },
                        Some(from_err_expr),
                    ),
                    thin_attrs,
                ))
            } else {
                P(self.expr(try_span, hir::ExprKind::Ret(Some(from_err_expr)), thin_attrs))
            };

            let err_pat = self.pat_err(try_span, err_local);
            self.arm(hir_vec![err_pat], ret_expr)
        };

        hir::ExprKind::Match(
            scrutinee,
            hir_vec![err_arm, ok_arm],
            hir::MatchSource::TryDesugar,
        )
    }

    // =========================================================================
    // Helper methods for building HIR.
    // =========================================================================

    /// Constructs a `true` or `false` literal expression.
    pub(super) fn expr_bool(&mut self, span: Span, val: bool) -> hir::Expr {
        let lit = Spanned { span, node: LitKind::Bool(val) };
        self.expr(span, hir::ExprKind::Lit(lit), ThinVec::new())
    }

    /// Wrap the given `expr` in a terminating scope using `hir::ExprKind::DropTemps`.
    ///
    /// In terms of drop order, it has the same effect as wrapping `expr` in
    /// `{ let _t = $expr; _t }` but should provide better compile-time performance.
    ///
    /// The drop order can be important in e.g. `if expr { .. }`.
    fn expr_drop_temps(
        &mut self,
        span: Span,
        expr: P<hir::Expr>,
        attrs: ThinVec<Attribute>
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::DropTemps(expr), attrs)
    }

    fn expr_match(
        &mut self,
        span: Span,
        arg: P<hir::Expr>,
        arms: hir::HirVec<hir::Arm>,
        source: hir::MatchSource,
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::Match(arg, arms, source), ThinVec::new())
    }

    fn expr_break(&mut self, span: Span, attrs: ThinVec<Attribute>) -> P<hir::Expr> {
        let expr_break = hir::ExprKind::Break(self.lower_loop_destination(None), None);
        P(self.expr(span, expr_break, attrs))
    }

    fn expr_mut_addr_of(&mut self, span: Span, e: P<hir::Expr>) -> hir::Expr {
        self.expr(span, hir::ExprKind::AddrOf(hir::MutMutable, e), ThinVec::new())
    }

    fn expr_unit(&mut self, sp: Span) -> hir::Expr {
        self.expr_tuple(sp, hir_vec![])
    }

    fn expr_tuple(&mut self, sp: Span, exprs: hir::HirVec<hir::Expr>) -> hir::Expr {
        self.expr(sp, hir::ExprKind::Tup(exprs), ThinVec::new())
    }

    fn expr_call(
        &mut self,
        span: Span,
        e: P<hir::Expr>,
        args: hir::HirVec<hir::Expr>,
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::Call(e, args), ThinVec::new())
    }

    // Note: associated functions must use `expr_call_std_path`.
    fn expr_call_std_path(
        &mut self,
        span: Span,
        path_components: &[Symbol],
        args: hir::HirVec<hir::Expr>,
    ) -> hir::Expr {
        let path = P(self.expr_std_path(span, path_components, None, ThinVec::new()));
        self.expr_call(span, path, args)
    }

    // Create an expression calling an associated function of an std type.
    //
    // Associated functions cannot be resolved through the normal `std_path` function,
    // as they are resolved differently and so cannot use `expr_call_std_path`.
    //
    // This function accepts the path component (`ty_path_components`) separately from
    // the name of the associated function (`assoc_fn_name`) in order to facilitate
    // separate resolution of the type and creation of a path referring to its associated
    // function.
    fn expr_call_std_assoc_fn(
        &mut self,
        ty_path_id: hir::HirId,
        span: Span,
        ty_path_components: &[Symbol],
        assoc_fn_name: &str,
        args: hir::HirVec<hir::Expr>,
    ) -> hir::ExprKind {
        let ty_path = P(self.std_path(span, ty_path_components, None, false));
        let ty = P(self.ty_path(ty_path_id, span, hir::QPath::Resolved(None, ty_path)));
        let fn_seg = P(hir::PathSegment::from_ident(Ident::from_str(assoc_fn_name)));
        let fn_path = hir::QPath::TypeRelative(ty, fn_seg);
        let fn_expr = P(self.expr(span, hir::ExprKind::Path(fn_path), ThinVec::new()));
        hir::ExprKind::Call(fn_expr, args)
    }

    fn expr_std_path(
        &mut self,
        span: Span,
        components: &[Symbol],
        params: Option<P<hir::GenericArgs>>,
        attrs: ThinVec<Attribute>,
    ) -> hir::Expr {
        let path = self.std_path(span, components, params, true);
        self.expr(
            span,
            hir::ExprKind::Path(hir::QPath::Resolved(None, P(path))),
            attrs,
        )
    }

    pub(super) fn expr_ident(&mut self, sp: Span, ident: Ident, binding: hir::HirId) -> hir::Expr {
        self.expr_ident_with_attrs(sp, ident, binding, ThinVec::new())
    }

    fn expr_ident_with_attrs(
        &mut self,
        span: Span,
        ident: Ident,
        binding: hir::HirId,
        attrs: ThinVec<Attribute>,
    ) -> hir::Expr {
        let expr_path = hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            P(hir::Path {
                span,
                res: Res::Local(binding),
                segments: hir_vec![hir::PathSegment::from_ident(ident)],
            }),
        ));

        self.expr(span, expr_path, attrs)
    }

    fn expr_unsafe(&mut self, expr: P<hir::Expr>) -> hir::Expr {
        let hir_id = self.next_id();
        let span = expr.span;
        self.expr(
            span,
            hir::ExprKind::Block(P(hir::Block {
                stmts: hir_vec![],
                expr: Some(expr),
                hir_id,
                rules: hir::UnsafeBlock(hir::CompilerGenerated),
                span,
                targeted_by_break: false,
            }), None),
            ThinVec::new(),
        )
    }

    fn expr_block_empty(&mut self, span: Span) -> hir::Expr {
        let blk = self.block_all(span, hir_vec![], None);
        self.expr_block(P(blk), ThinVec::new())
    }

    pub(super) fn expr_block(&mut self, b: P<hir::Block>, attrs: ThinVec<Attribute>) -> hir::Expr {
        self.expr(b.span, hir::ExprKind::Block(b, None), attrs)
    }

    pub(super) fn expr(
        &mut self,
        span: Span,
        node: hir::ExprKind,
        attrs: ThinVec<Attribute>
    ) -> hir::Expr {
        hir::Expr {
            hir_id: self.next_id(),
            node,
            span,
            attrs,
        }
    }

    fn field(&mut self, ident: Ident, expr: P<hir::Expr>, span: Span) -> hir::Field {
        hir::Field {
            hir_id: self.next_id(),
            ident,
            span,
            expr,
            is_shorthand: false,
        }
    }

    fn arm(&mut self, pats: hir::HirVec<P<hir::Pat>>, expr: P<hir::Expr>) -> hir::Arm {
        hir::Arm {
            hir_id: self.next_id(),
            attrs: hir_vec![],
            pats,
            guard: None,
            span: expr.span,
            body: expr,
        }
    }
}
