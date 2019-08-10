use super::{LoweringContext, ParamMode, ParenthesizedGenericArgs, ImplTraitContext};
use crate::hir::{self, HirVec};
use crate::hir::ptr::P;

use rustc_data_structures::thin_vec::ThinVec;

use syntax::attr;
use syntax::ptr::P as AstP;
use syntax::ast::*;
use syntax::source_map::{respan, DesugaringKind, Span};
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
            ExprKind::Let(ref pats, ref scrutinee) => {
                // If we got here, the `let` expression is not allowed.
                self.sess
                    .struct_span_err(e.span, "`let` expressions are not supported here")
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
                    let expr = self.expr_bool(e.span, true);
                    self.arm(pats, P(expr))
                };
                let else_arm = {
                    let pats = hir_vec![self.pat_wild(e.span)];
                    let expr = self.expr_bool(e.span, false);
                    self.arm(pats, P(expr))
                };
                hir::ExprKind::Match(
                    P(scrutinee),
                    vec![then_arm, else_arm].into(),
                    hir::MatchSource::Normal,
                )
            }
            // FIXME(#53667): handle lowering of && and parens.
            ExprKind::If(ref cond, ref then, ref else_opt) => {
                // `_ => else_block` where `else_block` is `{}` if there's `None`:
                let else_pat = self.pat_wild(e.span);
                let (else_expr, contains_else_clause) = match else_opt {
                    None => (self.expr_block_empty(e.span), false),
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
                            DesugaringKind::CondTemporary, cond.span, None
                        );
                        // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                        // to preserve drop semantics since `if cond { ... }` does not
                        // let temporaries live outside of `cond`.
                        let cond = self.expr_drop_temps(span_block, P(cond), ThinVec::new());

                        let desugar = hir::MatchSource::IfDesugar { contains_else_clause };
                        let pats = hir_vec![self.pat_bool(e.span, true)];
                        (pats, cond, desugar)
                    }
                };
                let then_arm = self.arm(then_pats, P(then_expr));

                hir::ExprKind::Match(P(scrutinee), vec![then_arm, else_arm].into(), desugar)
            }
            // FIXME(#53667): handle lowering of && and parens.
            ExprKind::While(ref cond, ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                // Note that the block AND the condition are evaluated in the loop scope.
                // This is done to allow `break` from inside the condition of the loop.

                // `_ => break`:
                let else_arm = {
                    let else_pat = this.pat_wild(e.span);
                    let else_expr = this.expr_break(e.span, ThinVec::new());
                    this.arm(hir_vec![else_pat], else_expr)
                };

                // Handle then + scrutinee:
                let then_blk = this.lower_block(body, false);
                let then_expr = this.expr_block(then_blk, ThinVec::new());
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
                        let scrutinee = this.with_loop_condition_scope(|t| t.lower_expr(scrutinee));
                        let pats = pats.iter().map(|pat| this.lower_pat(pat)).collect();
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
                        let cond = this.with_loop_condition_scope(|this| this.lower_expr(cond));
                        let span_block = this.mark_span_with_reason(
                            DesugaringKind::CondTemporary, cond.span, None
                        );
                        // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                        // to preserve drop semantics since `while cond { ... }` does not
                        // let temporaries live outside of `cond`.
                        let cond = this.expr_drop_temps(span_block, P(cond), ThinVec::new());

                        let desugar = hir::MatchSource::WhileDesugar;
                        // `true => <then>`:
                        let pats = hir_vec![this.pat_bool(e.span, true)];
                        (pats, cond, desugar, hir::LoopSource::While)
                    }
                };
                let then_arm = this.arm(then_pats, P(then_expr));

                // `match <scrutinee> { ... }`
                let match_expr = this.expr_match(
                    scrutinee.span,
                    P(scrutinee),
                    hir_vec![then_arm, else_arm],
                    desugar,
                );

                // `[opt_ident]: loop { ... }`
                hir::ExprKind::Loop(
                    P(this.block_expr(P(match_expr))),
                    this.lower_label(opt_label),
                    source
                )
            }),
            ExprKind::Loop(ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                hir::ExprKind::Loop(
                    this.lower_block(body, false),
                    this.lower_label(opt_label),
                    hir::LoopSource::Loop,
                )
            }),
            ExprKind::TryBlock(ref body) => {
                self.with_catch_scope(body.id, |this| {
                    let unstable_span = this.mark_span_with_reason(
                        DesugaringKind::TryBlock,
                        body.span,
                        this.allow_try_trait.clone(),
                    );
                    let mut block = this.lower_block(body, true).into_inner();
                    let tail = block.expr.take().map_or_else(
                        || {
                            let span = this.sess.source_map().end_point(unstable_span);
                            hir::Expr {
                                span,
                                node: hir::ExprKind::Tup(hir_vec![]),
                                attrs: ThinVec::new(),
                                hir_id: this.next_id(),
                            }
                        },
                        |x: P<hir::Expr>| x.into_inner(),
                    );
                    block.expr = Some(this.wrap_in_try_constructor(
                        sym::from_ok, tail, unstable_span));
                    hir::ExprKind::Block(P(block), None)
                })
            }
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
            ExprKind::Await(ref expr) => self.lower_await(e.span, expr),
            ExprKind::Closure(
                capture_clause, asyncness, movability, ref decl, ref body, fn_decl_span
            ) => {
                if let IsAsync::Async { closure_id, .. } = asyncness {
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
                        if capture_clause == CaptureBy::Ref &&
                            !decl.inputs.is_empty()
                        {
                            struct_span_err!(
                                this.sess,
                                fn_decl_span,
                                E0708,
                                "`async` non-`move` closures with arguments \
                                are not currently supported",
                            )
                                .help("consider using `let` statements to manually capture \
                                       variables by reference before entering an \
                                       `async move` closure")
                                .emit();
                        }

                        // Transform `async |x: u8| -> X { ... }` into
                        // `|x: u8| future_from_generator(|| -> X { ... })`.
                        let body_id = this.lower_fn_body(&outer_decl, |this| {
                            let async_ret_ty = if let FunctionRetTy::Ty(ty) = &decl.output {
                                Some(ty.clone())
                            } else { None };
                            let async_body = this.make_async_expr(
                                capture_clause, closure_id, async_ret_ty, body.span,
                                |this| {
                                    this.with_new_scopes(|this| this.lower_expr(body))
                                });
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
                } else {
                    // Lower outside new scope to preserve `is_in_loop_condition`.
                    let fn_decl = self.lower_fn_decl(decl, None, false, None);

                    self.with_new_scopes(|this| {
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
                        hir::ExprKind::Closure(
                            this.lower_capture_clause(capture_clause),
                            fn_decl,
                            body_id,
                            fn_decl_span,
                            generator_option,
                        )
                    })
                }
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
            // Desugar `<start>..=<end>` into `std::ops::RangeInclusive::new(<start>, <end>)`.
            ExprKind::Range(Some(ref e1), Some(ref e2), RangeLimits::Closed) => {
                let id = self.next_id();
                let e1 = self.lower_expr(e1);
                let e2 = self.lower_expr(e2);
                self.expr_call_std_assoc_fn(
                    id,
                    e.span,
                    &[sym::ops, sym::RangeInclusive],
                    "new",
                    hir_vec![e1, e2],
                )
            }
            ExprKind::Range(ref e1, ref e2, lims) => {
                use syntax::ast::RangeLimits::*;

                let path = match (e1, e2, lims) {
                    (&None, &None, HalfOpen) => sym::RangeFull,
                    (&Some(..), &None, HalfOpen) => sym::RangeFrom,
                    (&None, &Some(..), HalfOpen) => sym::RangeTo,
                    (&Some(..), &Some(..), HalfOpen) => sym::Range,
                    (&None, &Some(..), Closed) => sym::RangeToInclusive,
                    (&Some(..), &Some(..), Closed) => unreachable!(),
                    (_, &None, Closed) => self.diagnostic()
                        .span_fatal(e.span, "inclusive range with no end")
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
                let struct_path = self.std_path(e.span, &struct_path, None, is_unit);
                let struct_path = hir::QPath::Resolved(None, P(struct_path));

                return hir::Expr {
                    hir_id: self.lower_node_id(e.id),
                    node: if is_unit {
                        hir::ExprKind::Path(struct_path)
                    } else {
                        hir::ExprKind::Struct(P(struct_path), fields, None)
                    },
                    span: e.span,
                    attrs: e.attrs.clone(),
                };
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
                let destination = if self.is_in_loop_condition && opt_label.is_none() {
                    hir::Destination {
                        label: None,
                        target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition).into(),
                    }
                } else {
                    self.lower_loop_destination(opt_label.map(|label| (e.id, label)))
                };
                hir::ExprKind::Break(
                    destination,
                    opt_expr.as_ref().map(|x| P(self.lower_expr(x))),
                )
            }
            ExprKind::Continue(opt_label) => {
                hir::ExprKind::Continue(if self.is_in_loop_condition && opt_label.is_none() {
                    hir::Destination {
                        label: None,
                        target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition).into(),
                    }
                } else {
                    self.lower_loop_destination(opt_label.map(|label| (e.id, label)))
                })
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
}
