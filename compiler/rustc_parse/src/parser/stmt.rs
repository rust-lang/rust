use std::borrow::Cow;
use std::mem;
use std::ops::Bound;

use ast::Label;
use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, InvisibleOrigin, MetaVarKind, TokenKind};
use rustc_ast::util::classify::{self, TrailingBrace};
use rustc_ast::{
    AttrStyle, AttrVec, Block, BlockCheckMode, DUMMY_NODE_ID, Expr, ExprKind, HasAttrs, Local,
    LocalKind, MacCall, MacCallStmt, MacStmtStyle, Recovered, Stmt, StmtKind,
};
use rustc_errors::{Applicability, Diag, PResult};
use rustc_span::{BytePos, ErrorGuaranteed, Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use super::attr::InnerAttrForbiddenReason;
use super::diagnostics::AttemptLocalParseRecovery;
use super::pat::{PatternLocation, RecoverComma};
use super::path::PathStyle;
use super::{
    AttrWrapper, BlockMode, FnParseMode, ForceCollect, Parser, Restrictions, SemiColonMode,
    Trailing, UsePreAttrPos,
};
use crate::errors::{self, MalformedLoopLabel};
use crate::exp;

impl<'a> Parser<'a> {
    /// Parses a statement. This stops just before trailing semicolons on everything but items.
    /// e.g., a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    ///
    /// If `force_collect` is [`ForceCollect::Yes`], forces collection of tokens regardless of
    /// whether or not we have attributes.
    // Public for rustfmt usage.
    pub fn parse_stmt(&mut self, force_collect: ForceCollect) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_without_recovery(false, force_collect, false).unwrap_or_else(|e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        }))
    }

    /// If `force_collect` is [`ForceCollect::Yes`], forces collection of tokens regardless of
    /// whether or not we have attributes. If `force_full_expr` is true, parses the stmt without
    /// using `Restriction::STMT_EXPR`. Public for `cfg_eval` macro expansion.
    pub fn parse_stmt_without_recovery(
        &mut self,
        capture_semi: bool,
        force_collect: ForceCollect,
        force_full_expr: bool,
    ) -> PResult<'a, Option<Stmt>> {
        let pre_attr_pos = self.collect_pos();
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        if let Some(stmt) = self.eat_metavar_seq(MetaVarKind::Stmt, |this| {
            this.parse_stmt_without_recovery(false, ForceCollect::Yes, false)
        }) {
            let mut stmt = stmt.expect("an actual statement");
            stmt.visit_attrs(|stmt_attrs| {
                attrs.prepend_to_nt_inner(stmt_attrs);
            });
            return Ok(Some(stmt));
        }

        if self.token.is_keyword(kw::Mut) && self.is_keyword_ahead(1, &[kw::Let]) {
            self.bump();
            let mut_let_span = lo.to(self.token.span);
            self.dcx().emit_err(errors::InvalidVariableDeclaration {
                span: mut_let_span,
                sub: errors::InvalidVariableDeclarationSub::SwitchMutLetOrder(mut_let_span),
            });
        }

        let stmt = if self.token.is_keyword(kw::Super) && self.is_keyword_ahead(1, &[kw::Let]) {
            self.collect_tokens(None, attrs, force_collect, |this, attrs| {
                let super_span = this.token.span;
                this.expect_keyword(exp!(Super))?;
                this.expect_keyword(exp!(Let))?;
                this.psess.gated_spans.gate(sym::super_let, super_span);
                let local = this.parse_local(Some(super_span), attrs)?;
                let trailing = Trailing::from(capture_semi && this.token == token::Semi);
                Ok((
                    this.mk_stmt(lo.to(this.prev_token.span), StmtKind::Let(local)),
                    trailing,
                    UsePreAttrPos::No,
                ))
            })?
        } else if self.token.is_keyword(kw::Let) {
            self.collect_tokens(None, attrs, force_collect, |this, attrs| {
                this.expect_keyword(exp!(Let))?;
                let local = this.parse_local(None, attrs)?;
                let trailing = Trailing::from(capture_semi && this.token == token::Semi);
                Ok((
                    this.mk_stmt(lo.to(this.prev_token.span), StmtKind::Let(local)),
                    trailing,
                    UsePreAttrPos::No,
                ))
            })?
        } else if self.is_kw_followed_by_ident(kw::Mut) && self.may_recover() {
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::MissingLet,
                force_collect,
            )?
        } else if self.is_kw_followed_by_ident(kw::Auto) && self.may_recover() {
            self.bump(); // `auto`
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::UseLetNotAuto,
                force_collect,
            )?
        } else if self.is_kw_followed_by_ident(sym::var) && self.may_recover() {
            self.bump(); // `var`
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::UseLetNotVar,
                force_collect,
            )?
        } else if self.check_path()
            && !self.token.is_qpath_start()
            && !self.is_path_start_item()
            && !self.is_builtin()
        {
            // We have avoided contextual keywords like `union`, items with `crate` visibility,
            // or `auto trait` items. We aim to parse an arbitrary path `a::b` but not something
            // that starts like a path (1 token), but it fact not a path.
            // Also, we avoid stealing syntax from `parse_item_`.
            //
            // `UsePreAttrPos::Yes` here means the attribute belongs unconditionally to the
            // expression, not the statement. (But the statement attributes/tokens are obtained
            // from the expression anyway, because `Stmt` delegates `HasAttrs`/`HasTokens` to
            // the things within `StmtKind`.)
            let stmt = self.collect_tokens(
                Some(pre_attr_pos),
                AttrWrapper::empty(),
                force_collect,
                |this, _empty_attrs| {
                    Ok((this.parse_stmt_path_start(lo, attrs)?, Trailing::No, UsePreAttrPos::Yes))
                },
            );
            match stmt {
                Ok(stmt) => stmt,
                Err(mut err) => {
                    self.suggest_add_missing_let_for_stmt(&mut err);
                    return Err(err);
                }
            }
        } else if let Some(item) = self.parse_item_common(
            attrs.clone(), // FIXME: unwanted clone of attrs
            false,
            true,
            FnParseMode { req_name: |_| true, req_body: true },
            force_collect,
        )? {
            self.mk_stmt(lo.to(item.span), StmtKind::Item(P(item)))
        } else if self.eat(exp!(Semi)) {
            // Do not attempt to parse an expression if we're done here.
            self.error_outer_attrs(attrs);
            self.mk_stmt(lo, StmtKind::Empty)
        } else if self.token != token::CloseBrace {
            // Remainder are line-expr stmts. This is similar to the `parse_stmt_path_start` case
            // above.
            let restrictions =
                if force_full_expr { Restrictions::empty() } else { Restrictions::STMT_EXPR };
            let e = self.collect_tokens(
                Some(pre_attr_pos),
                AttrWrapper::empty(),
                force_collect,
                |this, _empty_attrs| {
                    let (expr, _) = this.parse_expr_res(restrictions, attrs)?;
                    Ok((expr, Trailing::No, UsePreAttrPos::Yes))
                },
            )?;
            if matches!(e.kind, ExprKind::Assign(..)) && self.eat_keyword(exp!(Else)) {
                let bl = self.parse_block()?;
                // Destructuring assignment ... else.
                // This is not allowed, but point it out in a nice way.
                self.dcx().emit_err(errors::AssignmentElseNotAllowed { span: e.span.to(bl.span) });
            }
            self.mk_stmt(lo.to(e.span), StmtKind::Expr(e))
        } else {
            self.error_outer_attrs(attrs);
            return Ok(None);
        };

        self.maybe_augment_stashed_expr_in_pats_with_suggestions(&stmt);
        Ok(Some(stmt))
    }

    fn parse_stmt_path_start(&mut self, lo: Span, attrs: AttrWrapper) -> PResult<'a, Stmt> {
        let stmt = self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let path = this.parse_path(PathStyle::Expr)?;

            if this.eat(exp!(Bang)) {
                let stmt_mac = this.parse_stmt_mac(lo, attrs, path)?;
                return Ok((
                    stmt_mac,
                    Trailing::from(this.token == token::Semi),
                    UsePreAttrPos::No,
                ));
            }

            let expr = if this.eat(exp!(OpenBrace)) {
                this.parse_expr_struct(None, path, true)?
            } else {
                let hi = this.prev_token.span;
                this.mk_expr(lo.to(hi), ExprKind::Path(None, path))
            };

            let expr = this.with_res(Restrictions::STMT_EXPR, |this| {
                this.parse_expr_dot_or_call_with(attrs, expr, lo)
            })?;
            // `DUMMY_SP` will get overwritten later in this function
            Ok((
                this.mk_stmt(rustc_span::DUMMY_SP, StmtKind::Expr(expr)),
                Trailing::No,
                UsePreAttrPos::No,
            ))
        })?;

        if let StmtKind::Expr(expr) = stmt.kind {
            // Perform this outside of the `collect_tokens` closure, since our
            // outer attributes do not apply to this part of the expression.
            let (expr, _) = self.with_res(Restrictions::STMT_EXPR, |this| {
                this.parse_expr_assoc_rest_with(Bound::Unbounded, true, expr)
            })?;
            Ok(self.mk_stmt(lo.to(self.prev_token.span), StmtKind::Expr(expr)))
        } else {
            Ok(stmt)
        }
    }

    /// Parses a statement macro `mac!(args)` provided a `path` representing `mac`.
    /// At this point, the `!` token after the path has already been eaten.
    fn parse_stmt_mac(&mut self, lo: Span, attrs: AttrVec, path: ast::Path) -> PResult<'a, Stmt> {
        let args = self.parse_delim_args()?;
        let hi = self.prev_token.span;

        let style = match args.delim {
            Delimiter::Brace => MacStmtStyle::Braces,
            _ => MacStmtStyle::NoBraces,
        };

        let mac = P(MacCall { path, args });

        let kind = if (style == MacStmtStyle::Braces
            && !matches!(self.token.kind, token::Dot | token::Question))
            || matches!(
                self.token.kind,
                token::Semi
                    | token::Eof
                    | token::CloseInvisible(InvisibleOrigin::MetaVar(MetaVarKind::Stmt))
            ) {
            StmtKind::MacCall(P(MacCallStmt { mac, style, attrs, tokens: None }))
        } else {
            // Since none of the above applied, this is an expression statement macro.
            let e = self.mk_expr(lo.to(hi), ExprKind::MacCall(mac));
            let e = self.maybe_recover_from_bad_qpath(e)?;
            let e = self.parse_expr_dot_or_call_with(attrs, e, lo)?;
            let (e, _) = self.parse_expr_assoc_rest_with(Bound::Unbounded, false, e)?;
            StmtKind::Expr(e)
        };
        Ok(self.mk_stmt(lo.to(hi), kind))
    }

    /// Error on outer attributes in this context.
    /// Also error if the previous token was a doc comment.
    fn error_outer_attrs(&self, attrs: AttrWrapper) {
        if !attrs.is_empty()
            && let attrs @ [.., last] = &*attrs.take_for_recovery(self.psess)
        {
            if last.is_doc_comment() {
                self.dcx().emit_err(errors::DocCommentDoesNotDocumentAnything {
                    span: last.span,
                    missing_comma: None,
                });
            } else if attrs.iter().any(|a| a.style == AttrStyle::Outer) {
                self.dcx().emit_err(errors::ExpectedStatementAfterOuterAttr { span: last.span });
            }
        }
    }

    fn recover_stmt_local_after_let(
        &mut self,
        lo: Span,
        attrs: AttrWrapper,
        subdiagnostic: fn(Span) -> errors::InvalidVariableDeclarationSub,
        force_collect: ForceCollect,
    ) -> PResult<'a, Stmt> {
        let stmt = self.collect_tokens(None, attrs, force_collect, |this, attrs| {
            let local = this.parse_local(None, attrs)?;
            // FIXME - maybe capture semicolon in recovery?
            Ok((
                this.mk_stmt(lo.to(this.prev_token.span), StmtKind::Let(local)),
                Trailing::No,
                UsePreAttrPos::No,
            ))
        })?;
        self.dcx()
            .emit_err(errors::InvalidVariableDeclaration { span: lo, sub: subdiagnostic(lo) });
        Ok(stmt)
    }

    /// Parses a local variable declaration.
    fn parse_local(&mut self, super_: Option<Span>, attrs: AttrVec) -> PResult<'a, P<Local>> {
        let lo = super_.unwrap_or(self.prev_token.span);

        if self.token.is_keyword(kw::Const) && self.look_ahead(1, |t| t.is_ident()) {
            self.dcx().emit_err(errors::ConstLetMutuallyExclusive { span: lo.to(self.token.span) });
            self.bump();
        }

        let (pat, colon) =
            self.parse_pat_before_ty(None, RecoverComma::Yes, PatternLocation::LetBinding)?;

        let (err, ty, colon_sp) = if colon {
            // Save the state of the parser before parsing type normally, in case there is a `:`
            // instead of an `=` typo.
            let parser_snapshot_before_type = self.clone();
            let colon_sp = self.prev_token.span;
            match self.parse_ty() {
                Ok(ty) => (None, Some(ty), Some(colon_sp)),
                Err(mut err) => {
                    err.span_label(
                        colon_sp,
                        format!(
                            "while parsing the type for {}",
                            pat.descr()
                                .map_or_else(|| "the binding".to_string(), |n| format!("`{n}`"))
                        ),
                    );
                    // we use noexpect here because we don't actually expect Eq to be here
                    // but we are still checking for it in order to be able to handle it if
                    // it is there
                    let err = if self.check_noexpect(&token::Eq) {
                        err.emit();
                        None
                    } else {
                        // Rewind to before attempting to parse the type and continue parsing.
                        let parser_snapshot_after_type =
                            mem::replace(self, parser_snapshot_before_type);
                        Some((parser_snapshot_after_type, colon_sp, err))
                    };
                    (err, None, Some(colon_sp))
                }
            }
        } else {
            (None, None, None)
        };
        let init = match (self.parse_initializer(err.is_some()), err) {
            (Ok(init), None) => {
                // init parsed, ty parsed
                init
            }
            (Ok(init), Some((_, colon_sp, mut err))) => {
                // init parsed, ty error
                // Could parse the type as if it were the initializer, it is likely there was a
                // typo in the code: `:` instead of `=`. Add suggestion and emit the error.
                err.span_suggestion_short(
                    colon_sp,
                    "use `=` if you meant to assign",
                    " =",
                    Applicability::MachineApplicable,
                );
                err.emit();
                // As this was parsed successfully, continue as if the code has been fixed for the
                // rest of the file. It will still fail due to the emitted error, but we avoid
                // extra noise.
                init
            }
            (Err(init_err), Some((snapshot, _, ty_err))) => {
                // init error, ty error
                init_err.cancel();
                // Couldn't parse the type nor the initializer, only raise the type error and
                // return to the parser state before parsing the type as the initializer.
                // let x: <parse_error>;
                *self = snapshot;
                return Err(ty_err);
            }
            (Err(err), None) => {
                // init error, ty parsed
                // Couldn't parse the initializer and we're not attempting to recover a failed
                // parse of the type, return the error.
                return Err(err);
            }
        };
        let kind = match init {
            None => LocalKind::Decl,
            Some(init) => {
                if self.eat_keyword(exp!(Else)) {
                    if self.token.is_keyword(kw::If) {
                        // `let...else if`. Emit the same error that `parse_block()` would,
                        // but explicitly point out that this pattern is not allowed.
                        let msg = "conditional `else if` is not supported for `let...else`";
                        return Err(self.error_block_no_opening_brace_msg(Cow::from(msg)));
                    }
                    let els = self.parse_block()?;
                    self.check_let_else_init_bool_expr(&init);
                    self.check_let_else_init_trailing_brace(&init);
                    LocalKind::InitElse(init, els)
                } else {
                    LocalKind::Init(init)
                }
            }
        };
        let hi = if self.token == token::Semi { self.token.span } else { self.prev_token.span };
        Ok(P(ast::Local {
            super_,
            ty,
            pat,
            kind,
            id: DUMMY_NODE_ID,
            span: lo.to(hi),
            colon_sp,
            attrs,
            tokens: None,
        }))
    }

    fn check_let_else_init_bool_expr(&self, init: &ast::Expr) {
        if let ast::ExprKind::Binary(op, ..) = init.kind {
            if op.node.is_lazy() {
                self.dcx().emit_err(errors::InvalidExpressionInLetElse {
                    span: init.span,
                    operator: op.node.as_str(),
                    sugg: errors::WrapInParentheses::Expression {
                        left: init.span.shrink_to_lo(),
                        right: init.span.shrink_to_hi(),
                    },
                });
            }
        }
    }

    fn check_let_else_init_trailing_brace(&self, init: &ast::Expr) {
        if let Some(trailing) = classify::expr_trailing_brace(init) {
            let (span, sugg) = match trailing {
                TrailingBrace::MacCall(mac) => (
                    mac.span(),
                    errors::WrapInParentheses::MacroArgs {
                        left: mac.args.dspan.open,
                        right: mac.args.dspan.close,
                    },
                ),
                TrailingBrace::Expr(expr) => (
                    expr.span,
                    errors::WrapInParentheses::Expression {
                        left: expr.span.shrink_to_lo(),
                        right: expr.span.shrink_to_hi(),
                    },
                ),
            };
            self.dcx().emit_err(errors::InvalidCurlyInLetElse {
                span: span.with_lo(span.hi() - BytePos(1)),
                sugg,
            });
        }
    }

    /// Parses the RHS of a local variable declaration (e.g., `= 14;`).
    fn parse_initializer(&mut self, eq_optional: bool) -> PResult<'a, Option<P<Expr>>> {
        let eq_consumed = match self.token.kind {
            token::PlusEq
            | token::MinusEq
            | token::StarEq
            | token::SlashEq
            | token::PercentEq
            | token::CaretEq
            | token::AndEq
            | token::OrEq
            | token::ShlEq
            | token::ShrEq => {
                // Recover `let x <op>= 1` as `let x = 1` We must not use `+ BytePos(1)` here
                // because `<op>` can be a multi-byte lookalike that was recovered, e.g. `➖=` (the
                // `➖` is a U+2796 Heavy Minus Sign Unicode Character) that was recovered as a
                // `-=`.
                let extra_op_span = self.psess.source_map().start_point(self.token.span);
                self.dcx().emit_err(errors::CompoundAssignmentExpressionInLet {
                    span: self.token.span,
                    suggestion: extra_op_span,
                });
                self.bump();
                true
            }
            _ => self.eat(exp!(Eq)),
        };

        Ok(if eq_consumed || eq_optional { Some(self.parse_expr()?) } else { None })
    }

    /// Parses a block. No inner attributes are allowed.
    pub fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        let (attrs, block) = self.parse_inner_attrs_and_block(None)?;
        if let [.., last] = &*attrs {
            let suggest_to_outer = match &last.kind {
                ast::AttrKind::Normal(attr) => attr.item.is_valid_for_outer_style(),
                _ => false,
            };
            self.error_on_forbidden_inner_attr(
                last.span,
                super::attr::InnerAttrPolicy::Forbidden(Some(
                    InnerAttrForbiddenReason::InCodeBlock,
                )),
                suggest_to_outer,
            );
        }
        Ok(block)
    }

    fn error_block_no_opening_brace_msg(&mut self, msg: Cow<'static, str>) -> Diag<'a> {
        let prev = self.prev_token.span;
        let sp = self.token.span;
        let mut e = self.dcx().struct_span_err(sp, msg);
        self.label_expected_raw_ref(&mut e);

        let do_not_suggest_help = self.token.is_keyword(kw::In)
            || self.token == token::Colon
            || self.prev_token.is_keyword(kw::Raw);

        // Check to see if the user has written something like
        //
        //    if (cond)
        //      bar;
        //
        // which is valid in other languages, but not Rust.
        match self.parse_stmt_without_recovery(false, ForceCollect::No, false) {
            // If the next token is an open brace, e.g., we have:
            //
            //     if expr other_expr {
            //        ^    ^          ^- lookahead(1) is a brace
            //        |    |- current token is not "else"
            //        |- (statement we just parsed)
            //
            // the place-inside-a-block suggestion would be more likely wrong than right.
            //
            // FIXME(compiler-errors): this should probably parse an arbitrary expr and not
            // just lookahead one token, so we can see if there's a brace after _that_,
            // since we want to protect against:
            //     `if 1 1 + 1 {` being suggested as  `if { 1 } 1 + 1 {`
            //                                            +   +
            Ok(Some(_))
                if (!self.token.is_keyword(kw::Else)
                    && self.look_ahead(1, |t| t == &token::OpenBrace))
                    || do_not_suggest_help => {}
            // Do not suggest `if foo println!("") {;}` (as would be seen in test for #46836).
            Ok(Some(Stmt { kind: StmtKind::Empty, .. })) => {}
            Ok(Some(stmt)) => {
                let stmt_own_line = self.psess.source_map().is_line_before_span_empty(sp);
                let stmt_span = if stmt_own_line && self.eat(exp!(Semi)) {
                    // Expand the span to include the semicolon.
                    stmt.span.with_hi(self.prev_token.span.hi())
                } else {
                    stmt.span
                };
                self.suggest_fixes_misparsed_for_loop_head(
                    &mut e,
                    prev.between(sp),
                    stmt_span,
                    &stmt.kind,
                );
            }
            Err(e) => {
                self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
                e.cancel();
            }
            _ => {}
        }
        e.span_label(sp, "expected `{`");
        e
    }

    fn suggest_fixes_misparsed_for_loop_head(
        &self,
        e: &mut Diag<'_>,
        between: Span,
        stmt_span: Span,
        stmt_kind: &StmtKind,
    ) {
        match (&self.token.kind, &stmt_kind) {
            (token::OpenBrace, StmtKind::Expr(expr)) if let ExprKind::Call(..) = expr.kind => {
                // for _ in x y() {}
                e.span_suggestion_verbose(
                    between,
                    "you might have meant to write a method call",
                    ".".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            (token::OpenBrace, StmtKind::Expr(expr)) if let ExprKind::Field(..) = expr.kind => {
                // for _ in x y.z {}
                e.span_suggestion_verbose(
                    between,
                    "you might have meant to write a field access",
                    ".".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            (token::CloseBrace, StmtKind::Expr(expr))
                if let ExprKind::Struct(expr) = &expr.kind
                    && let None = expr.qself
                    && expr.path.segments.len() == 1 =>
            {
                // This is specific to "mistyped `if` condition followed by empty body"
                //
                // for _ in x y {}
                e.span_suggestion_verbose(
                    between,
                    "you might have meant to write a field access",
                    ".".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            (token::OpenBrace, StmtKind::Expr(expr))
                if let ExprKind::Lit(lit) = expr.kind
                    && let None = lit.suffix
                    && let token::LitKind::Integer | token::LitKind::Float = lit.kind =>
            {
                // for _ in x 0 {}
                // for _ in x 0.0 {}
                e.span_suggestion_verbose(
                    between,
                    format!("you might have meant to write a field access"),
                    ".".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            (token::OpenBrace, StmtKind::Expr(expr))
                if let ExprKind::Loop(..)
                | ExprKind::If(..)
                | ExprKind::While(..)
                | ExprKind::Match(..)
                | ExprKind::ForLoop { .. }
                | ExprKind::TryBlock(..)
                | ExprKind::Ret(..)
                | ExprKind::Closure(..)
                | ExprKind::Struct(..)
                | ExprKind::Try(..) = expr.kind =>
            {
                // These are more likely to have been meant as a block body.
                e.multipart_suggestion(
                    "you might have meant to write this as part of a block",
                    vec![
                        (stmt_span.shrink_to_lo(), "{ ".to_string()),
                        (stmt_span.shrink_to_hi(), " }".to_string()),
                    ],
                    // Speculative; has been misleading in the past (#46836).
                    Applicability::MaybeIncorrect,
                );
            }
            (token::OpenBrace, _) => {}
            (_, _) => {
                e.multipart_suggestion(
                    "you might have meant to write this as part of a block",
                    vec![
                        (stmt_span.shrink_to_lo(), "{ ".to_string()),
                        (stmt_span.shrink_to_hi(), " }".to_string()),
                    ],
                    // Speculative; has been misleading in the past (#46836).
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    fn error_block_no_opening_brace<T>(&mut self) -> PResult<'a, T> {
        let tok = super::token_descr(&self.token);
        let msg = format!("expected `{{`, found {tok}");
        Err(self.error_block_no_opening_brace_msg(Cow::from(msg)))
    }

    /// Parses a block. Inner attributes are allowed, block labels are not.
    ///
    /// If `loop_header` is `Some` and an unexpected block label is encountered,
    /// it is suggested to be moved just before `loop_header`, else it is suggested to be removed.
    pub(super) fn parse_inner_attrs_and_block(
        &mut self,
        loop_header: Option<Span>,
    ) -> PResult<'a, (AttrVec, P<Block>)> {
        self.parse_block_common(self.token.span, BlockCheckMode::Default, loop_header)
    }

    /// Parses a block. Inner attributes are allowed, block labels are not.
    ///
    /// If `loop_header` is `Some` and an unexpected block label is encountered,
    /// it is suggested to be moved just before `loop_header`, else it is suggested to be removed.
    pub(super) fn parse_block_common(
        &mut self,
        lo: Span,
        blk_mode: BlockCheckMode,
        loop_header: Option<Span>,
    ) -> PResult<'a, (AttrVec, P<Block>)> {
        if let Some(block) = self.eat_metavar_seq(MetaVarKind::Block, |this| this.parse_block()) {
            return Ok((AttrVec::new(), block));
        }

        let maybe_ident = self.prev_token;
        self.maybe_recover_unexpected_block_label(loop_header);
        if !self.eat(exp!(OpenBrace)) {
            return self.error_block_no_opening_brace();
        }

        let attrs = self.parse_inner_attributes()?;
        let tail = match self.maybe_suggest_struct_literal(lo, blk_mode, maybe_ident) {
            Some(tail) => tail?,
            None => self.parse_block_tail(lo, blk_mode, AttemptLocalParseRecovery::Yes)?,
        };
        Ok((attrs, tail))
    }

    /// Parses the rest of a block expression or function body.
    /// Precondition: already parsed the '{'.
    pub(crate) fn parse_block_tail(
        &mut self,
        lo: Span,
        s: BlockCheckMode,
        recover: AttemptLocalParseRecovery,
    ) -> PResult<'a, P<Block>> {
        let mut stmts = ThinVec::new();
        let mut snapshot = None;
        while !self.eat(exp!(CloseBrace)) {
            if self.token == token::Eof {
                break;
            }
            if self.is_vcs_conflict_marker(&TokenKind::Shl, &TokenKind::Lt) {
                // Account for `<<<<<<<` diff markers. We can't proactively error here because
                // that can be a valid path start, so we snapshot and reparse only we've
                // encountered another parse error.
                snapshot = Some(self.create_snapshot_for_diagnostic());
            }
            let stmt = match self.parse_full_stmt(recover) {
                Err(mut err) if recover.yes() => {
                    if let Some(ref mut snapshot) = snapshot {
                        snapshot.recover_vcs_conflict_marker();
                    }
                    if self.token == token::Colon {
                        // if a previous and next token of the current one is
                        // integer literal (e.g. `1:42`), it's likely a range
                        // expression for Pythonistas and we can suggest so.
                        if self.prev_token.is_integer_lit()
                            && self.may_recover()
                            && self.look_ahead(1, |token| token.is_integer_lit())
                        {
                            // FIXME(hkmatsumoto): Might be better to trigger
                            // this only when parsing an index expression.
                            err.span_suggestion_verbose(
                                self.token.span,
                                "you might have meant a range expression",
                                "..",
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            // if next token is following a colon, it's likely a path
                            // and we can suggest a path separator
                            self.bump();
                            if self.token.span.lo() == self.prev_token.span.hi() {
                                err.span_suggestion_verbose(
                                    self.prev_token.span,
                                    "maybe write a path separator here",
                                    "::",
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                    }

                    let guar = err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    Some(self.mk_stmt_err(self.token.span, guar))
                }
                Ok(stmt) => stmt,
                Err(err) => return Err(err),
            };
            if let Some(stmt) = stmt {
                stmts.push(stmt);
            } else {
                // Found only `;` or `}`.
                continue;
            };
        }
        Ok(self.mk_block(stmts, s, lo.to(self.prev_token.span)))
    }

    fn recover_missing_dot(&mut self, err: &mut Diag<'_>) {
        let Some((ident, _)) = self.token.ident() else {
            return;
        };
        if let Some(c) = ident.name.as_str().chars().next()
            && c.is_uppercase()
        {
            return;
        }
        if self.token.is_reserved_ident() && !self.token.is_ident_named(kw::Await) {
            return;
        }
        if self.prev_token.is_reserved_ident() && self.prev_token.is_ident_named(kw::Await) {
            // Likely `foo.await bar`
        } else if !self.prev_token.is_reserved_ident() && self.prev_token.is_ident() {
            // Likely `foo bar`
        } else if self.prev_token.kind == token::Question {
            // `foo? bar`
        } else if self.prev_token.kind == token::CloseParen {
            // `foo() bar`
        } else {
            return;
        }
        if self.token.span == self.prev_token.span {
            // Account for syntax errors in proc-macros.
            return;
        }
        if self.look_ahead(1, |t| [token::Semi, token::Question, token::Dot].contains(&t.kind)) {
            err.span_suggestion_verbose(
                self.prev_token.span.between(self.token.span),
                "you might have meant to write a field access",
                ".".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
        if self.look_ahead(1, |t| t.kind == token::OpenParen) {
            err.span_suggestion_verbose(
                self.prev_token.span.between(self.token.span),
                "you might have meant to write a method call",
                ".".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Parses a statement, including the trailing semicolon.
    pub fn parse_full_stmt(
        &mut self,
        recover: AttemptLocalParseRecovery,
    ) -> PResult<'a, Option<Stmt>> {
        // Skip looking for a trailing semicolon when we have a metavar seq.
        if let Some(stmt) = self.eat_metavar_seq(MetaVarKind::Stmt, |this| {
            // Why pass `true` for `force_full_expr`? Statement expressions are less expressive
            // than "full" expressions, due to the `STMT_EXPR` restriction, and sometimes need
            // parentheses. E.g. the "full" expression `match paren_around_match {} | true` when
            // used in statement context must be written `(match paren_around_match {} | true)`.
            // However, if the expression we are parsing in this statement context was pasted by a
            // declarative macro, it may have come from a "full" expression context, and lack
            // these parentheses. So we lift the `STMT_EXPR` restriction to ensure the statement
            // will reparse successfully.
            this.parse_stmt_without_recovery(false, ForceCollect::No, true)
        }) {
            let stmt = stmt.expect("an actual statement");
            return Ok(Some(stmt));
        }

        let Some(mut stmt) = self.parse_stmt_without_recovery(true, ForceCollect::No, false)?
        else {
            return Ok(None);
        };

        let mut eat_semi = true;
        let mut add_semi_to_stmt = false;

        match &mut stmt.kind {
            // Expression without semicolon.
            StmtKind::Expr(expr)
                if classify::expr_requires_semi_to_be_stmt(expr)
                    && !expr.attrs.is_empty()
                    && !matches!(self.token.kind, token::Eof | token::Semi | token::CloseBrace) =>
            {
                // The user has written `#[attr] expr` which is unsupported. (#106020)
                let guar = self.attr_on_non_tail_expr(&expr);
                // We already emitted an error, so don't emit another type error
                let sp = expr.span.to(self.prev_token.span);
                *expr = self.mk_expr_err(sp, guar);
            }

            // Expression without semicolon.
            StmtKind::Expr(expr)
                if self.token != token::Eof && classify::expr_requires_semi_to_be_stmt(expr) =>
            {
                // Just check for errors and recover; do not eat semicolon yet.

                let expect_result = self.expect_one_of(&[], &[exp!(Semi), exp!(CloseBrace)]);

                // Try to both emit a better diagnostic, and avoid further errors by replacing
                // the `expr` with `ExprKind::Err`.
                let replace_with_err = 'break_recover: {
                    match expect_result {
                        Ok(Recovered::No) => None,
                        Ok(Recovered::Yes(guar)) => {
                            // Skip type error to avoid extra errors.
                            Some(guar)
                        }
                        Err(e) => {
                            if self.recover_colon_as_semi() {
                                // recover_colon_as_semi has already emitted a nicer error.
                                e.delay_as_bug();
                                add_semi_to_stmt = true;
                                eat_semi = false;

                                break 'break_recover None;
                            }

                            match &expr.kind {
                                ExprKind::Path(None, ast::Path { segments, .. })
                                    if let [segment] = segments.as_slice() =>
                                {
                                    if self.token == token::Colon
                                        && self.look_ahead(1, |token| {
                                            token.is_metavar_block()
                                                || matches!(
                                                    token.kind,
                                                    token::Ident(
                                                        kw::For | kw::Loop | kw::While,
                                                        token::IdentIsRaw::No
                                                    ) | token::OpenBrace
                                                )
                                        })
                                    {
                                        let snapshot = self.create_snapshot_for_diagnostic();
                                        let label = Label {
                                            ident: Ident::from_str_and_span(
                                                &format!("'{}", segment.ident),
                                                segment.ident.span,
                                            ),
                                        };
                                        match self.parse_expr_labeled(label, false) {
                                            Ok(labeled_expr) => {
                                                e.cancel();
                                                self.dcx().emit_err(MalformedLoopLabel {
                                                    span: label.ident.span,
                                                    suggestion: label.ident.span.shrink_to_lo(),
                                                });
                                                *expr = labeled_expr;
                                                break 'break_recover None;
                                            }
                                            Err(err) => {
                                                err.cancel();
                                                self.restore_snapshot(snapshot);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }

                            let res =
                                self.check_mistyped_turbofish_with_multiple_type_params(e, expr);

                            Some(if recover.no() {
                                res?
                            } else {
                                res.unwrap_or_else(|mut e| {
                                    self.recover_missing_dot(&mut e);
                                    let guar = e.emit();
                                    self.recover_stmt();
                                    guar
                                })
                            })
                        }
                    }
                };

                if let Some(guar) = replace_with_err {
                    // We already emitted an error, so don't emit another type error
                    let sp = expr.span.to(self.prev_token.span);
                    *expr = self.mk_expr_err(sp, guar);
                }
            }
            StmtKind::Expr(_) | StmtKind::MacCall(_) => {}
            StmtKind::Let(local) if let Err(mut e) = self.expect_semi() => {
                // We might be at the `,` in `let x = foo<bar, baz>;`. Try to recover.
                match &mut local.kind {
                    LocalKind::Init(expr) | LocalKind::InitElse(expr, _) => {
                        self.check_mistyped_turbofish_with_multiple_type_params(e, expr).map_err(
                            |mut e| {
                                self.recover_missing_dot(&mut e);
                                e
                            },
                        )?;
                        // We found `foo<bar, baz>`, have we fully recovered?
                        self.expect_semi()?;
                    }
                    LocalKind::Decl => {
                        if let Some(colon_sp) = local.colon_sp {
                            e.span_label(
                                colon_sp,
                                format!(
                                    "while parsing the type for {}",
                                    local.pat.descr().map_or_else(
                                        || "the binding".to_string(),
                                        |n| format!("`{n}`")
                                    )
                                ),
                            );
                            let suggest_eq = if self.token == token::Dot
                                && let _ = self.bump()
                                && let mut snapshot = self.create_snapshot_for_diagnostic()
                                && let Ok(_) = snapshot
                                    .parse_dot_suffix_expr(
                                        colon_sp,
                                        self.mk_expr_err(
                                            colon_sp,
                                            self.dcx()
                                                .delayed_bug("error during `:` -> `=` recovery"),
                                        ),
                                    )
                                    .map_err(Diag::cancel)
                            {
                                true
                            } else if let Some(op) = self.check_assoc_op()
                                && op.node.can_continue_expr_unambiguously()
                            {
                                true
                            } else {
                                false
                            };
                            if suggest_eq {
                                e.span_suggestion_short(
                                    colon_sp,
                                    "use `=` if you meant to assign",
                                    "=",
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        return Err(e);
                    }
                }
                eat_semi = false;
            }
            StmtKind::Empty | StmtKind::Item(_) | StmtKind::Let(_) | StmtKind::Semi(_) => {
                eat_semi = false
            }
        }

        if add_semi_to_stmt || (eat_semi && self.eat(exp!(Semi))) {
            stmt = stmt.add_trailing_semicolon();
        }

        stmt.span = stmt.span.to(self.prev_token.span);
        Ok(Some(stmt))
    }

    pub(super) fn mk_block(
        &self,
        stmts: ThinVec<Stmt>,
        rules: BlockCheckMode,
        span: Span,
    ) -> P<Block> {
        P(Block { stmts, id: DUMMY_NODE_ID, rules, span, tokens: None })
    }

    pub(super) fn mk_stmt(&self, span: Span, kind: StmtKind) -> Stmt {
        Stmt { id: DUMMY_NODE_ID, kind, span }
    }

    pub(super) fn mk_stmt_err(&self, span: Span, guar: ErrorGuaranteed) -> Stmt {
        self.mk_stmt(span, StmtKind::Expr(self.mk_expr_err(span, guar)))
    }

    pub(super) fn mk_block_err(&self, span: Span, guar: ErrorGuaranteed) -> P<Block> {
        self.mk_block(thin_vec![self.mk_stmt_err(span, guar)], BlockCheckMode::Default, span)
    }
}
