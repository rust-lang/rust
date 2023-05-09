use super::attr::InnerAttrForbiddenReason;
use super::diagnostics::AttemptLocalParseRecovery;
use super::expr::LhsExpr;
use super::pat::{PatternLocation, RecoverComma};
use super::path::PathStyle;
use super::TrailingToken;
use super::{
    AttrWrapper, BlockMode, FnParseMode, ForceCollect, Parser, Restrictions, SemiColonMode,
};
use crate::errors;
use crate::maybe_whole;

use crate::errors::MalformedLoopLabel;
use ast::Label;
use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, TokenKind};
use rustc_ast::util::classify;
use rustc_ast::{AttrStyle, AttrVec, LocalKind, MacCall, MacCallStmt, MacStmtStyle};
use rustc_ast::{Block, BlockCheckMode, Expr, ExprKind, HasAttrs, Local, Stmt};
use rustc_ast::{StmtKind, DUMMY_NODE_ID};
use rustc_errors::{Applicability, DiagnosticBuilder, ErrorGuaranteed, PResult};
use rustc_span::source_map::{BytePos, Span};
use rustc_span::symbol::{kw, sym, Ident};

use std::mem;
use thin_vec::{thin_vec, ThinVec};

impl<'a> Parser<'a> {
    /// Parses a statement. This stops just before trailing semicolons on everything but items.
    /// e.g., a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    // Public for rustfmt usage.
    pub fn parse_stmt(&mut self, force_collect: ForceCollect) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_without_recovery(false, force_collect).unwrap_or_else(|mut e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        }))
    }

    /// If `force_collect` is [`ForceCollect::Yes`], forces collection of tokens regardless of whether
    /// or not we have attributes
    pub(crate) fn parse_stmt_without_recovery(
        &mut self,
        capture_semi: bool,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Stmt>> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        // Don't use `maybe_whole` so that we have precise control
        // over when we bump the parser
        if let token::Interpolated(nt) = &self.token.kind && let token::NtStmt(stmt) = &**nt {
            let mut stmt = stmt.clone();
            self.bump();
            stmt.visit_attrs(|stmt_attrs| {
                attrs.prepend_to_nt_inner(stmt_attrs);
            });
            return Ok(Some(stmt.into_inner()));
        }

        if self.token.is_keyword(kw::Mut) && self.is_keyword_ahead(1, &[kw::Let]) {
            self.bump();
            let mut_let_span = lo.to(self.token.span);
            self.sess.emit_err(errors::InvalidVariableDeclaration {
                span: mut_let_span,
                sub: errors::InvalidVariableDeclarationSub::SwitchMutLetOrder(mut_let_span),
            });
        }

        Ok(Some(if self.token.is_keyword(kw::Let) {
            self.parse_local_mk(lo, attrs, capture_semi, force_collect)?
        } else if self.is_kw_followed_by_ident(kw::Mut) && self.may_recover() {
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::MissingLet,
            )?
        } else if self.is_kw_followed_by_ident(kw::Auto) && self.may_recover() {
            self.bump(); // `auto`
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::UseLetNotAuto,
            )?
        } else if self.is_kw_followed_by_ident(sym::var) && self.may_recover() {
            self.bump(); // `var`
            self.recover_stmt_local_after_let(
                lo,
                attrs,
                errors::InvalidVariableDeclarationSub::UseLetNotVar,
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
            match force_collect {
                ForceCollect::Yes => {
                    self.collect_tokens_no_attrs(|this| this.parse_stmt_path_start(lo, attrs))?
                }
                ForceCollect::No => match self.parse_stmt_path_start(lo, attrs) {
                    Ok(stmt) => stmt,
                    Err(mut err) => {
                        self.suggest_add_missing_let_for_stmt(&mut err);
                        return Err(err);
                    }
                },
            }
        } else if let Some(item) = self.parse_item_common(
            attrs.clone(),
            false,
            true,
            FnParseMode { req_name: |_| true, req_body: true },
            force_collect,
        )? {
            // FIXME: Bad copy of attrs
            self.mk_stmt(lo.to(item.span), StmtKind::Item(P(item)))
        } else if self.eat(&token::Semi) {
            // Do not attempt to parse an expression if we're done here.
            self.error_outer_attrs(attrs);
            self.mk_stmt(lo, StmtKind::Empty)
        } else if self.token != token::CloseDelim(Delimiter::Brace) {
            // Remainder are line-expr stmts.
            let e = match force_collect {
                ForceCollect::Yes => self.collect_tokens_no_attrs(|this| {
                    this.parse_expr_res(Restrictions::STMT_EXPR, Some(attrs))
                })?,
                ForceCollect::No => self.parse_expr_res(Restrictions::STMT_EXPR, Some(attrs))?,
            };
            if matches!(e.kind, ExprKind::Assign(..)) && self.eat_keyword(kw::Else) {
                let bl = self.parse_block()?;
                // Destructuring assignment ... else.
                // This is not allowed, but point it out in a nice way.
                self.sess.emit_err(errors::AssignmentElseNotAllowed { span: e.span.to(bl.span) });
            }
            self.mk_stmt(lo.to(e.span), StmtKind::Expr(e))
        } else {
            self.error_outer_attrs(attrs);
            return Ok(None);
        }))
    }

    fn parse_stmt_path_start(&mut self, lo: Span, attrs: AttrWrapper) -> PResult<'a, Stmt> {
        let stmt = self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
            let path = this.parse_path(PathStyle::Expr)?;

            if this.eat(&token::Not) {
                let stmt_mac = this.parse_stmt_mac(lo, attrs, path)?;
                if this.token == token::Semi {
                    return Ok((stmt_mac, TrailingToken::Semi));
                } else {
                    return Ok((stmt_mac, TrailingToken::None));
                }
            }

            let expr = if this.eat(&token::OpenDelim(Delimiter::Brace)) {
                this.parse_expr_struct(None, path, true)?
            } else {
                let hi = this.prev_token.span;
                this.mk_expr(lo.to(hi), ExprKind::Path(None, path))
            };

            let expr = this.with_res(Restrictions::STMT_EXPR, |this| {
                this.parse_expr_dot_or_call_with(expr, lo, attrs)
            })?;
            // `DUMMY_SP` will get overwritten later in this function
            Ok((this.mk_stmt(rustc_span::DUMMY_SP, StmtKind::Expr(expr)), TrailingToken::None))
        })?;

        if let StmtKind::Expr(expr) = stmt.kind {
            // Perform this outside of the `collect_tokens_trailing_token` closure,
            // since our outer attributes do not apply to this part of the expression
            let expr = self.with_res(Restrictions::STMT_EXPR, |this| {
                this.parse_expr_assoc_with(
                    0,
                    LhsExpr::AlreadyParsed { expr, starts_statement: true },
                )
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
        let delim = args.delim.to_token();
        let hi = self.prev_token.span;

        let style = match delim {
            Delimiter::Brace => MacStmtStyle::Braces,
            _ => MacStmtStyle::NoBraces,
        };

        let mac = P(MacCall { path, args });

        let kind = if (style == MacStmtStyle::Braces
            && self.token != token::Dot
            && self.token != token::Question)
            || self.token == token::Semi
            || self.token == token::Eof
        {
            StmtKind::MacCall(P(MacCallStmt { mac, style, attrs, tokens: None }))
        } else {
            // Since none of the above applied, this is an expression statement macro.
            let e = self.mk_expr(lo.to(hi), ExprKind::MacCall(mac));
            let e = self.maybe_recover_from_bad_qpath(e)?;
            let e = self.parse_expr_dot_or_call_with(e, lo, attrs)?;
            let e = self.parse_expr_assoc_with(
                0,
                LhsExpr::AlreadyParsed { expr: e, starts_statement: false },
            )?;
            StmtKind::Expr(e)
        };
        Ok(self.mk_stmt(lo.to(hi), kind))
    }

    /// Error on outer attributes in this context.
    /// Also error if the previous token was a doc comment.
    fn error_outer_attrs(&self, attrs: AttrWrapper) {
        if !attrs.is_empty()
        && let attrs = attrs.take_for_recovery(self.sess)
        && let attrs @ [.., last] = &*attrs {
            if last.is_doc_comment() {
                self.sess.emit_err(errors::DocCommentDoesNotDocumentAnything {
                    span: last.span,
                    missing_comma: None,
                });
            } else if attrs.iter().any(|a| a.style == AttrStyle::Outer) {
                self.sess.emit_err(errors::ExpectedStatementAfterOuterAttr { span: last.span });
            }
        }
    }

    fn recover_stmt_local_after_let(
        &mut self,
        lo: Span,
        attrs: AttrWrapper,
        subdiagnostic: fn(Span) -> errors::InvalidVariableDeclarationSub,
    ) -> PResult<'a, Stmt> {
        let stmt =
            self.collect_tokens_trailing_token(attrs, ForceCollect::Yes, |this, attrs| {
                let local = this.parse_local(attrs)?;
                // FIXME - maybe capture semicolon in recovery?
                Ok((
                    this.mk_stmt(lo.to(this.prev_token.span), StmtKind::Local(local)),
                    TrailingToken::None,
                ))
            })?;
        self.sess.emit_err(errors::InvalidVariableDeclaration { span: lo, sub: subdiagnostic(lo) });
        Ok(stmt)
    }

    fn parse_local_mk(
        &mut self,
        lo: Span,
        attrs: AttrWrapper,
        capture_semi: bool,
        force_collect: ForceCollect,
    ) -> PResult<'a, Stmt> {
        self.collect_tokens_trailing_token(attrs, force_collect, |this, attrs| {
            this.expect_keyword(kw::Let)?;
            let local = this.parse_local(attrs)?;
            let trailing = if capture_semi && this.token.kind == token::Semi {
                TrailingToken::Semi
            } else {
                TrailingToken::None
            };
            Ok((this.mk_stmt(lo.to(this.prev_token.span), StmtKind::Local(local)), trailing))
        })
    }

    /// Parses a local variable declaration.
    fn parse_local(&mut self, attrs: AttrVec) -> PResult<'a, P<Local>> {
        let lo = self.prev_token.span;

        if self.token.is_keyword(kw::Const) && self.look_ahead(1, |t| t.is_ident()) {
            self.sess.emit_err(errors::ConstLetMutuallyExclusive { span: lo.to(self.token.span) });
            self.bump();
        }

        let (pat, colon) =
            self.parse_pat_before_ty(None, RecoverComma::Yes, PatternLocation::LetBinding)?;

        let (err, ty) = if colon {
            // Save the state of the parser before parsing type normally, in case there is a `:`
            // instead of an `=` typo.
            let parser_snapshot_before_type = self.clone();
            let colon_sp = self.prev_token.span;
            match self.parse_ty() {
                Ok(ty) => (None, Some(ty)),
                Err(mut err) => {
                    if let Ok(snip) = self.span_to_snippet(pat.span) {
                        err.span_label(pat.span, format!("while parsing the type for `{}`", snip));
                    }
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
                    (err, None)
                }
            }
        } else {
            (None, None)
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
                if self.eat_keyword(kw::Else) {
                    if self.token.is_keyword(kw::If) {
                        // `let...else if`. Emit the same error that `parse_block()` would,
                        // but explicitly point out that this pattern is not allowed.
                        let msg = "conditional `else if` is not supported for `let...else`";
                        return Err(self.error_block_no_opening_brace_msg(msg));
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
        Ok(P(ast::Local { ty, pat, kind, id: DUMMY_NODE_ID, span: lo.to(hi), attrs, tokens: None }))
    }

    fn check_let_else_init_bool_expr(&self, init: &ast::Expr) {
        if let ast::ExprKind::Binary(op, ..) = init.kind {
            if op.node.lazy() {
                self.sess.emit_err(errors::InvalidExpressionInLetElse {
                    span: init.span,
                    operator: op.node.to_string(),
                    sugg: errors::WrapExpressionInParentheses {
                        left: init.span.shrink_to_lo(),
                        right: init.span.shrink_to_hi(),
                    },
                });
            }
        }
    }

    fn check_let_else_init_trailing_brace(&self, init: &ast::Expr) {
        if let Some(trailing) = classify::expr_trailing_brace(init) {
            self.sess.emit_err(errors::InvalidCurlyInLetElse {
                span: trailing.span.with_lo(trailing.span.hi() - BytePos(1)),
                sugg: errors::WrapExpressionInParentheses {
                    left: trailing.span.shrink_to_lo(),
                    right: trailing.span.shrink_to_hi(),
                },
            });
        }
    }

    /// Parses the RHS of a local variable declaration (e.g., `= 14;`).
    fn parse_initializer(&mut self, eq_optional: bool) -> PResult<'a, Option<P<Expr>>> {
        let eq_consumed = match self.token.kind {
            token::BinOpEq(..) => {
                // Recover `let x <op>= 1` as `let x = 1`
                self.sess
                    .emit_err(errors::CompoundAssignmentExpressionInLet { span: self.token.span });
                self.bump();
                true
            }
            _ => self.eat(&token::Eq),
        };

        Ok(if eq_consumed || eq_optional { Some(self.parse_expr()?) } else { None })
    }

    /// Parses a block. No inner attributes are allowed.
    pub(super) fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        let (attrs, block) = self.parse_inner_attrs_and_block()?;
        if let [.., last] = &*attrs {
            self.error_on_forbidden_inner_attr(
                last.span,
                super::attr::InnerAttrPolicy::Forbidden(Some(
                    InnerAttrForbiddenReason::InCodeBlock,
                )),
            );
        }
        Ok(block)
    }

    fn error_block_no_opening_brace_msg(
        &mut self,
        msg: &str,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let sp = self.token.span;
        let mut e = self.struct_span_err(sp, msg);
        let do_not_suggest_help = self.token.is_keyword(kw::In) || self.token == token::Colon;

        // Check to see if the user has written something like
        //
        //    if (cond)
        //      bar;
        //
        // which is valid in other languages, but not Rust.
        match self.parse_stmt_without_recovery(false, ForceCollect::No) {
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
                    && self.look_ahead(1, |t| t == &token::OpenDelim(Delimiter::Brace)))
                    || do_not_suggest_help => {}
            // Do not suggest `if foo println!("") {;}` (as would be seen in test for #46836).
            Ok(Some(Stmt { kind: StmtKind::Empty, .. })) => {}
            Ok(Some(stmt)) => {
                let stmt_own_line = self.sess.source_map().is_line_before_span_empty(sp);
                let stmt_span = if stmt_own_line && self.eat(&token::Semi) {
                    // Expand the span to include the semicolon.
                    stmt.span.with_hi(self.prev_token.span.hi())
                } else {
                    stmt.span
                };
                e.multipart_suggestion(
                    "try placing this code inside a block",
                    vec![
                        (stmt_span.shrink_to_lo(), "{ ".to_string()),
                        (stmt_span.shrink_to_hi(), " }".to_string()),
                    ],
                    // Speculative; has been misleading in the past (#46836).
                    Applicability::MaybeIncorrect,
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

    fn error_block_no_opening_brace<T>(&mut self) -> PResult<'a, T> {
        let tok = super::token_descr(&self.token);
        let msg = format!("expected `{{`, found {}", tok);
        Err(self.error_block_no_opening_brace_msg(&msg))
    }

    /// Parses a block. Inner attributes are allowed.
    pub(super) fn parse_inner_attrs_and_block(&mut self) -> PResult<'a, (AttrVec, P<Block>)> {
        self.parse_block_common(self.token.span, BlockCheckMode::Default, true)
    }

    /// Parses a block. Inner attributes are allowed.
    pub(super) fn parse_block_common(
        &mut self,
        lo: Span,
        blk_mode: BlockCheckMode,
        can_be_struct_literal: bool,
    ) -> PResult<'a, (AttrVec, P<Block>)> {
        maybe_whole!(self, NtBlock, |x| (AttrVec::new(), x));

        let maybe_ident = self.prev_token.clone();
        self.maybe_recover_unexpected_block_label();
        if !self.eat(&token::OpenDelim(Delimiter::Brace)) {
            return self.error_block_no_opening_brace();
        }

        let attrs = self.parse_inner_attributes()?;
        let tail = match self.maybe_suggest_struct_literal(
            lo,
            blk_mode,
            maybe_ident,
            can_be_struct_literal,
        ) {
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
        while !self.eat(&token::CloseDelim(Delimiter::Brace)) {
            if self.token == token::Eof {
                break;
            }
            if self.is_diff_marker(&TokenKind::BinOp(token::Shl), &TokenKind::Lt) {
                // Account for `<<<<<<<` diff markers. We can't proactively error here because
                // that can be a valid path start, so we snapshot and reparse only we've
                // encountered another parse error.
                snapshot = Some(self.create_snapshot_for_diagnostic());
            }
            let stmt = match self.parse_full_stmt(recover) {
                Err(mut err) if recover.yes() => {
                    if let Some(ref mut snapshot) = snapshot {
                        snapshot.recover_diff_marker();
                    }
                    if self.token == token::Colon {
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
                        if self.sess.unstable_features.is_nightly_build() {
                            // FIXME(Nilstrieb): Remove this again after a few months.
                            err.note("type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>");
                        }
                    }

                    err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    Some(self.mk_stmt_err(self.token.span))
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

    /// Parses a statement, including the trailing semicolon.
    pub fn parse_full_stmt(
        &mut self,
        recover: AttemptLocalParseRecovery,
    ) -> PResult<'a, Option<Stmt>> {
        // Skip looking for a trailing semicolon when we have an interpolated statement.
        maybe_whole!(self, NtStmt, |x| Some(x.into_inner()));

        let Some(mut stmt) = self.parse_stmt_without_recovery(true, ForceCollect::No)? else {
            return Ok(None);
        };

        let mut eat_semi = true;
        let mut add_semi_to_stmt = false;

        match &mut stmt.kind {
            // Expression without semicolon.
            StmtKind::Expr(expr)
                if self.token != token::Eof && classify::expr_requires_semi_to_be_stmt(expr) => {
                // Just check for errors and recover; do not eat semicolon yet.
                // `expect_one_of` returns PResult<'a, bool /* recovered */>

                let expect_result = self.expect_one_of(&[], &[token::Semi, token::CloseDelim(Delimiter::Brace)]);

                let replace_with_err = 'break_recover: {
                    match expect_result {
                    // Recover from parser, skip type error to avoid extra errors.
                        Ok(true) => true,
                        Err(mut e) => {
                            if let TokenKind::DocComment(..) = self.token.kind
                                && let Ok(snippet) = self.span_to_snippet(self.token.span)
                            {
                                let sp = self.token.span;
                                let marker = &snippet[..3];
                                let (comment_marker, doc_comment_marker) = marker.split_at(2);

                                e.span_suggestion(
                                    sp.with_hi(sp.lo() + BytePos(marker.len() as u32)),
                                    format!(
                                        "add a space before `{}` to use a regular comment",
                                        doc_comment_marker,
                                    ),
                                    format!("{} {}", comment_marker, doc_comment_marker),
                                    Applicability::MaybeIncorrect,
                                );
                            }

                            if self.recover_colon_as_semi() {
                                // recover_colon_as_semi has already emitted a nicer error.
                                e.delay_as_bug();
                                add_semi_to_stmt = true;
                                eat_semi = false;

                                break 'break_recover false;
                            }

                            match &expr.kind {
                                ExprKind::Path(None, ast::Path { segments, .. }) if segments.len() == 1 => {
                                    if self.token == token::Colon
                                        && self.look_ahead(1, |token| {
                                            token.is_whole_block() || matches!(
                                                token.kind,
                                                token::Ident(kw::For | kw::Loop | kw::While, false)
                                                    | token::OpenDelim(Delimiter::Brace)
                                            )
                                        })
                                    {
                                        let snapshot = self.create_snapshot_for_diagnostic();
                                        let label = Label {
                                            ident: Ident::from_str_and_span(
                                                &format!("'{}", segments[0].ident),
                                                segments[0].ident.span,
                                            ),
                                        };
                                        match self.parse_expr_labeled(label, false) {
                                            Ok(labeled_expr) => {
                                                e.delay_as_bug();
                                                self.sess.emit_err(MalformedLoopLabel {
                                                    span: label.ident.span,
                                                    correct_label: label.ident,
                                                });
                                                *expr = labeled_expr;
                                                break 'break_recover false;
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

                            if let Err(mut e) =
                                self.check_mistyped_turbofish_with_multiple_type_params(e, expr)
                            {
                                if recover.no() {
                                    return Err(e);
                                }
                                e.emit();
                                self.recover_stmt();
                            }

                            true

                        }
                        Ok(false) => false
                    }
                };

                if replace_with_err {
                    // We already emitted an error, so don't emit another type error
                    let sp = expr.span.to(self.prev_token.span);
                    *expr = self.mk_expr_err(sp);
                }
            }
            StmtKind::Expr(_) | StmtKind::MacCall(_) => {}
            StmtKind::Local(local) if let Err(e) = self.expect_semi() => {
                // We might be at the `,` in `let x = foo<bar, baz>;`. Try to recover.
                match &mut local.kind {
                    LocalKind::Init(expr) | LocalKind::InitElse(expr, _) => {
                        self.check_mistyped_turbofish_with_multiple_type_params(e, expr)?;
                        // We found `foo<bar, baz>`, have we fully recovered?
                        self.expect_semi()?;
                    }
                    LocalKind::Decl => return Err(e),
                }
                eat_semi = false;
            }
            StmtKind::Empty | StmtKind::Item(_) | StmtKind::Local(_) | StmtKind::Semi(_) => eat_semi = false,
        }

        if add_semi_to_stmt || (eat_semi && self.eat(&token::Semi)) {
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
        P(Block {
            stmts,
            id: DUMMY_NODE_ID,
            rules,
            span,
            tokens: None,
            could_be_bare_literal: false,
        })
    }

    pub(super) fn mk_stmt(&self, span: Span, kind: StmtKind) -> Stmt {
        Stmt { id: DUMMY_NODE_ID, kind, span }
    }

    pub(super) fn mk_stmt_err(&self, span: Span) -> Stmt {
        self.mk_stmt(span, StmtKind::Expr(self.mk_expr_err(span)))
    }

    pub(super) fn mk_block_err(&self, span: Span) -> P<Block> {
        self.mk_block(thin_vec![self.mk_stmt_err(span)], BlockCheckMode::Default, span)
    }
}
