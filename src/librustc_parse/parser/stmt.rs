use super::attr::DEFAULT_INNER_ATTR_FORBIDDEN;
use super::diagnostics::Error;
use super::expr::LhsExpr;
use super::pat::GateOr;
use super::path::PathStyle;
use super::{BlockMode, Parser, Restrictions, SemiColonMode};
use crate::maybe_whole;

use rustc_ast::ast;
use rustc_ast::ast::{AttrStyle, AttrVec, Attribute, MacCall, MacStmtStyle};
use rustc_ast::ast::{Block, BlockCheckMode, Expr, ExprKind, Local, Stmt, StmtKind, DUMMY_NODE_ID};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, TokenKind};
use rustc_ast::util::classify;
use rustc_errors::{Applicability, PResult};
use rustc_span::source_map::{BytePos, Span};
use rustc_span::symbol::{kw, sym};

use std::mem;

impl<'a> Parser<'a> {
    /// Parses a statement. This stops just before trailing semicolons on everything but items.
    /// e.g., a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    pub fn parse_stmt(&mut self) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_without_recovery().unwrap_or_else(|mut e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        }))
    }

    fn parse_stmt_without_recovery(&mut self) -> PResult<'a, Option<Stmt>> {
        maybe_whole!(self, NtStmt, |x| Some(x));

        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        let stmt = if self.eat_keyword(kw::Let) {
            self.parse_local_mk(lo, attrs.into())?
        } else if self.is_kw_followed_by_ident(kw::Mut) {
            self.recover_stmt_local(lo, attrs.into(), "missing keyword", "let mut")?
        } else if self.is_kw_followed_by_ident(kw::Auto) {
            self.bump(); // `auto`
            let msg = "write `let` instead of `auto` to introduce a new variable";
            self.recover_stmt_local(lo, attrs.into(), msg, "let")?
        } else if self.is_kw_followed_by_ident(sym::var) {
            self.bump(); // `var`
            let msg = "write `let` instead of `var` to introduce a new variable";
            self.recover_stmt_local(lo, attrs.into(), msg, "let")?
        } else if self.check_path() && !self.token.is_qpath_start() && !self.is_path_start_item() {
            // We have avoided contextual keywords like `union`, items with `crate` visibility,
            // or `auto trait` items. We aim to parse an arbitrary path `a::b` but not something
            // that starts like a path (1 token), but it fact not a path.
            // Also, we avoid stealing syntax from `parse_item_`.
            self.parse_stmt_path_start(lo, attrs)?
        } else if let Some(item) = self.parse_item_common(attrs.clone(), false, true, |_| true)? {
            // FIXME: Bad copy of attrs
            self.mk_stmt(lo.to(item.span), StmtKind::Item(P(item)))
        } else if self.eat(&token::Semi) {
            // Do not attempt to parse an expression if we're done here.
            self.error_outer_attrs(&attrs);
            self.mk_stmt(lo, StmtKind::Empty)
        } else if self.token != token::CloseDelim(token::Brace) {
            // Remainder are line-expr stmts.
            let e = self.parse_expr_res(Restrictions::STMT_EXPR, Some(attrs.into()))?;
            self.mk_stmt(lo.to(e.span), StmtKind::Expr(e))
        } else {
            self.error_outer_attrs(&attrs);
            return Ok(None);
        };
        Ok(Some(stmt))
    }

    fn parse_stmt_path_start(&mut self, lo: Span, attrs: Vec<Attribute>) -> PResult<'a, Stmt> {
        let path = self.parse_path(PathStyle::Expr)?;

        if self.eat(&token::Not) {
            return self.parse_stmt_mac(lo, attrs.into(), path);
        }

        let expr = if self.check(&token::OpenDelim(token::Brace)) {
            self.parse_struct_expr(lo, path, AttrVec::new())?
        } else {
            let hi = self.prev_token.span;
            self.mk_expr(lo.to(hi), ExprKind::Path(None, path), AttrVec::new())
        };

        let expr = self.with_res(Restrictions::STMT_EXPR, |this| {
            let expr = this.parse_dot_or_call_expr_with(expr, lo, attrs.into())?;
            this.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(expr))
        })?;
        Ok(self.mk_stmt(lo.to(self.prev_token.span), StmtKind::Expr(expr)))
    }

    /// Parses a statement macro `mac!(args)` provided a `path` representing `mac`.
    /// At this point, the `!` token after the path has already been eaten.
    fn parse_stmt_mac(&mut self, lo: Span, attrs: AttrVec, path: ast::Path) -> PResult<'a, Stmt> {
        let args = self.parse_mac_args()?;
        let delim = args.delim();
        let hi = self.prev_token.span;

        let style =
            if delim == token::Brace { MacStmtStyle::Braces } else { MacStmtStyle::NoBraces };

        let mac = MacCall { path, args, prior_type_ascription: self.last_type_ascription };

        let kind = if delim == token::Brace || self.token == token::Semi || self.token == token::Eof
        {
            StmtKind::MacCall(P((mac, style, attrs)))
        } else {
            // Since none of the above applied, this is an expression statement macro.
            let e = self.mk_expr(lo.to(hi), ExprKind::MacCall(mac), AttrVec::new());
            let e = self.maybe_recover_from_bad_qpath(e, true)?;
            let e = self.parse_dot_or_call_expr_with(e, lo, attrs)?;
            let e = self.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(e))?;
            StmtKind::Expr(e)
        };
        Ok(self.mk_stmt(lo.to(hi), kind))
    }

    /// Error on outer attributes in this context.
    /// Also error if the previous token was a doc comment.
    fn error_outer_attrs(&self, attrs: &[Attribute]) {
        if let [.., last] = attrs {
            if last.is_doc_comment() {
                self.span_fatal_err(last.span, Error::UselessDocComment).emit();
            } else if attrs.iter().any(|a| a.style == AttrStyle::Outer) {
                self.struct_span_err(last.span, "expected statement after outer attribute").emit();
            }
        }
    }

    fn recover_stmt_local(
        &mut self,
        lo: Span,
        attrs: AttrVec,
        msg: &str,
        sugg: &str,
    ) -> PResult<'a, Stmt> {
        let stmt = self.parse_local_mk(lo, attrs)?;
        self.struct_span_err(lo, "invalid variable declaration")
            .span_suggestion(lo, msg, sugg.to_string(), Applicability::MachineApplicable)
            .emit();
        Ok(stmt)
    }

    fn parse_local_mk(&mut self, lo: Span, attrs: AttrVec) -> PResult<'a, Stmt> {
        let local = self.parse_local(attrs)?;
        Ok(self.mk_stmt(lo.to(self.prev_token.span), StmtKind::Local(local)))
    }

    /// Parses a local variable declaration.
    fn parse_local(&mut self, attrs: AttrVec) -> PResult<'a, P<Local>> {
        let lo = self.prev_token.span;
        let pat = self.parse_top_pat(GateOr::Yes)?;

        let (err, ty) = if self.eat(&token::Colon) {
            // Save the state of the parser before parsing type normally, in case there is a `:`
            // instead of an `=` typo.
            let parser_snapshot_before_type = self.clone();
            let colon_sp = self.prev_token.span;
            match self.parse_ty() {
                Ok(ty) => (None, Some(ty)),
                Err(mut err) => {
                    // Rewind to before attempting to parse the type and continue parsing.
                    let parser_snapshot_after_type = self.clone();
                    mem::replace(self, parser_snapshot_before_type);

                    let snippet = self.span_to_snippet(pat.span).unwrap();
                    err.span_label(pat.span, format!("while parsing the type for `{}`", snippet));
                    (Some((parser_snapshot_after_type, colon_sp, err)), None)
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
                    " =".to_string(),
                    Applicability::MachineApplicable,
                );
                err.emit();
                // As this was parsed successfully, continue as if the code has been fixed for the
                // rest of the file. It will still fail due to the emitted error, but we avoid
                // extra noise.
                init
            }
            (Err(mut init_err), Some((snapshot, _, ty_err))) => {
                // init error, ty error
                init_err.cancel();
                // Couldn't parse the type nor the initializer, only raise the type error and
                // return to the parser state before parsing the type as the initializer.
                // let x: <parse_error>;
                mem::replace(self, snapshot);
                return Err(ty_err);
            }
            (Err(err), None) => {
                // init error, ty parsed
                // Couldn't parse the initializer and we're not attempting to recover a failed
                // parse of the type, return the error.
                return Err(err);
            }
        };
        let hi = if self.token == token::Semi { self.token.span } else { self.prev_token.span };
        Ok(P(ast::Local { ty, pat, init, id: DUMMY_NODE_ID, span: lo.to(hi), attrs }))
    }

    /// Parses the RHS of a local variable declaration (e.g., '= 14;').
    fn parse_initializer(&mut self, skip_eq: bool) -> PResult<'a, Option<P<Expr>>> {
        if self.eat(&token::Eq) {
            Ok(Some(self.parse_expr()?))
        } else if skip_eq {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    /// Parses a block. No inner attributes are allowed.
    pub fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        let (attrs, block) = self.parse_inner_attrs_and_block()?;
        if let [.., last] = &*attrs {
            self.error_on_forbidden_inner_attr(last.span, DEFAULT_INNER_ATTR_FORBIDDEN);
        }
        Ok(block)
    }

    fn error_block_no_opening_brace<T>(&mut self) -> PResult<'a, T> {
        let sp = self.token.span;
        let tok = super::token_descr(&self.token);
        let mut e = self.struct_span_err(sp, &format!("expected `{{`, found {}", tok));
        let do_not_suggest_help = self.token.is_keyword(kw::In) || self.token == token::Colon;

        // Check to see if the user has written something like
        //
        //    if (cond)
        //      bar;
        //
        // which is valid in other languages, but not Rust.
        match self.parse_stmt_without_recovery() {
            // If the next token is an open brace (e.g., `if a b {`), the place-
            // inside-a-block suggestion would be more likely wrong than right.
            Ok(Some(_))
                if self.look_ahead(1, |t| t == &token::OpenDelim(token::Brace))
                    || do_not_suggest_help => {}
            Ok(Some(stmt)) => {
                let stmt_own_line = self.sess.source_map().is_line_before_span_empty(sp);
                let stmt_span = if stmt_own_line && self.eat(&token::Semi) {
                    // Expand the span to include the semicolon.
                    stmt.span.with_hi(self.prev_token.span.hi())
                } else {
                    stmt.span
                };
                if let Ok(snippet) = self.span_to_snippet(stmt_span) {
                    e.span_suggestion(
                        stmt_span,
                        "try placing this code inside a block",
                        format!("{{ {} }}", snippet),
                        // Speculative; has been misleading in the past (#46836).
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            Err(mut e) => {
                self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
                e.cancel();
            }
            _ => {}
        }
        e.span_label(sp, "expected `{`");
        Err(e)
    }

    /// Parses a block. Inner attributes are allowed.
    pub(super) fn parse_inner_attrs_and_block(
        &mut self,
    ) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
        self.parse_block_common(self.token.span, BlockCheckMode::Default)
    }

    /// Parses a block. Inner attributes are allowed.
    pub(super) fn parse_block_common(
        &mut self,
        lo: Span,
        blk_mode: BlockCheckMode,
    ) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
        maybe_whole!(self, NtBlock, |x| (Vec::new(), x));

        if !self.eat(&token::OpenDelim(token::Brace)) {
            return self.error_block_no_opening_brace();
        }

        Ok((self.parse_inner_attributes()?, self.parse_block_tail(lo, blk_mode)?))
    }

    /// Parses the rest of a block expression or function body.
    /// Precondition: already parsed the '{'.
    fn parse_block_tail(&mut self, lo: Span, s: BlockCheckMode) -> PResult<'a, P<Block>> {
        let mut stmts = vec![];
        while !self.eat(&token::CloseDelim(token::Brace)) {
            if self.token == token::Eof {
                break;
            }
            let stmt = match self.parse_full_stmt() {
                Err(mut err) => {
                    self.maybe_annotate_with_ascription(&mut err, false);
                    err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    Some(self.mk_stmt_err(self.token.span))
                }
                Ok(stmt) => stmt,
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
    pub fn parse_full_stmt(&mut self) -> PResult<'a, Option<Stmt>> {
        // Skip looking for a trailing semicolon when we have an interpolated statement.
        maybe_whole!(self, NtStmt, |x| Some(x));

        let mut stmt = match self.parse_stmt_without_recovery()? {
            Some(stmt) => stmt,
            None => return Ok(None),
        };

        let mut eat_semi = true;
        match stmt.kind {
            // Expression without semicolon.
            StmtKind::Expr(ref expr)
                if self.token != token::Eof && classify::expr_requires_semi_to_be_stmt(expr) =>
            {
                // Just check for errors and recover; do not eat semicolon yet.
                if let Err(mut e) =
                    self.expect_one_of(&[], &[token::Semi, token::CloseDelim(token::Brace)])
                {
                    if let TokenKind::DocComment(..) = self.token.kind {
                        if let Ok(snippet) = self.span_to_snippet(self.token.span) {
                            let sp = self.token.span;
                            let marker = &snippet[..3];
                            let (comment_marker, doc_comment_marker) = marker.split_at(2);

                            e.span_suggestion(
                                sp.with_hi(sp.lo() + BytePos(marker.len() as u32)),
                                &format!(
                                    "add a space before `{}` to use a regular comment",
                                    doc_comment_marker,
                                ),
                                format!("{} {}", comment_marker, doc_comment_marker),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    e.emit();
                    self.recover_stmt();
                    // Don't complain about type errors in body tail after parse error (#57383).
                    let sp = expr.span.to(self.prev_token.span);
                    stmt.kind = StmtKind::Expr(self.mk_expr_err(sp));
                }
            }
            StmtKind::Local(..) => {
                self.expect_semi()?;
                eat_semi = false;
            }
            StmtKind::Empty => eat_semi = false,
            _ => {}
        }

        if eat_semi && self.eat(&token::Semi) {
            stmt = stmt.add_trailing_semicolon();
        }
        stmt.span = stmt.span.to(self.prev_token.span);
        Ok(Some(stmt))
    }

    pub(super) fn mk_block(&self, stmts: Vec<Stmt>, rules: BlockCheckMode, span: Span) -> P<Block> {
        P(Block { stmts, id: DUMMY_NODE_ID, rules, span })
    }

    pub(super) fn mk_stmt(&self, span: Span, kind: StmtKind) -> Stmt {
        Stmt { id: DUMMY_NODE_ID, kind, span }
    }

    fn mk_stmt_err(&self, span: Span) -> Stmt {
        self.mk_stmt(span, StmtKind::Expr(self.mk_expr_err(span)))
    }

    pub(super) fn mk_block_err(&self, span: Span) -> P<Block> {
        self.mk_block(vec![self.mk_stmt_err(span)], BlockCheckMode::Default, span)
    }
}
