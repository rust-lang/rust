use super::{Parser, Restrictions, PrevTokenKind, SemiColonMode, BlockMode};
use super::expr::LhsExpr;
use super::path::PathStyle;
use super::pat::GateOr;
use super::diagnostics::Error;
use crate::maybe_whole;
use crate::DirectoryOwnership;

use rustc_errors::{PResult, Applicability};
use syntax::ThinVec;
use syntax::ptr::P;
use syntax::ast;
use syntax::ast::{DUMMY_NODE_ID, Stmt, StmtKind, Local, Block, BlockCheckMode, Expr, ExprKind};
use syntax::ast::{Attribute, AttrStyle, VisibilityKind, MacStmtStyle, Mac};
use syntax::util::classify;
use syntax::token;
use syntax_pos::source_map::{respan, Span};
use syntax_pos::symbol::{kw, sym};

use std::mem;

impl<'a> Parser<'a> {
    /// Parses a statement. This stops just before trailing semicolons on everything but items.
    /// e.g., a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    pub fn parse_stmt(&mut self) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_(true))
    }

    fn parse_stmt_(&mut self, macro_legacy_warnings: bool) -> Option<Stmt> {
        self.parse_stmt_without_recovery(macro_legacy_warnings).unwrap_or_else(|mut e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        })
    }

    fn parse_stmt_without_recovery(
        &mut self,
        macro_legacy_warnings: bool,
    ) -> PResult<'a, Option<Stmt>> {
        maybe_whole!(self, NtStmt, |x| Some(x));

        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        Ok(Some(if self.eat_keyword(kw::Let) {
            Stmt {
                id: DUMMY_NODE_ID,
                kind: StmtKind::Local(self.parse_local(attrs.into())?),
                span: lo.to(self.prev_span),
            }
        } else if let Some(macro_def) = self.eat_macro_def(
            &attrs,
            &respan(lo, VisibilityKind::Inherited),
            lo,
        )? {
            Stmt {
                id: DUMMY_NODE_ID,
                kind: StmtKind::Item(macro_def),
                span: lo.to(self.prev_span),
            }
        // Starts like a simple path, being careful to avoid contextual keywords
        // such as a union items, item with `crate` visibility or auto trait items.
        // Our goal here is to parse an arbitrary path `a::b::c` but not something that starts
        // like a path (1 token), but it fact not a path.
        // `union::b::c` - path, `union U { ... }` - not a path.
        // `crate::b::c` - path, `crate struct S;` - not a path.
        } else if self.token.is_path_start() &&
                  !self.token.is_qpath_start() &&
                  !self.is_union_item() &&
                  !self.is_crate_vis() &&
                  !self.is_auto_trait_item() &&
                  !self.is_async_fn() {
            let path = self.parse_path(PathStyle::Expr)?;

            if !self.eat(&token::Not) {
                let expr = if self.check(&token::OpenDelim(token::Brace)) {
                    self.parse_struct_expr(lo, path, ThinVec::new())?
                } else {
                    let hi = self.prev_span;
                    self.mk_expr(lo.to(hi), ExprKind::Path(None, path), ThinVec::new())
                };

                let expr = self.with_res(Restrictions::STMT_EXPR, |this| {
                    let expr = this.parse_dot_or_call_expr_with(expr, lo, attrs.into())?;
                    this.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(expr))
                })?;

                return Ok(Some(Stmt {
                    id: DUMMY_NODE_ID,
                    kind: StmtKind::Expr(expr),
                    span: lo.to(self.prev_span),
                }));
            }

            let args = self.parse_mac_args()?;
            let delim = args.delim();
            let hi = self.prev_span;

            let style = if delim == token::Brace {
                MacStmtStyle::Braces
            } else {
                MacStmtStyle::NoBraces
            };

            let mac = Mac {
                path,
                args,
                prior_type_ascription: self.last_type_ascription,
            };
            let kind = if delim == token::Brace ||
                          self.token == token::Semi || self.token == token::Eof {
                StmtKind::Mac(P((mac, style, attrs.into())))
            }
            // We used to incorrectly stop parsing macro-expanded statements here.
            // If the next token will be an error anyway but could have parsed with the
            // earlier behavior, stop parsing here and emit a warning to avoid breakage.
            else if macro_legacy_warnings && self.token.can_begin_expr() &&
                match self.token.kind {
                    // These can continue an expression, so we can't stop parsing and warn.
                    token::OpenDelim(token::Paren) | token::OpenDelim(token::Bracket) |
                    token::BinOp(token::Minus) | token::BinOp(token::Star) |
                    token::BinOp(token::And) | token::BinOp(token::Or) |
                    token::AndAnd | token::OrOr |
                    token::DotDot | token::DotDotDot | token::DotDotEq => false,
                    _ => true,
                }
            {
                self.warn_missing_semicolon();
                StmtKind::Mac(P((mac, style, attrs.into())))
            } else {
                let e = self.mk_expr(lo.to(hi), ExprKind::Mac(mac), ThinVec::new());
                let e = self.maybe_recover_from_bad_qpath(e, true)?;
                let e = self.parse_dot_or_call_expr_with(e, lo, attrs.into())?;
                let e = self.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(e))?;
                StmtKind::Expr(e)
            };
            Stmt {
                id: DUMMY_NODE_ID,
                span: lo.to(hi),
                kind,
            }
        } else {
            // FIXME: Bad copy of attrs
            let old_directory_ownership =
                mem::replace(&mut self.directory.ownership, DirectoryOwnership::UnownedViaBlock);
            let item = self.parse_item_(attrs.clone(), false, true)?;
            self.directory.ownership = old_directory_ownership;

            match item {
                Some(i) => Stmt {
                    id: DUMMY_NODE_ID,
                    span: lo.to(i.span),
                    kind: StmtKind::Item(i),
                },
                None => {
                    let unused_attrs = |attrs: &[Attribute], s: &mut Self| {
                        if !attrs.is_empty() {
                            if s.prev_token_kind == PrevTokenKind::DocComment {
                                s.span_fatal_err(s.prev_span, Error::UselessDocComment).emit();
                            } else if attrs.iter().any(|a| a.style == AttrStyle::Outer) {
                                s.span_err(
                                    s.token.span, "expected statement after outer attribute"
                                );
                            }
                        }
                    };

                    // Do not attempt to parse an expression if we're done here.
                    if self.token == token::Semi {
                        unused_attrs(&attrs, self);
                        self.bump();
                        let mut last_semi = lo;
                        while self.token == token::Semi {
                            last_semi = self.token.span;
                            self.bump();
                        }
                        // We are encoding a string of semicolons as an
                        // an empty tuple that spans the excess semicolons
                        // to preserve this info until the lint stage
                        return Ok(Some(Stmt {
                            id: DUMMY_NODE_ID,
                            span: lo.to(last_semi),
                            kind: StmtKind::Semi(self.mk_expr(lo.to(last_semi),
                                ExprKind::Tup(Vec::new()),
                                ThinVec::new()
                            )),
                        }));
                    }

                    if self.token == token::CloseDelim(token::Brace) {
                        unused_attrs(&attrs, self);
                        return Ok(None);
                    }

                    // Remainder are line-expr stmts.
                    let e = self.parse_expr_res(
                        Restrictions::STMT_EXPR, Some(attrs.into()))?;
                    Stmt {
                        id: DUMMY_NODE_ID,
                        span: lo.to(e.span),
                        kind: StmtKind::Expr(e),
                    }
                }
            }
        }))
    }

    /// Parses a local variable declaration.
    fn parse_local(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Local>> {
        let lo = self.prev_span;
        let pat = self.parse_top_pat(GateOr::Yes)?;

        let (err, ty) = if self.eat(&token::Colon) {
            // Save the state of the parser before parsing type normally, in case there is a `:`
            // instead of an `=` typo.
            let parser_snapshot_before_type = self.clone();
            let colon_sp = self.prev_span;
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
            (Ok(init), None) => {  // init parsed, ty parsed
                init
            }
            (Ok(init), Some((_, colon_sp, mut err))) => {  // init parsed, ty error
                // Could parse the type as if it were the initializer, it is likely there was a
                // typo in the code: `:` instead of `=`. Add suggestion and emit the error.
                err.span_suggestion_short(
                    colon_sp,
                    "use `=` if you meant to assign",
                    " =".to_string(),
                    Applicability::MachineApplicable
                );
                err.emit();
                // As this was parsed successfully, continue as if the code has been fixed for the
                // rest of the file. It will still fail due to the emitted error, but we avoid
                // extra noise.
                init
            }
            (Err(mut init_err), Some((snapshot, _, ty_err))) => {  // init error, ty error
                init_err.cancel();
                // Couldn't parse the type nor the initializer, only raise the type error and
                // return to the parser state before parsing the type as the initializer.
                // let x: <parse_error>;
                mem::replace(self, snapshot);
                return Err(ty_err);
            }
            (Err(err), None) => {  // init error, ty parsed
                // Couldn't parse the initializer and we're not attempting to recover a failed
                // parse of the type, return the error.
                return Err(err);
            }
        };
        let hi = if self.token == token::Semi {
            self.token.span
        } else {
            self.prev_span
        };
        Ok(P(ast::Local {
            ty,
            pat,
            init,
            id: DUMMY_NODE_ID,
            span: lo.to(hi),
            attrs,
        }))
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

    fn is_auto_trait_item(&self) -> bool {
        // auto trait
        (self.token.is_keyword(kw::Auto) &&
            self.is_keyword_ahead(1, &[kw::Trait]))
        || // unsafe auto trait
        (self.token.is_keyword(kw::Unsafe) &&
         self.is_keyword_ahead(1, &[kw::Auto]) &&
         self.is_keyword_ahead(2, &[kw::Trait]))
    }

    /// Parses a block. No inner attributes are allowed.
    pub fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        maybe_whole!(self, NtBlock, |x| x);

        let lo = self.token.span;

        if !self.eat(&token::OpenDelim(token::Brace)) {
            let sp = self.token.span;
            let tok = self.this_token_descr();
            let mut e = self.span_fatal(sp, &format!("expected `{{`, found {}", tok));
            let do_not_suggest_help =
                self.token.is_keyword(kw::In) || self.token == token::Colon;

            if self.token.is_ident_named(sym::and) {
                e.span_suggestion_short(
                    self.token.span,
                    "use `&&` instead of `and` for the boolean operator",
                    "&&".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            if self.token.is_ident_named(sym::or) {
                e.span_suggestion_short(
                    self.token.span,
                    "use `||` instead of `or` for the boolean operator",
                    "||".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }

            // Check to see if the user has written something like
            //
            //    if (cond)
            //      bar;
            //
            // which is valid in other languages, but not Rust.
            match self.parse_stmt_without_recovery(false) {
                Ok(Some(stmt)) => {
                    if self.look_ahead(1, |t| t == &token::OpenDelim(token::Brace))
                        || do_not_suggest_help {
                        // If the next token is an open brace (e.g., `if a b {`), the place-
                        // inside-a-block suggestion would be more likely wrong than right.
                        e.span_label(sp, "expected `{`");
                        return Err(e);
                    }
                    let mut stmt_span = stmt.span;
                    // Expand the span to include the semicolon, if it exists.
                    if self.eat(&token::Semi) {
                        stmt_span = stmt_span.with_hi(self.prev_span.hi());
                    }
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
                _ => ()
            }
            e.span_label(sp, "expected `{`");
            return Err(e);
        }

        self.parse_block_tail(lo, BlockCheckMode::Default)
    }

    /// Parses a block. Inner attributes are allowed.
    pub(super) fn parse_inner_attrs_and_block(
        &mut self
    ) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
        maybe_whole!(self, NtBlock, |x| (Vec::new(), x));

        let lo = self.token.span;
        self.expect(&token::OpenDelim(token::Brace))?;
        Ok((self.parse_inner_attributes()?,
            self.parse_block_tail(lo, BlockCheckMode::Default)?))
    }

    /// Parses the rest of a block expression or function body.
    /// Precondition: already parsed the '{'.
    pub(super) fn parse_block_tail(
        &mut self,
        lo: Span,
        s: BlockCheckMode
    ) -> PResult<'a, P<Block>> {
        let mut stmts = vec![];
        while !self.eat(&token::CloseDelim(token::Brace)) {
            if self.token == token::Eof {
                break;
            }
            let stmt = match self.parse_full_stmt(false) {
                Err(mut err) => {
                    self.maybe_annotate_with_ascription(&mut err, false);
                    err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    Some(Stmt {
                        id: DUMMY_NODE_ID,
                        kind: StmtKind::Expr(self.mk_expr_err(self.token.span)),
                        span: self.token.span,
                    })
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
        Ok(P(ast::Block {
            stmts,
            id: DUMMY_NODE_ID,
            rules: s,
            span: lo.to(self.prev_span),
        }))
    }

    /// Parses a statement, including the trailing semicolon.
    pub fn parse_full_stmt(&mut self, macro_legacy_warnings: bool) -> PResult<'a, Option<Stmt>> {
        // Skip looking for a trailing semicolon when we have an interpolated statement.
        maybe_whole!(self, NtStmt, |x| Some(x));

        let mut stmt = match self.parse_stmt_without_recovery(macro_legacy_warnings)? {
            Some(stmt) => stmt,
            None => return Ok(None),
        };

        let mut eat_semi = true;
        match stmt.kind {
            StmtKind::Expr(ref expr) if self.token != token::Eof => {
                // expression without semicolon
                if classify::expr_requires_semi_to_be_stmt(expr) {
                    // Just check for errors and recover; do not eat semicolon yet.
                    if let Err(mut e) =
                        self.expect_one_of(&[], &[token::Semi, token::CloseDelim(token::Brace)])
                    {
                        e.emit();
                        self.recover_stmt();
                        // Don't complain about type errors in body tail after parse error (#57383).
                        let sp = expr.span.to(self.prev_span);
                        stmt.kind = StmtKind::Expr(self.mk_expr_err(sp));
                    }
                }
            }
            StmtKind::Local(..) => {
                // We used to incorrectly allow a macro-expanded let statement to lack a semicolon.
                if macro_legacy_warnings && self.token != token::Semi {
                    self.warn_missing_semicolon();
                } else {
                    self.expect_semi()?;
                    eat_semi = false;
                }
            }
            _ => {}
        }

        if eat_semi && self.eat(&token::Semi) {
            stmt = stmt.add_trailing_semicolon();
        }
        stmt.span = stmt.span.to(self.prev_span);
        Ok(Some(stmt))
    }

    fn warn_missing_semicolon(&self) {
        self.diagnostic().struct_span_warn(self.token.span, {
            &format!("expected `;`, found {}", self.this_token_descr())
        }).note({
            "this was erroneously allowed and will become a hard error in a future release"
        }).emit();
    }
}
