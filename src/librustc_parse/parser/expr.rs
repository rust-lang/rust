use super::{Parser, Restrictions, PrevTokenKind, TokenType, PathStyle, BlockMode};
use super::{SemiColonMode, SeqSep, TokenExpectType};
use super::pat::{GateOr, PARAM_EXPECTED};
use super::diagnostics::Error;
use crate::maybe_recover_from_interpolated_ty_qpath;

use syntax::ast::{
    self, DUMMY_NODE_ID, Attribute, AttrStyle, Ident, CaptureBy, BlockCheckMode,
    Expr, ExprKind, RangeLimits, Label, Movability, IsAsync, Arm, Ty, TyKind,
    FunctionRetTy, Param, FnDecl, BinOpKind, BinOp, UnOp, Mac, AnonConst, Field, Lit,
};
use syntax::token::{self, Token, TokenKind};
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::source_map::{self, Span};
use syntax::util::classify;
use syntax::util::literal::LitError;
use syntax::util::parser::{AssocOp, Fixity, prec_let_scrutinee_needs_par};
use syntax_pos::symbol::{kw, sym};
use syntax_pos::Symbol;
use errors::{PResult, Applicability};
use std::mem;
use rustc_data_structures::thin_vec::ThinVec;

/// Possibly accepts an `token::Interpolated` expression (a pre-parsed expression
/// dropped into the token stream, which happens while parsing the result of
/// macro expansion). Placement of these is not as complex as I feared it would
/// be. The important thing is to make sure that lookahead doesn't balk at
/// `token::Interpolated` tokens.
macro_rules! maybe_whole_expr {
    ($p:expr) => {
        if let token::Interpolated(nt) = &$p.token.kind {
            match &**nt {
                token::NtExpr(e) | token::NtLiteral(e) => {
                    let e = e.clone();
                    $p.bump();
                    return Ok(e);
                }
                token::NtPath(path) => {
                    let path = path.clone();
                    $p.bump();
                    return Ok($p.mk_expr(
                        $p.token.span, ExprKind::Path(None, path), ThinVec::new()
                    ));
                }
                token::NtBlock(block) => {
                    let block = block.clone();
                    $p.bump();
                    return Ok($p.mk_expr(
                        $p.token.span, ExprKind::Block(block, None), ThinVec::new()
                    ));
                }
                // N.B., `NtIdent(ident)` is normalized to `Ident` in `fn bump`.
                _ => {},
            };
        }
    }
}

#[derive(Debug)]
pub(super) enum LhsExpr {
    NotYetParsed,
    AttributesParsed(ThinVec<Attribute>),
    AlreadyParsed(P<Expr>),
}

impl From<Option<ThinVec<Attribute>>> for LhsExpr {
    /// Converts `Some(attrs)` into `LhsExpr::AttributesParsed(attrs)`
    /// and `None` into `LhsExpr::NotYetParsed`.
    ///
    /// This conversion does not allocate.
    fn from(o: Option<ThinVec<Attribute>>) -> Self {
        if let Some(attrs) = o {
            LhsExpr::AttributesParsed(attrs)
        } else {
            LhsExpr::NotYetParsed
        }
    }
}

impl From<P<Expr>> for LhsExpr {
    /// Converts the `expr: P<Expr>` into `LhsExpr::AlreadyParsed(expr)`.
    ///
    /// This conversion does not allocate.
    fn from(expr: P<Expr>) -> Self {
        LhsExpr::AlreadyParsed(expr)
    }
}

impl<'a> Parser<'a> {
    /// Parses an expression.
    #[inline]
    pub fn parse_expr(&mut self) -> PResult<'a, P<Expr>> {
        self.parse_expr_res(Restrictions::empty(), None)
    }

    fn parse_paren_expr_seq(&mut self) -> PResult<'a, Vec<P<Expr>>> {
        self.parse_paren_comma_seq(|p| {
            match p.parse_expr() {
                Ok(expr) => Ok(expr),
                Err(mut err) => match p.token.kind {
                    token::Ident(name, false)
                    if name == kw::Underscore && p.look_ahead(1, |t| {
                        t == &token::Comma
                    }) => {
                        // Special-case handling of `foo(_, _, _)`
                        err.emit();
                        let sp = p.token.span;
                        p.bump();
                        Ok(p.mk_expr(sp, ExprKind::Err, ThinVec::new()))
                    }
                    _ => Err(err),
                },
            }
        }).map(|(r, _)| r)
    }

    /// Parses an expression, subject to the given restrictions.
    #[inline]
    pub(super) fn parse_expr_res(
        &mut self,
        r: Restrictions,
        already_parsed_attrs: Option<ThinVec<Attribute>>
    ) -> PResult<'a, P<Expr>> {
        self.with_res(r, |this| this.parse_assoc_expr(already_parsed_attrs))
    }

    /// Parses an associative expression.
    ///
    /// This parses an expression accounting for associativity and precedence of the operators in
    /// the expression.
    #[inline]
    fn parse_assoc_expr(
        &mut self,
        already_parsed_attrs: Option<ThinVec<Attribute>>,
    ) -> PResult<'a, P<Expr>> {
        self.parse_assoc_expr_with(0, already_parsed_attrs.into())
    }

    /// Parses an associative expression with operators of at least `min_prec` precedence.
    pub(super) fn parse_assoc_expr_with(
        &mut self,
        min_prec: usize,
        lhs: LhsExpr,
    ) -> PResult<'a, P<Expr>> {
        let mut lhs = if let LhsExpr::AlreadyParsed(expr) = lhs {
            expr
        } else {
            let attrs = match lhs {
                LhsExpr::AttributesParsed(attrs) => Some(attrs),
                _ => None,
            };
            if [token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token.kind) {
                return self.parse_prefix_range_expr(attrs);
            } else {
                self.parse_prefix_expr(attrs)?
            }
        };
        let last_type_ascription_set = self.last_type_ascription.is_some();

        match (self.expr_is_complete(&lhs), AssocOp::from_token(&self.token)) {
            (true, None) => {
                self.last_type_ascription = None;
                // Semi-statement forms are odd. See https://github.com/rust-lang/rust/issues/29071
                return Ok(lhs);
            }
            (false, _) => {} // continue parsing the expression
            // An exhaustive check is done in the following block, but these are checked first
            // because they *are* ambiguous but also reasonable looking incorrect syntax, so we
            // want to keep their span info to improve diagnostics in these cases in a later stage.
            (true, Some(AssocOp::Multiply)) | // `{ 42 } *foo = bar;` or `{ 42 } * 3`
            (true, Some(AssocOp::Subtract)) | // `{ 42 } -5`
            (true, Some(AssocOp::LAnd)) | // `{ 42 } &&x` (#61475)
            (true, Some(AssocOp::Add)) // `{ 42 } + 42
            // If the next token is a keyword, then the tokens above *are* unambiguously incorrect:
            // `if x { a } else { b } && if y { c } else { d }`
            if !self.look_ahead(1, |t| t.is_reserved_ident()) => {
                self.last_type_ascription = None;
                // These cases are ambiguous and can't be identified in the parser alone
                let sp = self.sess.source_map().start_point(self.token.span);
                self.sess.ambiguous_block_expr_parse.borrow_mut().insert(sp, lhs.span);
                return Ok(lhs);
            }
            (true, Some(ref op)) if !op.can_continue_expr_unambiguously() => {
                self.last_type_ascription = None;
                return Ok(lhs);
            }
            (true, Some(_)) => {
                // We've found an expression that would be parsed as a statement, but the next
                // token implies this should be parsed as an expression.
                // For example: `if let Some(x) = x { x } else { 0 } / 2`
                let mut err = self.struct_span_err(self.token.span, &format!(
                    "expected expression, found `{}`",
                    pprust::token_to_string(&self.token),
                ));
                err.span_label(self.token.span, "expected expression");
                self.sess.expr_parentheses_needed(
                    &mut err,
                    lhs.span,
                    Some(pprust::expr_to_string(&lhs),
                ));
                err.emit();
            }
        }
        self.expected_tokens.push(TokenType::Operator);
        while let Some(op) = AssocOp::from_token(&self.token) {

            // Adjust the span for interpolated LHS to point to the `$lhs` token and not to what
            // it refers to. Interpolated identifiers are unwrapped early and never show up here
            // as `PrevTokenKind::Interpolated` so if LHS is a single identifier we always process
            // it as "interpolated", it doesn't change the answer for non-interpolated idents.
            let lhs_span = match (self.prev_token_kind, &lhs.kind) {
                (PrevTokenKind::Interpolated, _) => self.prev_span,
                (PrevTokenKind::Ident, &ExprKind::Path(None, ref path))
                    if path.segments.len() == 1 => self.prev_span,
                _ => lhs.span,
            };

            let cur_op_span = self.token.span;
            let restrictions = if op.is_assign_like() {
                self.restrictions & Restrictions::NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            // Check for deprecated `...` syntax
            if self.token == token::DotDotDot && op == AssocOp::DotDotEq {
                self.err_dotdotdot_syntax(self.token.span);
            }

            if self.token == token::LArrow {
                self.err_larrow_operator(self.token.span);
            }

            self.bump();
            if op.is_comparison() {
                if let Some(expr) = self.check_no_chained_comparison(&lhs, &op)? {
                    return Ok(expr);
                }
            }
            // Special cases:
            if op == AssocOp::As {
                lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Cast)?;
                continue
            } else if op == AssocOp::Colon {
                let maybe_path = self.could_ascription_be_path(&lhs.kind);
                self.last_type_ascription = Some((self.prev_span, maybe_path));

                lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Type)?;
                self.sess.gated_spans.gate(sym::type_ascription, lhs.span);
                continue
            } else if op == AssocOp::DotDot || op == AssocOp::DotDotEq {
                // If we didn’t have to handle `x..`/`x..=`, it would be pretty easy to
                // generalise it to the Fixity::None code.
                //
                // We have 2 alternatives here: `x..y`/`x..=y` and `x..`/`x..=` The other
                // two variants are handled with `parse_prefix_range_expr` call above.
                let rhs = if self.is_at_start_of_range_notation_rhs() {
                    Some(self.parse_assoc_expr_with(prec + 1, LhsExpr::NotYetParsed)?)
                } else {
                    None
                };
                let (lhs_span, rhs_span) = (lhs.span, if let Some(ref x) = rhs {
                    x.span
                } else {
                    cur_op_span
                });
                let limits = if op == AssocOp::DotDot {
                    RangeLimits::HalfOpen
                } else {
                    RangeLimits::Closed
                };

                let r = self.mk_range(Some(lhs), rhs, limits)?;
                lhs = self.mk_expr(lhs_span.to(rhs_span), r, ThinVec::new());
                break
            }

            let fixity = op.fixity();
            let prec_adjustment = match fixity {
                Fixity::Right => 0,
                Fixity::Left => 1,
                // We currently have no non-associative operators that are not handled above by
                // the special cases. The code is here only for future convenience.
                Fixity::None => 1,
            };
            let rhs = self.with_res(
                restrictions - Restrictions::STMT_EXPR,
                |this| this.parse_assoc_expr_with(prec + prec_adjustment, LhsExpr::NotYetParsed)
            )?;

            // Make sure that the span of the parent node is larger than the span of lhs and rhs,
            // including the attributes.
            let lhs_span = lhs
                .attrs
                .iter()
                .filter(|a| a.style == AttrStyle::Outer)
                .next()
                .map_or(lhs_span, |a| a.span);
            let span = lhs_span.to(rhs.span);
            lhs = match op {
                AssocOp::Add | AssocOp::Subtract | AssocOp::Multiply | AssocOp::Divide |
                AssocOp::Modulus | AssocOp::LAnd | AssocOp::LOr | AssocOp::BitXor |
                AssocOp::BitAnd | AssocOp::BitOr | AssocOp::ShiftLeft | AssocOp::ShiftRight |
                AssocOp::Equal | AssocOp::Less | AssocOp::LessEqual | AssocOp::NotEqual |
                AssocOp::Greater | AssocOp::GreaterEqual => {
                    let ast_op = op.to_ast_binop().unwrap();
                    let binary = self.mk_binary(source_map::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(span, binary, ThinVec::new())
                }
                AssocOp::Assign => self.mk_expr(span, ExprKind::Assign(lhs, rhs), ThinVec::new()),
                AssocOp::AssignOp(k) => {
                    let aop = match k {
                        token::Plus =>    BinOpKind::Add,
                        token::Minus =>   BinOpKind::Sub,
                        token::Star =>    BinOpKind::Mul,
                        token::Slash =>   BinOpKind::Div,
                        token::Percent => BinOpKind::Rem,
                        token::Caret =>   BinOpKind::BitXor,
                        token::And =>     BinOpKind::BitAnd,
                        token::Or =>      BinOpKind::BitOr,
                        token::Shl =>     BinOpKind::Shl,
                        token::Shr =>     BinOpKind::Shr,
                    };
                    let aopexpr = self.mk_assign_op(source_map::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(span, aopexpr, ThinVec::new())
                }
                AssocOp::As | AssocOp::Colon | AssocOp::DotDot | AssocOp::DotDotEq => {
                    self.bug("AssocOp should have been handled by special case")
                }
            };

            if let Fixity::None = fixity { break }
        }
        if last_type_ascription_set {
            self.last_type_ascription = None;
        }
        Ok(lhs)
    }

    /// Checks if this expression is a successfully parsed statement.
    fn expr_is_complete(&self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::STMT_EXPR) &&
            !classify::expr_requires_semi_to_be_stmt(e)
    }

    fn is_at_start_of_range_notation_rhs(&self) -> bool {
        if self.token.can_begin_expr() {
            // Parse `for i in 1.. { }` as infinite loop, not as `for i in (1..{})`.
            if self.token == token::OpenDelim(token::Brace) {
                return !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
            }
            true
        } else {
            false
        }
    }

    /// Parses prefix-forms of range notation: `..expr`, `..`, `..=expr`.
    fn parse_prefix_range_expr(
        &mut self,
        already_parsed_attrs: Option<ThinVec<Attribute>>
    ) -> PResult<'a, P<Expr>> {
        // Check for deprecated `...` syntax.
        if self.token == token::DotDotDot {
            self.err_dotdotdot_syntax(self.token.span);
        }

        debug_assert!([token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token.kind),
                      "parse_prefix_range_expr: token {:?} is not DotDot/DotDotEq",
                      self.token);
        let tok = self.token.clone();
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.token.span;
        let mut hi = self.token.span;
        self.bump();
        let opt_end = if self.is_at_start_of_range_notation_rhs() {
            // RHS must be parsed with more associativity than the dots.
            let next_prec = AssocOp::from_token(&tok).unwrap().precedence() + 1;
            Some(self.parse_assoc_expr_with(next_prec, LhsExpr::NotYetParsed)
                .map(|x| {
                    hi = x.span;
                    x
                })?)
        } else {
            None
        };
        let limits = if tok == token::DotDot {
            RangeLimits::HalfOpen
        } else {
            RangeLimits::Closed
        };

        let r = self.mk_range(None, opt_end, limits)?;
        Ok(self.mk_expr(lo.to(hi), r, attrs))
    }

    /// Parses a prefix-unary-operator expr.
    fn parse_prefix_expr(
        &mut self,
        already_parsed_attrs: Option<ThinVec<Attribute>>
    ) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.token.span;
        // Note: when adding new unary operators, don't forget to adjust TokenKind::can_begin_expr()
        let (hi, ex) = match self.token.kind {
            token::Not => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Not, e))
            }
            // Suggest `!` for bitwise negation when encountering a `~`
            token::Tilde => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                let span_of_tilde = lo;
                self.struct_span_err(span_of_tilde, "`~` cannot be used as a unary operator")
                    .span_suggestion_short(
                        span_of_tilde,
                        "use `!` to perform bitwise not",
                        "!".to_owned(),
                        Applicability::MachineApplicable
                    )
                    .emit();
                (lo.to(span), self.mk_unary(UnOp::Not, e))
            }
            token::BinOp(token::Minus) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Neg, e))
            }
            token::BinOp(token::Star) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Deref, e))
            }
            token::BinOp(token::And) | token::AndAnd => {
                self.parse_address_of(lo)?
            }
            token::Ident(..) if self.token.is_keyword(kw::Box) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                let span = lo.to(span);
                self.sess.gated_spans.gate(sym::box_syntax, span);
                (span, ExprKind::Box(e))
            }
            token::Ident(..) if self.token.is_ident_named(sym::not) => {
                // `not` is just an ordinary identifier in Rust-the-language,
                // but as `rustc`-the-compiler, we can issue clever diagnostics
                // for confused users who really want to say `!`
                let token_cannot_continue_expr = |t: &Token| match t.kind {
                    // These tokens can start an expression after `!`, but
                    // can't continue an expression after an ident
                    token::Ident(name, is_raw) => token::ident_can_begin_expr(name, t.span, is_raw),
                    token::Literal(..) | token::Pound => true,
                    _ => t.is_whole_expr(),
                };
                let cannot_continue_expr = self.look_ahead(1, token_cannot_continue_expr);
                if cannot_continue_expr {
                    self.bump();
                    // Emit the error ...
                    self.struct_span_err(
                        self.token.span,
                        &format!("unexpected {} after identifier",self.this_token_descr())
                    )
                    .span_suggestion_short(
                        // Span the `not` plus trailing whitespace to avoid
                        // trailing whitespace after the `!` in our suggestion
                        self.sess.source_map()
                            .span_until_non_whitespace(lo.to(self.token.span)),
                        "use `!` to perform logical negation",
                        "!".to_owned(),
                        Applicability::MachineApplicable
                    )
                    .emit();
                    // —and recover! (just as if we were in the block
                    // for the `token::Not` arm)
                    let e = self.parse_prefix_expr(None);
                    let (span, e) = self.interpolated_or_expr_span(e)?;
                    (lo.to(span), self.mk_unary(UnOp::Not, e))
                } else {
                    return self.parse_dot_or_call_expr(Some(attrs));
                }
            }
            _ => { return self.parse_dot_or_call_expr(Some(attrs)); }
        };
        return Ok(self.mk_expr(lo.to(hi), ex, attrs));
    }

    /// Returns the span of expr, if it was not interpolated or the span of the interpolated token.
    fn interpolated_or_expr_span(
        &self,
        expr: PResult<'a, P<Expr>>,
    ) -> PResult<'a, (Span, P<Expr>)> {
        expr.map(|e| {
            if self.prev_token_kind == PrevTokenKind::Interpolated {
                (self.prev_span, e)
            } else {
                (e.span, e)
            }
        })
    }

    fn parse_assoc_op_cast(&mut self, lhs: P<Expr>, lhs_span: Span,
                           expr_kind: fn(P<Expr>, P<Ty>) -> ExprKind)
                           -> PResult<'a, P<Expr>> {
        let mk_expr = |this: &mut Self, rhs: P<Ty>| {
            this.mk_expr(lhs_span.to(rhs.span), expr_kind(lhs, rhs), ThinVec::new())
        };

        // Save the state of the parser before parsing type normally, in case there is a
        // LessThan comparison after this cast.
        let parser_snapshot_before_type = self.clone();
        match self.parse_ty_no_plus() {
            Ok(rhs) => {
                Ok(mk_expr(self, rhs))
            }
            Err(mut type_err) => {
                // Rewind to before attempting to parse the type with generics, to recover
                // from situations like `x as usize < y` in which we first tried to parse
                // `usize < y` as a type with generic arguments.
                let parser_snapshot_after_type = self.clone();
                mem::replace(self, parser_snapshot_before_type);

                match self.parse_path(PathStyle::Expr) {
                    Ok(path) => {
                        let (op_noun, op_verb) = match self.token.kind {
                            token::Lt => ("comparison", "comparing"),
                            token::BinOp(token::Shl) => ("shift", "shifting"),
                            _ => {
                                // We can end up here even without `<` being the next token, for
                                // example because `parse_ty_no_plus` returns `Err` on keywords,
                                // but `parse_path` returns `Ok` on them due to error recovery.
                                // Return original error and parser state.
                                mem::replace(self, parser_snapshot_after_type);
                                return Err(type_err);
                            }
                        };

                        // Successfully parsed the type path leaving a `<` yet to parse.
                        type_err.cancel();

                        // Report non-fatal diagnostics, keep `x as usize` as an expression
                        // in AST and continue parsing.
                        let msg = format!(
                            "`<` is interpreted as a start of generic arguments for `{}`, not a {}",
                            pprust::path_to_string(&path),
                            op_noun,
                        );
                        let span_after_type = parser_snapshot_after_type.token.span;
                        let expr = mk_expr(self, P(Ty {
                            span: path.span,
                            kind: TyKind::Path(None, path),
                            id: DUMMY_NODE_ID,
                        }));

                        let expr_str = self.span_to_snippet(expr.span)
                            .unwrap_or_else(|_| pprust::expr_to_string(&expr));

                        self.struct_span_err(self.token.span, &msg)
                            .span_label(
                                self.look_ahead(1, |t| t.span).to(span_after_type),
                                "interpreted as generic arguments"
                            )
                            .span_label(self.token.span, format!("not interpreted as {}", op_noun))
                            .span_suggestion(
                                expr.span,
                                &format!("try {} the cast value", op_verb),
                                format!("({})", expr_str),
                                Applicability::MachineApplicable,
                            )
                            .emit();

                        Ok(expr)
                    }
                    Err(mut path_err) => {
                        // Couldn't parse as a path, return original error and parser state.
                        path_err.cancel();
                        mem::replace(self, parser_snapshot_after_type);
                        Err(type_err)
                    }
                }
            }
        }
    }

    /// Parse `& mut? <expr>` or `& raw [ const | mut ] <expr>`.
    fn parse_address_of(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        self.expect_and()?;
        let (k, m) = if self.check_keyword(kw::Raw)
            && self.look_ahead(1, Token::is_mutability)
        {
            let found_raw = self.eat_keyword(kw::Raw);
            assert!(found_raw);
            let mutability = self.parse_const_or_mut().unwrap();
            self.sess.gated_spans.gate(sym::raw_ref_op, lo.to(self.prev_span));
            (ast::BorrowKind::Raw, mutability)
        } else {
            (ast::BorrowKind::Ref, self.parse_mutability())
        };
        let e = self.parse_prefix_expr(None);
        let (span, e) = self.interpolated_or_expr_span(e)?;
        Ok((lo.to(span), ExprKind::AddrOf(k, m, e)))
    }

    /// Parses `a.b` or `a(13)` or `a[4]` or just `a`.
    fn parse_dot_or_call_expr(
        &mut self,
        already_parsed_attrs: Option<ThinVec<Attribute>>,
    ) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;

        let b = self.parse_bottom_expr();
        let (span, b) = self.interpolated_or_expr_span(b)?;
        self.parse_dot_or_call_expr_with(b, span, attrs)
    }

    pub(super) fn parse_dot_or_call_expr_with(
        &mut self,
        e0: P<Expr>,
        lo: Span,
        mut attrs: ThinVec<Attribute>,
    ) -> PResult<'a, P<Expr>> {
        // Stitch the list of outer attributes onto the return value.
        // A little bit ugly, but the best way given the current code
        // structure
        self.parse_dot_or_call_expr_with_(e0, lo).map(|expr|
            expr.map(|mut expr| {
                attrs.extend::<Vec<_>>(expr.attrs.into());
                expr.attrs = attrs;
                match expr.kind {
                    ExprKind::If(..) if !expr.attrs.is_empty() => {
                        // Just point to the first attribute in there...
                        let span = expr.attrs[0].span;
                        self.span_err(span, "attributes are not yet allowed on `if` expressions");
                    }
                    _ => {}
                }
                expr
            })
        )
    }

    fn parse_dot_or_call_expr_with_(&mut self, e0: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        let mut e = e0;
        let mut hi;
        loop {
            // expr?
            while self.eat(&token::Question) {
                let hi = self.prev_span;
                e = self.mk_expr(lo.to(hi), ExprKind::Try(e), ThinVec::new());
            }

            // expr.f
            if self.eat(&token::Dot) {
                match self.token.kind {
                    token::Ident(..) => {
                        e = self.parse_dot_suffix(e, lo)?;
                    }
                    token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) => {
                        let span = self.token.span;
                        self.bump();
                        let field = ExprKind::Field(e, Ident::new(symbol, span));
                        e = self.mk_expr(lo.to(span), field, ThinVec::new());

                        self.expect_no_suffix(span, "a tuple index", suffix);
                    }
                    token::Literal(token::Lit { kind: token::Float, symbol, .. }) => {
                      self.bump();
                      let fstr = symbol.as_str();
                      let msg = format!("unexpected token: `{}`", symbol);
                      let mut err = self.diagnostic().struct_span_err(self.prev_span, &msg);
                      err.span_label(self.prev_span, "unexpected token");
                      if fstr.chars().all(|x| "0123456789.".contains(x)) {
                          let float = match fstr.parse::<f64>().ok() {
                              Some(f) => f,
                              None => continue,
                          };
                          let sugg = pprust::to_string(|s| {
                              s.popen();
                              s.print_expr(&e);
                              s.s.word( ".");
                              s.print_usize(float.trunc() as usize);
                              s.pclose();
                              s.s.word(".");
                              s.s.word(fstr.splitn(2, ".").last().unwrap().to_string())
                          });
                          err.span_suggestion(
                              lo.to(self.prev_span),
                              "try parenthesizing the first index",
                              sugg,
                              Applicability::MachineApplicable
                          );
                      }
                      return Err(err);

                    }
                    _ => {
                        // FIXME Could factor this out into non_fatal_unexpected or something.
                        let actual = self.this_token_to_string();
                        self.span_err(self.token.span, &format!("unexpected token: `{}`", actual));
                    }
                }
                continue;
            }
            if self.expr_is_complete(&e) { break; }
            match self.token.kind {
                // expr(...)
                token::OpenDelim(token::Paren) => {
                    let seq = self.parse_paren_expr_seq().map(|es| {
                        let nd = self.mk_call(e, es);
                        let hi = self.prev_span;
                        self.mk_expr(lo.to(hi), nd, ThinVec::new())
                    });
                    e = self.recover_seq_parse_error(token::Paren, lo, seq);
                }

                // expr[...]
                // Could be either an index expression or a slicing expression.
                token::OpenDelim(token::Bracket) => {
                    self.bump();
                    let ix = self.parse_expr()?;
                    hi = self.token.span;
                    self.expect(&token::CloseDelim(token::Bracket))?;
                    let index = self.mk_index(e, ix);
                    e = self.mk_expr(lo.to(hi), index, ThinVec::new())
                }
                _ => return Ok(e)
            }
        }
        return Ok(e);
    }

    /// Assuming we have just parsed `.`, continue parsing into an expression.
    fn parse_dot_suffix(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        if self.token.span.rust_2018() && self.eat_keyword(kw::Await) {
            return self.mk_await_expr(self_arg, lo);
        }

        let segment = self.parse_path_segment(PathStyle::Expr)?;
        self.check_trailing_angle_brackets(&segment, token::OpenDelim(token::Paren));

        Ok(match self.token.kind {
            token::OpenDelim(token::Paren) => {
                // Method call `expr.f()`
                let mut args = self.parse_paren_expr_seq()?;
                args.insert(0, self_arg);

                let span = lo.to(self.prev_span);
                self.mk_expr(span, ExprKind::MethodCall(segment, args), ThinVec::new())
            }
            _ => {
                // Field access `expr.f`
                if let Some(args) = segment.args {
                    self.span_err(args.span(),
                                  "field expressions may not have generic arguments");
                }

                let span = lo.to(self.prev_span);
                self.mk_expr(span, ExprKind::Field(self_arg, segment.ident), ThinVec::new())
            }
        })
    }

    /// At the bottom (top?) of the precedence hierarchy,
    /// Parses things like parenthesized exprs, macros, `return`, etc.
    ///
    /// N.B., this does not parse outer attributes, and is private because it only works
    /// correctly if called from `parse_dot_or_call_expr()`.
    fn parse_bottom_expr(&mut self) -> PResult<'a, P<Expr>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole_expr!(self);

        // Outer attributes are already parsed and will be
        // added to the return value after the fact.
        //
        // Therefore, prevent sub-parser from parsing
        // attributes by giving them a empty "already-parsed" list.
        let mut attrs = ThinVec::new();

        let lo = self.token.span;
        let mut hi = self.token.span;

        let ex: ExprKind;

        macro_rules! parse_lit {
            () => {
                match self.parse_opt_lit() {
                    Some(literal) => {
                        hi = self.prev_span;
                        ex = ExprKind::Lit(literal);
                    }
                    None => {
                        return Err(self.expected_expression_found());
                    }
                }
            }
        }

        // Note: when adding new syntax here, don't forget to adjust `TokenKind::can_begin_expr()`.
        match self.token.kind {
            // This match arm is a special-case of the `_` match arm below and
            // could be removed without changing functionality, but it's faster
            // to have it here, especially for programs with large constants.
            token::Literal(_) => {
                parse_lit!()
            }
            token::OpenDelim(token::Paren) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                // `(e)` is parenthesized `e`.
                // `(e,)` is a tuple with only one field, `e`.
                let mut es = vec![];
                let mut trailing_comma = false;
                let mut recovered = false;
                while self.token != token::CloseDelim(token::Paren) {
                    es.push(match self.parse_expr() {
                        Ok(es) => es,
                        Err(mut err) => {
                            // Recover from parse error in tuple list.
                            match self.token.kind {
                                token::Ident(name, false)
                                if name == kw::Underscore && self.look_ahead(1, |t| {
                                    t == &token::Comma
                                }) => {
                                    // Special-case handling of `Foo<(_, _, _)>`
                                    err.emit();
                                    let sp = self.token.span;
                                    self.bump();
                                    self.mk_expr(sp, ExprKind::Err, ThinVec::new())
                                }
                                _ => return Ok(
                                    self.recover_seq_parse_error(token::Paren, lo, Err(err)),
                                ),
                            }
                        }
                    });
                    recovered = self.expect_one_of(
                        &[],
                        &[token::Comma, token::CloseDelim(token::Paren)],
                    )?;
                    if self.eat(&token::Comma) {
                        trailing_comma = true;
                    } else {
                        trailing_comma = false;
                        break;
                    }
                }
                if !recovered {
                    self.bump();
                }

                hi = self.prev_span;
                ex = if es.len() == 1 && !trailing_comma {
                    ExprKind::Paren(es.into_iter().nth(0).unwrap())
                } else {
                    ExprKind::Tup(es)
                };
            }
            token::OpenDelim(token::Brace) => {
                return self.parse_block_expr(None, lo, BlockCheckMode::Default, attrs);
            }
            token::BinOp(token::Or) | token::OrOr => {
                return self.parse_closure_expr(attrs);
            }
            token::OpenDelim(token::Bracket) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                if self.eat(&token::CloseDelim(token::Bracket)) {
                    // Empty vector
                    ex = ExprKind::Array(Vec::new());
                } else {
                    // Non-empty vector
                    let first_expr = self.parse_expr()?;
                    if self.eat(&token::Semi) {
                        // Repeating array syntax: `[ 0; 512 ]`
                        let count = AnonConst {
                            id: DUMMY_NODE_ID,
                            value: self.parse_expr()?,
                        };
                        self.expect(&token::CloseDelim(token::Bracket))?;
                        ex = ExprKind::Repeat(first_expr, count);
                    } else if self.eat(&token::Comma) {
                        // Vector with two or more elements
                        let remaining_exprs = self.parse_seq_to_end(
                            &token::CloseDelim(token::Bracket),
                            SeqSep::trailing_allowed(token::Comma),
                            |p| Ok(p.parse_expr()?)
                        )?;
                        let mut exprs = vec![first_expr];
                        exprs.extend(remaining_exprs);
                        ex = ExprKind::Array(exprs);
                    } else {
                        // Vector with one element
                        self.expect(&token::CloseDelim(token::Bracket))?;
                        ex = ExprKind::Array(vec![first_expr]);
                    }
                }
                hi = self.prev_span;
            }
            _ => {
                if self.eat_lt() {
                    let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                    hi = path.span;
                    return Ok(self.mk_expr(lo.to(hi), ExprKind::Path(Some(qself), path), attrs));
                }
                if self.token.is_path_start() {
                    let path = self.parse_path(PathStyle::Expr)?;

                    // `!`, as an operator, is prefix, so we know this isn't that.
                    if self.eat(&token::Not) {
                        // MACRO INVOCATION expression
                        let args = self.parse_mac_args()?;
                        hi = self.prev_span;
                        ex = ExprKind::Mac(Mac {
                            path,
                            args,
                            prior_type_ascription: self.last_type_ascription,
                        });
                    } else if self.check(&token::OpenDelim(token::Brace)) {
                        if let Some(expr) = self.maybe_parse_struct_expr(lo, &path, &attrs) {
                            return expr;
                        } else {
                            hi = path.span;
                            ex = ExprKind::Path(None, path);
                        }
                    } else {
                        hi = path.span;
                        ex = ExprKind::Path(None, path);
                    }

                    let expr = self.mk_expr(lo.to(hi), ex, attrs);
                    return self.maybe_recover_from_bad_qpath(expr, true);
                }
                if self.check_keyword(kw::Move) || self.check_keyword(kw::Static) {
                    return self.parse_closure_expr(attrs);
                }
                if self.eat_keyword(kw::If) {
                    return self.parse_if_expr(attrs);
                }
                if self.eat_keyword(kw::For) {
                    let lo = self.prev_span;
                    return self.parse_for_expr(None, lo, attrs);
                }
                if self.eat_keyword(kw::While) {
                    let lo = self.prev_span;
                    return self.parse_while_expr(None, lo, attrs);
                }
                if let Some(label) = self.eat_label() {
                    let lo = label.ident.span;
                    self.expect(&token::Colon)?;
                    if self.eat_keyword(kw::While) {
                        return self.parse_while_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(kw::For) {
                        return self.parse_for_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(kw::Loop) {
                        return self.parse_loop_expr(Some(label), lo, attrs)
                    }
                    if self.token == token::OpenDelim(token::Brace) {
                        return self.parse_block_expr(Some(label),
                                                     lo,
                                                     BlockCheckMode::Default,
                                                     attrs);
                    }
                    let msg = "expected `while`, `for`, `loop` or `{` after a label";
                    let mut err = self.fatal(msg);
                    err.span_label(self.token.span, msg);
                    return Err(err);
                }
                if self.eat_keyword(kw::Loop) {
                    let lo = self.prev_span;
                    return self.parse_loop_expr(None, lo, attrs);
                }
                if self.eat_keyword(kw::Continue) {
                    let label = self.eat_label();
                    let ex = ExprKind::Continue(label);
                    let hi = self.prev_span;
                    return Ok(self.mk_expr(lo.to(hi), ex, attrs));
                }
                if self.eat_keyword(kw::Match) {
                    let match_sp = self.prev_span;
                    return self.parse_match_expr(attrs).map_err(|mut err| {
                        err.span_label(match_sp, "while parsing this match expression");
                        err
                    });
                }
                if self.eat_keyword(kw::Unsafe) {
                    return self.parse_block_expr(
                        None,
                        lo,
                        BlockCheckMode::Unsafe(ast::UserProvided),
                        attrs);
                }
                if self.is_do_catch_block() {
                    let mut db = self.fatal("found removed `do catch` syntax");
                    db.help("following RFC #2388, the new non-placeholder syntax is `try`");
                    return Err(db);
                }
                if self.is_try_block() {
                    let lo = self.token.span;
                    assert!(self.eat_keyword(kw::Try));
                    return self.parse_try_block(lo, attrs);
                }

                // `Span::rust_2018()` is somewhat expensive; don't get it repeatedly.
                let is_span_rust_2018 = self.token.span.rust_2018();
                if is_span_rust_2018 && self.check_keyword(kw::Async) {
                    return if self.is_async_block() { // Check for `async {` and `async move {`.
                        self.parse_async_block(attrs)
                    } else {
                        self.parse_closure_expr(attrs)
                    };
                }
                if self.eat_keyword(kw::Return) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Ret(Some(e));
                    } else {
                        ex = ExprKind::Ret(None);
                    }
                } else if self.eat_keyword(kw::Break) {
                    let label = self.eat_label();
                    let e = if self.token.can_begin_expr()
                               && !(self.token == token::OpenDelim(token::Brace)
                                    && self.restrictions.contains(
                                           Restrictions::NO_STRUCT_LITERAL)) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    ex = ExprKind::Break(label, e);
                    hi = self.prev_span;
                } else if self.eat_keyword(kw::Yield) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Yield(Some(e));
                    } else {
                        ex = ExprKind::Yield(None);
                    }

                    let span = lo.to(hi);
                    self.sess.gated_spans.gate(sym::generators, span);
                } else if self.eat_keyword(kw::Let) {
                    return self.parse_let_expr(attrs);
                } else if is_span_rust_2018 && self.eat_keyword(kw::Await) {
                    let (await_hi, e_kind) = self.parse_incorrect_await_syntax(lo, self.prev_span)?;
                    hi = await_hi;
                    ex = e_kind;
                } else {
                    if !self.unclosed_delims.is_empty() && self.check(&token::Semi) {
                        // Don't complain about bare semicolons after unclosed braces
                        // recovery in order to keep the error count down. Fixing the
                        // delimiters will possibly also fix the bare semicolon found in
                        // expression context. For example, silence the following error:
                        //
                        //     error: expected expression, found `;`
                        //      --> file.rs:2:13
                        //       |
                        //     2 |     foo(bar(;
                        //       |             ^ expected expression
                        self.bump();
                        return Ok(self.mk_expr(self.token.span, ExprKind::Err, ThinVec::new()));
                    }
                    parse_lit!()
                }
            }
        }

        let expr = self.mk_expr(lo.to(hi), ex, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    /// Returns a string literal if the next token is a string literal.
    /// In case of error returns `Some(lit)` if the next token is a literal with a wrong kind,
    /// and returns `None` if the next token is not literal at all.
    pub fn parse_str_lit(&mut self) -> Result<ast::StrLit, Option<Lit>> {
        match self.parse_opt_lit() {
            Some(lit) => match lit.kind {
                ast::LitKind::Str(symbol_unescaped, style) => Ok(ast::StrLit {
                    style,
                    symbol: lit.token.symbol,
                    suffix: lit.token.suffix,
                    span: lit.span,
                    symbol_unescaped,
                }),
                _ => Err(Some(lit)),
            }
            None => Err(None),
        }
    }

    pub(super) fn parse_lit(&mut self) -> PResult<'a, Lit> {
        self.parse_opt_lit().ok_or_else(|| {
            let msg = format!("unexpected token: {}", self.this_token_descr());
            self.span_fatal(self.token.span, &msg)
        })
    }

    /// Matches `lit = true | false | token_lit`.
    /// Returns `None` if the next token is not a literal.
    pub(super) fn parse_opt_lit(&mut self) -> Option<Lit> {
        let mut recovered = None;
        if self.token == token::Dot {
            // Attempt to recover `.4` as `0.4`. We don't currently have any syntax where
            // dot would follow an optional literal, so we do this unconditionally.
            recovered = self.look_ahead(1, |next_token| {
                if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix })
                        = next_token.kind {
                    if self.token.span.hi() == next_token.span.lo() {
                        let s = String::from("0.") + &symbol.as_str();
                        let kind = TokenKind::lit(token::Float, Symbol::intern(&s), suffix);
                        return Some(Token::new(kind, self.token.span.to(next_token.span)));
                    }
                }
                None
            });
            if let Some(token) = &recovered {
                self.bump();
                self.struct_span_err(token.span, "float literals must have an integer part")
                    .span_suggestion(
                        token.span,
                        "must have an integer part",
                        pprust::token_to_string(token),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            }
        }

        let token = recovered.as_ref().unwrap_or(&self.token);
        match Lit::from_token(token) {
            Ok(lit) => {
                self.bump();
                Some(lit)
            }
            Err(LitError::NotLiteral) => {
                None
            }
            Err(err) => {
                let span = token.span;
                let lit = match token.kind {
                    token::Literal(lit) => lit,
                    _ => unreachable!(),
                };
                self.bump();
                self.report_lit_error(err, lit, span);
                // Pack possible quotes and prefixes from the original literal into
                // the error literal's symbol so they can be pretty-printed faithfully.
                let suffixless_lit = token::Lit::new(lit.kind, lit.symbol, None);
                let symbol = Symbol::intern(&suffixless_lit.to_string());
                let lit = token::Lit::new(token::Err, symbol, lit.suffix);
                Some(Lit::from_lit_token(lit, span).unwrap_or_else(|_| unreachable!()))
            }
        }
    }

    fn report_lit_error(&self, err: LitError, lit: token::Lit, span: Span) {
        // Checks if `s` looks like i32 or u1234 etc.
        fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
            s.len() > 1
            && s.starts_with(first_chars)
            && s[1..].chars().all(|c| c.is_ascii_digit())
        }

        let token::Lit { kind, suffix, .. } = lit;
        match err {
            // `NotLiteral` is not an error by itself, so we don't report
            // it and give the parser opportunity to try something else.
            LitError::NotLiteral => {}
            // `LexerError` *is* an error, but it was already reported
            // by lexer, so here we don't report it the second time.
            LitError::LexerError => {}
            LitError::InvalidSuffix => {
                self.expect_no_suffix(
                    span,
                    &format!("{} {} literal", kind.article(), kind.descr()),
                    suffix,
                );
            }
            LitError::InvalidIntSuffix => {
                let suf = suffix.expect("suffix error with no suffix").as_str();
                if looks_like_width_suffix(&['i', 'u'], &suf) {
                    // If it looks like a width, try to be helpful.
                    let msg = format!("invalid width `{}` for integer literal", &suf[1..]);
                    self.struct_span_err(span, &msg)
                        .help("valid widths are 8, 16, 32, 64 and 128")
                        .emit();
                } else {
                    let msg = format!("invalid suffix `{}` for integer literal", suf);
                    self.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("the suffix must be one of the integral types (`u32`, `isize`, etc)")
                        .emit();
                }
            }
            LitError::InvalidFloatSuffix => {
                let suf = suffix.expect("suffix error with no suffix").as_str();
                if looks_like_width_suffix(&['f'], &suf) {
                    // If it looks like a width, try to be helpful.
                    let msg = format!("invalid width `{}` for float literal", &suf[1..]);
                    self.struct_span_err(span, &msg)
                        .help("valid widths are 32 and 64")
                        .emit();
                } else {
                    let msg = format!("invalid suffix `{}` for float literal", suf);
                    self.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("valid suffixes are `f32` and `f64`")
                        .emit();
                }
            }
            LitError::NonDecimalFloat(base) => {
                let descr = match base {
                    16 => "hexadecimal",
                    8 => "octal",
                    2 => "binary",
                    _ => unreachable!(),
                };
                self.struct_span_err(span, &format!("{} float literal is not supported", descr))
                    .span_label(span, "not supported")
                    .emit();
            }
            LitError::IntTooLarge => {
                self.struct_span_err(span, "integer literal is too large")
                    .emit();
            }
        }
    }

    pub(super) fn expect_no_suffix(&self, sp: Span, kind: &str, suffix: Option<Symbol>) {
        if let Some(suf) = suffix {
            let mut err = if kind == "a tuple index"
                && [sym::i32, sym::u32, sym::isize, sym::usize].contains(&suf)
            {
                // #59553: warn instead of reject out of hand to allow the fix to percolate
                // through the ecosystem when people fix their macros
                let mut err = self.sess.span_diagnostic.struct_span_warn(
                    sp,
                    &format!("suffixes on {} are invalid", kind),
                );
                err.note(&format!(
                    "`{}` is *temporarily* accepted on tuple index fields as it was \
                        incorrectly accepted on stable for a few releases",
                    suf,
                ));
                err.help(
                    "on proc macros, you'll want to use `syn::Index::from` or \
                        `proc_macro::Literal::*_unsuffixed` for code that will desugar \
                        to tuple field access",
                );
                err.note(
                    "for more context, see https://github.com/rust-lang/rust/issues/60210",
                );
                err
            } else {
                self.struct_span_err(sp, &format!("suffixes on {} are invalid", kind))
            };
            err.span_label(sp, format!("invalid suffix `{}`", suf));
            err.emit();
        }
    }

    /// Matches `'-' lit | lit` (cf. `ast_validation::AstValidator::check_expr_within_pat`).
    pub fn parse_literal_maybe_minus(&mut self) -> PResult<'a, P<Expr>> {
        maybe_whole_expr!(self);

        let minus_lo = self.token.span;
        let minus_present = self.eat(&token::BinOp(token::Minus));
        let lo = self.token.span;
        let literal = self.parse_lit()?;
        let hi = self.prev_span;
        let expr = self.mk_expr(lo.to(hi), ExprKind::Lit(literal), ThinVec::new());

        if minus_present {
            let minus_hi = self.prev_span;
            let unary = self.mk_unary(UnOp::Neg, expr);
            Ok(self.mk_expr(minus_lo.to(minus_hi), unary, ThinVec::new()))
        } else {
            Ok(expr)
        }
    }

    /// Parses a block or unsafe block.
    pub(super) fn parse_block_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        blk_mode: BlockCheckMode,
        outer_attrs: ThinVec<Attribute>,
    ) -> PResult<'a, P<Expr>> {
        if let Some(label) = opt_label {
            self.sess.gated_spans.gate(sym::label_break_value, label.ident.span);
        }

        self.expect(&token::OpenDelim(token::Brace))?;

        let mut attrs = outer_attrs;
        attrs.extend(self.parse_inner_attributes()?);

        let blk = self.parse_block_tail(lo, blk_mode)?;
        Ok(self.mk_expr(blk.span, ExprKind::Block(blk, opt_label), attrs))
    }

    /// Parses a closure expression (e.g., `move |args| expr`).
    fn parse_closure_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        let movability = if self.eat_keyword(kw::Static) {
            Movability::Static
        } else {
            Movability::Movable
        };

        let asyncness = if self.token.span.rust_2018() {
            self.parse_asyncness()
        } else {
            IsAsync::NotAsync
        };
        if asyncness.is_async() {
            // Feature-gate `async ||` closures.
            self.sess.gated_spans.gate(sym::async_closure, self.prev_span);
        }

        let capture_clause = self.parse_capture_clause();
        let decl = self.parse_fn_block_decl()?;
        let decl_hi = self.prev_span;
        let body = match decl.output {
            FunctionRetTy::Default(_) => {
                let restrictions = self.restrictions - Restrictions::STMT_EXPR;
                self.parse_expr_res(restrictions, None)?
            },
            _ => {
                // If an explicit return type is given, require a block to appear (RFC 968).
                let body_lo = self.token.span;
                self.parse_block_expr(None, body_lo, BlockCheckMode::Default, ThinVec::new())?
            }
        };

        Ok(self.mk_expr(
            lo.to(body.span),
            ExprKind::Closure(capture_clause, asyncness, movability, decl, body, lo.to(decl_hi)),
            attrs))
    }

    /// Parses an optional `move` prefix to a closure lke construct.
    fn parse_capture_clause(&mut self) -> CaptureBy {
        if self.eat_keyword(kw::Move) {
            CaptureBy::Value
        } else {
            CaptureBy::Ref
        }
    }

    /// Parses the `|arg, arg|` header of a closure.
    fn parse_fn_block_decl(&mut self) -> PResult<'a, P<FnDecl>> {
        let inputs_captures = {
            if self.eat(&token::OrOr) {
                Vec::new()
            } else {
                self.expect(&token::BinOp(token::Or))?;
                let args = self.parse_seq_to_before_tokens(
                    &[&token::BinOp(token::Or), &token::OrOr],
                    SeqSep::trailing_allowed(token::Comma),
                    TokenExpectType::NoExpect,
                    |p| p.parse_fn_block_param()
                )?.0;
                self.expect_or()?;
                args
            }
        };
        let output = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: inputs_captures,
            output,
        }))
    }

    /// Parses a parameter in a closure header (e.g., `|arg, arg|`).
    fn parse_fn_block_param(&mut self) -> PResult<'a, Param> {
        let lo = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        let pat = self.parse_pat(PARAM_EXPECTED)?;
        let t = if self.eat(&token::Colon) {
            self.parse_ty()?
        } else {
            P(Ty {
                id: DUMMY_NODE_ID,
                kind: TyKind::Infer,
                span: self.prev_span,
            })
        };
        let span = lo.to(self.token.span);
        Ok(Param {
            attrs: attrs.into(),
            ty: t,
            pat,
            span,
            id: DUMMY_NODE_ID,
            is_placeholder: false,
        })
    }

    /// Parses an `if` expression (`if` token already eaten).
    fn parse_if_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let lo = self.prev_span;
        let cond = self.parse_cond_expr()?;

        // Verify that the parsed `if` condition makes sense as a condition. If it is a block, then
        // verify that the last statement is either an implicit return (no `;`) or an explicit
        // return. This won't catch blocks with an explicit `return`, but that would be caught by
        // the dead code lint.
        if self.eat_keyword(kw::Else) || !cond.returns() {
            let sp = self.sess.source_map().next_point(lo);
            let mut err = self.diagnostic()
                .struct_span_err(sp, "missing condition for `if` expression");
            err.span_label(sp, "expected if condition here");
            return Err(err)
        }
        let not_block = self.token != token::OpenDelim(token::Brace);
        let thn = self.parse_block().map_err(|mut err| {
            if not_block {
                err.span_label(lo, "this `if` statement has a condition, but no block");
            }
            err
        })?;
        let mut els: Option<P<Expr>> = None;
        let mut hi = thn.span;
        if self.eat_keyword(kw::Else) {
            let elexpr = self.parse_else_expr()?;
            hi = elexpr.span;
            els = Some(elexpr);
        }
        Ok(self.mk_expr(lo.to(hi), ExprKind::If(cond, thn, els), attrs))
    }

    /// Parses the condition of a `if` or `while` expression.
    fn parse_cond_expr(&mut self) -> PResult<'a, P<Expr>> {
        let cond = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;

        if let ExprKind::Let(..) = cond.kind {
            // Remove the last feature gating of a `let` expression since it's stable.
            self.sess.gated_spans.ungate_last(sym::let_chains, cond.span);
        }

        Ok(cond)
    }

    /// Parses a `let $pat = $expr` pseudo-expression.
    /// The `let` token has already been eaten.
    fn parse_let_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let lo = self.prev_span;
        let pat = self.parse_top_pat(GateOr::No)?;
        self.expect(&token::Eq)?;
        let expr = self.with_res(
            Restrictions::NO_STRUCT_LITERAL,
            |this| this.parse_assoc_expr_with(1 + prec_let_scrutinee_needs_par(), None.into())
        )?;
        let span = lo.to(expr.span);
        self.sess.gated_spans.gate(sym::let_chains, span);
        Ok(self.mk_expr(span, ExprKind::Let(pat, expr), attrs))
    }

    /// Parses an `else { ... }` expression (`else` token already eaten).
    fn parse_else_expr(&mut self) -> PResult<'a, P<Expr>> {
        if self.eat_keyword(kw::If) {
            return self.parse_if_expr(ThinVec::new());
        } else {
            let blk = self.parse_block()?;
            return Ok(self.mk_expr(blk.span, ExprKind::Block(blk, None), ThinVec::new()));
        }
    }

    /// Parses a `for ... in` expression (`for` token already eaten).
    fn parse_for_expr(
        &mut self,
        opt_label: Option<Label>,
        span_lo: Span,
        mut attrs: ThinVec<Attribute>
    ) -> PResult<'a, P<Expr>> {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        // Record whether we are about to parse `for (`.
        // This is used below for recovery in case of `for ( $stuff ) $block`
        // in which case we will suggest `for $stuff $block`.
        let begin_paren = match self.token.kind {
            token::OpenDelim(token::Paren) => Some(self.token.span),
            _ => None,
        };

        let pat = self.parse_top_pat(GateOr::Yes)?;
        if !self.eat_keyword(kw::In) {
            let in_span = self.prev_span.between(self.token.span);
            self.struct_span_err(in_span, "missing `in` in `for` loop")
                .span_suggestion_short(
                    in_span,
                    "try adding `in` here", " in ".into(),
                    // has been misleading, at least in the past (closed Issue #48492)
                    Applicability::MaybeIncorrect
                )
                .emit();
        }
        let in_span = self.prev_span;
        self.check_for_for_in_in_typo(in_span);
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;

        let pat = self.recover_parens_around_for_head(pat, &expr, begin_paren);

        let (iattrs, loop_block) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);

        let hi = self.prev_span;
        Ok(self.mk_expr(span_lo.to(hi), ExprKind::ForLoop(pat, expr, loop_block, opt_label), attrs))
    }

    /// Parses a `while` or `while let` expression (`while` token already eaten).
    fn parse_while_expr(
        &mut self,
        opt_label: Option<Label>,
        span_lo: Span,
        mut attrs: ThinVec<Attribute>
    ) -> PResult<'a, P<Expr>> {
        let cond = self.parse_cond_expr()?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        Ok(self.mk_expr(span, ExprKind::While(cond, body, opt_label), attrs))
    }

    /// Parses `loop { ... }` (`loop` token already eaten).
    fn parse_loop_expr(
        &mut self,
        opt_label: Option<Label>,
        span_lo: Span,
        mut attrs: ThinVec<Attribute>
    ) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        Ok(self.mk_expr(span, ExprKind::Loop(body, opt_label), attrs))
    }

    fn eat_label(&mut self) -> Option<Label> {
        if let Some(ident) = self.token.lifetime() {
            let span = self.token.span;
            self.bump();
            Some(Label { ident: Ident::new(ident.name, span) })
        } else {
            None
        }
    }

    /// Parses a `match ... { ... }` expression (`match` token already eaten).
    fn parse_match_expr(&mut self, mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let match_span = self.prev_span;
        let lo = self.prev_span;
        let discriminant = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        if let Err(mut e) = self.expect(&token::OpenDelim(token::Brace)) {
            if self.token == token::Semi {
                e.span_suggestion_short(
                    match_span,
                    "try removing this `match`",
                    String::new(),
                    Applicability::MaybeIncorrect // speculative
                );
            }
            return Err(e)
        }
        attrs.extend(self.parse_inner_attributes()?);

        let mut arms: Vec<Arm> = Vec::new();
        while self.token != token::CloseDelim(token::Brace) {
            match self.parse_arm() {
                Ok(arm) => arms.push(arm),
                Err(mut e) => {
                    // Recover by skipping to the end of the block.
                    e.emit();
                    self.recover_stmt();
                    let span = lo.to(self.token.span);
                    if self.token == token::CloseDelim(token::Brace) {
                        self.bump();
                    }
                    return Ok(self.mk_expr(span, ExprKind::Match(discriminant, arms), attrs));
                }
            }
        }
        let hi = self.token.span;
        self.bump();
        return Ok(self.mk_expr(lo.to(hi), ExprKind::Match(discriminant, arms), attrs));
    }

    pub(super) fn parse_arm(&mut self) -> PResult<'a, Arm> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let pat = self.parse_top_pat(GateOr::No)?;
        let guard = if self.eat_keyword(kw::If) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let arrow_span = self.token.span;
        self.expect(&token::FatArrow)?;
        let arm_start_span = self.token.span;

        let expr = self.parse_expr_res(Restrictions::STMT_EXPR, None)
            .map_err(|mut err| {
                err.span_label(arrow_span, "while parsing the `match` arm starting here");
                err
            })?;

        let require_comma = classify::expr_requires_semi_to_be_stmt(&expr)
            && self.token != token::CloseDelim(token::Brace);

        let hi = self.token.span;

        if require_comma {
            let cm = self.sess.source_map();
            self.expect_one_of(&[token::Comma], &[token::CloseDelim(token::Brace)])
                .map_err(|mut err| {
                    match (cm.span_to_lines(expr.span), cm.span_to_lines(arm_start_span)) {
                        (Ok(ref expr_lines), Ok(ref arm_start_lines))
                        if arm_start_lines.lines[0].end_col == expr_lines.lines[0].end_col
                            && expr_lines.lines.len() == 2
                            && self.token == token::FatArrow => {
                            // We check whether there's any trailing code in the parse span,
                            // if there isn't, we very likely have the following:
                            //
                            // X |     &Y => "y"
                            //   |        --    - missing comma
                            //   |        |
                            //   |        arrow_span
                            // X |     &X => "x"
                            //   |      - ^^ self.token.span
                            //   |      |
                            //   |      parsed until here as `"y" & X`
                            err.span_suggestion_short(
                                cm.next_point(arm_start_span),
                                "missing a comma here to end this `match` arm",
                                ",".to_owned(),
                                Applicability::MachineApplicable
                            );
                        }
                        _ => {
                            err.span_label(arrow_span,
                                           "while parsing the `match` arm starting here");
                        }
                    }
                    err
                })?;
        } else {
            self.eat(&token::Comma);
        }

        Ok(ast::Arm {
            attrs,
            pat,
            guard,
            body: expr,
            span: lo.to(hi),
            id: DUMMY_NODE_ID,
            is_placeholder: false,
        })
    }

    /// Parses a `try {...}` expression (`try` token already eaten).
    fn parse_try_block(
        &mut self,
        span_lo: Span,
        mut attrs: ThinVec<Attribute>
    ) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        if self.eat_keyword(kw::Catch) {
            let mut error = self.struct_span_err(self.prev_span,
                                                 "keyword `catch` cannot follow a `try` block");
            error.help("try using `match` on the result of the `try` block instead");
            error.emit();
            Err(error)
        } else {
            let span = span_lo.to(body.span);
            self.sess.gated_spans.gate(sym::try_blocks, span);
            Ok(self.mk_expr(span, ExprKind::TryBlock(body), attrs))
        }
    }

    fn is_do_catch_block(&self) -> bool {
        self.token.is_keyword(kw::Do) &&
        self.is_keyword_ahead(1, &[kw::Catch]) &&
        self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace)) &&
        !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_try_block(&self) -> bool {
        self.token.is_keyword(kw::Try) &&
        self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace)) &&
        self.token.span.rust_2018() &&
        // Prevent `while try {} {}`, `if try {} {} else {}`, etc.
        !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    /// Parses an `async move? {...}` expression.
    fn parse_async_block(&mut self, mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let span_lo = self.token.span;
        self.expect_keyword(kw::Async)?;
        let capture_clause = self.parse_capture_clause();
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        Ok(self.mk_expr(
            span_lo.to(body.span),
            ExprKind::Async(capture_clause, DUMMY_NODE_ID, body), attrs))
    }

    fn is_async_block(&self) -> bool {
        self.token.is_keyword(kw::Async) &&
        (
            ( // `async move {`
                self.is_keyword_ahead(1, &[kw::Move]) &&
                self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))
            ) || ( // `async {`
                self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace))
            )
        )
    }

    fn maybe_parse_struct_expr(
        &mut self,
        lo: Span,
        path: &ast::Path,
        attrs: &ThinVec<Attribute>,
    ) -> Option<PResult<'a, P<Expr>>> {
        let struct_allowed = !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
        let certainly_not_a_block = || self.look_ahead(1, |t| t.is_ident()) && (
            // `{ ident, ` cannot start a block.
            self.look_ahead(2, |t| t == &token::Comma) ||
            self.look_ahead(2, |t| t == &token::Colon) && (
                // `{ ident: token, ` cannot start a block.
                self.look_ahead(4, |t| t == &token::Comma) ||
                // `{ ident: ` cannot start a block unless it's a type ascription `ident: Type`.
                self.look_ahead(3, |t| !t.can_begin_type())
            )
        );

        if struct_allowed || certainly_not_a_block() {
            // This is a struct literal, but we don't can't accept them here.
            let expr = self.parse_struct_expr(lo, path.clone(), attrs.clone());
            if let (Ok(expr), false) = (&expr, struct_allowed) {
                self.struct_span_err(
                    expr.span,
                    "struct literals are not allowed here",
                )
                .multipart_suggestion(
                    "surround the struct literal with parentheses",
                    vec![
                        (lo.shrink_to_lo(), "(".to_string()),
                        (expr.span.shrink_to_hi(), ")".to_string()),
                    ],
                    Applicability::MachineApplicable,
                )
                .emit();
            }
            return Some(expr);
        }
        None
    }

    pub(super) fn parse_struct_expr(
        &mut self,
        lo: Span,
        pth: ast::Path,
        mut attrs: ThinVec<Attribute>
    ) -> PResult<'a, P<Expr>> {
        let struct_sp = lo.to(self.prev_span);
        self.bump();
        let mut fields = Vec::new();
        let mut base = None;

        attrs.extend(self.parse_inner_attributes()?);

        while self.token != token::CloseDelim(token::Brace) {
            if self.eat(&token::DotDot) {
                let exp_span = self.prev_span;
                match self.parse_expr() {
                    Ok(e) => {
                        base = Some(e);
                    }
                    Err(mut e) => {
                        e.emit();
                        self.recover_stmt();
                    }
                }
                if self.token == token::Comma {
                    self.struct_span_err(
                        exp_span.to(self.prev_span),
                        "cannot use a comma after the base struct",
                    )
                    .span_suggestion_short(
                        self.token.span,
                        "remove this comma",
                        String::new(),
                        Applicability::MachineApplicable
                    )
                    .note("the base struct must always be the last field")
                    .emit();
                    self.recover_stmt();
                }
                break;
            }

            let mut recovery_field = None;
            if let token::Ident(name, _) = self.token.kind {
                if !self.token.is_reserved_ident() && self.look_ahead(1, |t| *t == token::Colon) {
                    // Use in case of error after field-looking code: `S { foo: () with a }`.
                    recovery_field = Some(ast::Field {
                        ident: Ident::new(name, self.token.span),
                        span: self.token.span,
                        expr: self.mk_expr(self.token.span, ExprKind::Err, ThinVec::new()),
                        is_shorthand: false,
                        attrs: ThinVec::new(),
                        id: DUMMY_NODE_ID,
                        is_placeholder: false,
                    });
                }
            }
            let mut parsed_field = None;
            match self.parse_field() {
                Ok(f) => parsed_field = Some(f),
                Err(mut e) => {
                    e.span_label(struct_sp, "while parsing this struct");
                    e.emit();

                    // If the next token is a comma, then try to parse
                    // what comes next as additional fields, rather than
                    // bailing out until next `}`.
                    if self.token != token::Comma {
                        self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                        if self.token != token::Comma {
                            break;
                        }
                    }
                }
            }

            match self.expect_one_of(&[token::Comma],
                                     &[token::CloseDelim(token::Brace)]) {
                Ok(_) => if let Some(f) = parsed_field.or(recovery_field) {
                    // Only include the field if there's no parse error for the field name.
                    fields.push(f);
                }
                Err(mut e) => {
                    if let Some(f) = recovery_field {
                        fields.push(f);
                    }
                    e.span_label(struct_sp, "while parsing this struct");
                    e.emit();
                    self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                    self.eat(&token::Comma);
                }
            }
        }

        let span = lo.to(self.token.span);
        self.expect(&token::CloseDelim(token::Brace))?;
        return Ok(self.mk_expr(span, ExprKind::Struct(pth, fields, base), attrs));
    }

    /// Parses `ident (COLON expr)?`.
    fn parse_field(&mut self) -> PResult<'a, Field> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let (fieldname, expr, is_shorthand) = if self.look_ahead(1, |t| {
            t == &token::Colon || t == &token::Eq
        }) {
            let fieldname = self.parse_field_name()?;

            // Check for an equals token. This means the source incorrectly attempts to
            // initialize a field with an eq rather than a colon.
            if self.token == token::Eq {
                self.diagnostic()
                    .struct_span_err(self.token.span, "expected `:`, found `=`")
                    .span_suggestion(
                        fieldname.span.shrink_to_hi().to(self.token.span),
                        "replace equals symbol with a colon",
                        ":".to_string(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            }
            self.bump(); // `:`
            (fieldname, self.parse_expr()?, false)
        } else {
            let fieldname = self.parse_ident_common(false)?;

            // Mimic `x: x` for the `x` field shorthand.
            let path = ast::Path::from_ident(fieldname);
            let expr = self.mk_expr(fieldname.span, ExprKind::Path(None, path), ThinVec::new());
            (fieldname, expr, true)
        };
        Ok(ast::Field {
            ident: fieldname,
            span: lo.to(expr.span),
            expr,
            is_shorthand,
            attrs: attrs.into(),
            id: DUMMY_NODE_ID,
            is_placeholder: false,
        })
    }

    fn err_dotdotdot_syntax(&self, span: Span) {
        self.struct_span_err(span, "unexpected token: `...`")
            .span_suggestion(
                span,
                "use `..` for an exclusive range", "..".to_owned(),
                Applicability::MaybeIncorrect
            )
            .span_suggestion(
                span,
                "or `..=` for an inclusive range", "..=".to_owned(),
                Applicability::MaybeIncorrect
            )
            .emit();
    }

    fn err_larrow_operator(&self, span: Span) {
        self.struct_span_err(
            span,
            "unexpected token: `<-`"
        ).span_suggestion(
            span,
            "if you meant to write a comparison against a negative value, add a \
             space in between `<` and `-`",
            "< -".to_string(),
            Applicability::MaybeIncorrect
        ).emit();
    }

    fn mk_assign_op(&self, binop: BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ExprKind {
        ExprKind::AssignOp(binop, lhs, rhs)
    }

    fn mk_range(
        &self,
        start: Option<P<Expr>>,
        end: Option<P<Expr>>,
        limits: RangeLimits
    ) -> PResult<'a, ExprKind> {
        if end.is_none() && limits == RangeLimits::Closed {
            Err(self.span_fatal_err(self.token.span, Error::InclusiveRangeWithNoEnd))
        } else {
            Ok(ExprKind::Range(start, end, limits))
        }
    }

    fn mk_unary(&self, unop: UnOp, expr: P<Expr>) -> ExprKind {
        ExprKind::Unary(unop, expr)
    }

    fn mk_binary(&self, binop: BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ExprKind {
        ExprKind::Binary(binop, lhs, rhs)
    }

    fn mk_index(&self, expr: P<Expr>, idx: P<Expr>) -> ExprKind {
        ExprKind::Index(expr, idx)
    }

    fn mk_call(&self, f: P<Expr>, args: Vec<P<Expr>>) -> ExprKind {
        ExprKind::Call(f, args)
    }

    fn mk_await_expr(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        let span = lo.to(self.prev_span);
        let await_expr = self.mk_expr(span, ExprKind::Await(self_arg), ThinVec::new());
        self.recover_from_await_method_call();
        Ok(await_expr)
    }

    crate fn mk_expr(&self, span: Span, kind: ExprKind, attrs: ThinVec<Attribute>) -> P<Expr> {
        P(Expr { kind, span, attrs, id: DUMMY_NODE_ID })
    }

    pub(super) fn mk_expr_err(&self, span: Span) -> P<Expr> {
        self.mk_expr(span, ExprKind::Err, ThinVec::new())
    }
}
