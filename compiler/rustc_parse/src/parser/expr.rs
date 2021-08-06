use super::pat::{RecoverColon, RecoverComma, PARAM_EXPECTED};
use super::ty::{AllowPlus, RecoverQPath, RecoverReturnSign};
use super::{
    AttrWrapper, BlockMode, ClosureSpans, ForceCollect, Parser, PathStyle, Restrictions, TokenType,
};
use super::{SemiColonMode, SeqSep, TokenExpectType, TrailingToken};
use crate::maybe_recover_from_interpolated_ty_qpath;

use ast::token::DelimToken;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::tokenstream::Spacing;
use rustc_ast::util::classify;
use rustc_ast::util::literal::LitError;
use rustc_ast::util::parser::{prec_let_scrutinee_needs_par, AssocOp, Fixity};
use rustc_ast::{self as ast, AttrStyle, AttrVec, CaptureBy, ExprField, Lit, UnOp, DUMMY_NODE_ID};
use rustc_ast::{AnonConst, BinOp, BinOpKind, FnDecl, FnRetTy, MacCall, Param, Ty, TyKind};
use rustc_ast::{Arm, Async, BlockCheckMode, Expr, ExprKind, Label, Movability, RangeLimits};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, DiagnosticBuilder, PResult};
use rustc_session::lint::builtin::BREAK_WITH_LABEL_AND_LOOP;
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_span::edition::LATEST_STABLE_EDITION;
use rustc_span::source_map::{self, Span, Spanned};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, Pos};
use std::mem;

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
                        $p.prev_token.span,
                        ExprKind::Path(None, path),
                        AttrVec::new(),
                    ));
                }
                token::NtBlock(block) => {
                    let block = block.clone();
                    $p.bump();
                    return Ok($p.mk_expr(
                        $p.prev_token.span,
                        ExprKind::Block(block, None),
                        AttrVec::new(),
                    ));
                }
                _ => {}
            };
        }
    };
}

#[derive(Debug)]
pub(super) enum LhsExpr {
    NotYetParsed,
    AttributesParsed(AttrWrapper),
    AlreadyParsed(P<Expr>),
}

impl From<Option<AttrWrapper>> for LhsExpr {
    /// Converts `Some(attrs)` into `LhsExpr::AttributesParsed(attrs)`
    /// and `None` into `LhsExpr::NotYetParsed`.
    ///
    /// This conversion does not allocate.
    fn from(o: Option<AttrWrapper>) -> Self {
        if let Some(attrs) = o { LhsExpr::AttributesParsed(attrs) } else { LhsExpr::NotYetParsed }
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
        self.current_closure.take();

        self.parse_expr_res(Restrictions::empty(), None)
    }

    /// Parses an expression, forcing tokens to be collected
    pub fn parse_expr_force_collect(&mut self) -> PResult<'a, P<Expr>> {
        self.collect_tokens_no_attrs(|this| this.parse_expr())
    }

    pub fn parse_anon_const_expr(&mut self) -> PResult<'a, AnonConst> {
        self.parse_expr().map(|value| AnonConst { id: DUMMY_NODE_ID, value })
    }

    fn parse_expr_catch_underscore(&mut self) -> PResult<'a, P<Expr>> {
        match self.parse_expr() {
            Ok(expr) => Ok(expr),
            Err(mut err) => match self.token.ident() {
                Some((Ident { name: kw::Underscore, .. }, false))
                    if self.look_ahead(1, |t| t == &token::Comma) =>
                {
                    // Special-case handling of `foo(_, _, _)`
                    err.emit();
                    self.bump();
                    Ok(self.mk_expr(self.prev_token.span, ExprKind::Err, AttrVec::new()))
                }
                _ => Err(err),
            },
        }
    }

    /// Parses a sequence of expressions delimited by parentheses.
    fn parse_paren_expr_seq(&mut self) -> PResult<'a, Vec<P<Expr>>> {
        self.parse_paren_comma_seq(|p| p.parse_expr_catch_underscore()).map(|(r, _)| r)
    }

    /// Parses an expression, subject to the given restrictions.
    #[inline]
    pub(super) fn parse_expr_res(
        &mut self,
        r: Restrictions,
        already_parsed_attrs: Option<AttrWrapper>,
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
        already_parsed_attrs: Option<AttrWrapper>,
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

        if !self.should_continue_as_assoc_expr(&lhs) {
            self.last_type_ascription = None;
            return Ok(lhs);
        }

        self.expected_tokens.push(TokenType::Operator);
        while let Some(op) = self.check_assoc_op() {
            // Adjust the span for interpolated LHS to point to the `$lhs` token
            // and not to what it refers to.
            let lhs_span = match self.prev_token.kind {
                TokenKind::Interpolated(..) => self.prev_token.span,
                _ => lhs.span,
            };

            let cur_op_span = self.token.span;
            let restrictions = if op.node.is_assign_like() {
                self.restrictions & Restrictions::NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            let prec = op.node.precedence();
            if prec < min_prec {
                break;
            }
            // Check for deprecated `...` syntax
            if self.token == token::DotDotDot && op.node == AssocOp::DotDotEq {
                self.err_dotdotdot_syntax(self.token.span);
            }

            if self.token == token::LArrow {
                self.err_larrow_operator(self.token.span);
            }

            self.bump();
            if op.node.is_comparison() {
                if let Some(expr) = self.check_no_chained_comparison(&lhs, &op)? {
                    return Ok(expr);
                }
            }

            if (op.node == AssocOp::Equal || op.node == AssocOp::NotEqual)
                && self.token.kind == token::Eq
                && self.prev_token.span.hi() == self.token.span.lo()
            {
                // Look for JS' `===` and `!==` and recover 😇
                let sp = op.span.to(self.token.span);
                let sugg = match op.node {
                    AssocOp::Equal => "==",
                    AssocOp::NotEqual => "!=",
                    _ => unreachable!(),
                };
                self.struct_span_err(sp, &format!("invalid comparison operator `{}=`", sugg))
                    .span_suggestion_short(
                        sp,
                        &format!("`{s}=` is not a valid comparison operator, use `{s}`", s = sugg),
                        sugg.to_string(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
                self.bump();
            }

            let op = op.node;
            // Special cases:
            if op == AssocOp::As {
                lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Cast)?;
                continue;
            } else if op == AssocOp::Colon {
                lhs = self.parse_assoc_op_ascribe(lhs, lhs_span)?;
                continue;
            } else if op == AssocOp::DotDot || op == AssocOp::DotDotEq {
                // If we didn’t have to handle `x..`/`x..=`, it would be pretty easy to
                // generalise it to the Fixity::None code.
                lhs = self.parse_range_expr(prec, lhs, op, cur_op_span)?;
                break;
            }

            let fixity = op.fixity();
            let prec_adjustment = match fixity {
                Fixity::Right => 0,
                Fixity::Left => 1,
                // We currently have no non-associative operators that are not handled above by
                // the special cases. The code is here only for future convenience.
                Fixity::None => 1,
            };
            let rhs = self.with_res(restrictions - Restrictions::STMT_EXPR, |this| {
                this.parse_assoc_expr_with(prec + prec_adjustment, LhsExpr::NotYetParsed)
            })?;

            let span = self.mk_expr_sp(&lhs, lhs_span, rhs.span);
            lhs = match op {
                AssocOp::Add
                | AssocOp::Subtract
                | AssocOp::Multiply
                | AssocOp::Divide
                | AssocOp::Modulus
                | AssocOp::LAnd
                | AssocOp::LOr
                | AssocOp::BitXor
                | AssocOp::BitAnd
                | AssocOp::BitOr
                | AssocOp::ShiftLeft
                | AssocOp::ShiftRight
                | AssocOp::Equal
                | AssocOp::Less
                | AssocOp::LessEqual
                | AssocOp::NotEqual
                | AssocOp::Greater
                | AssocOp::GreaterEqual => {
                    let ast_op = op.to_ast_binop().unwrap();
                    let binary = self.mk_binary(source_map::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(span, binary, AttrVec::new())
                }
                AssocOp::Assign => {
                    self.mk_expr(span, ExprKind::Assign(lhs, rhs, cur_op_span), AttrVec::new())
                }
                AssocOp::AssignOp(k) => {
                    let aop = match k {
                        token::Plus => BinOpKind::Add,
                        token::Minus => BinOpKind::Sub,
                        token::Star => BinOpKind::Mul,
                        token::Slash => BinOpKind::Div,
                        token::Percent => BinOpKind::Rem,
                        token::Caret => BinOpKind::BitXor,
                        token::And => BinOpKind::BitAnd,
                        token::Or => BinOpKind::BitOr,
                        token::Shl => BinOpKind::Shl,
                        token::Shr => BinOpKind::Shr,
                    };
                    let aopexpr = self.mk_assign_op(source_map::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(span, aopexpr, AttrVec::new())
                }
                AssocOp::As | AssocOp::Colon | AssocOp::DotDot | AssocOp::DotDotEq => {
                    self.span_bug(span, "AssocOp should have been handled by special case")
                }
            };

            if let Fixity::None = fixity {
                break;
            }
        }
        if last_type_ascription_set {
            self.last_type_ascription = None;
        }
        Ok(lhs)
    }

    fn should_continue_as_assoc_expr(&mut self, lhs: &Expr) -> bool {
        match (self.expr_is_complete(lhs), AssocOp::from_token(&self.token)) {
            // Semi-statement forms are odd:
            // See https://github.com/rust-lang/rust/issues/29071
            (true, None) => false,
            (false, _) => true, // Continue parsing the expression.
            // An exhaustive check is done in the following block, but these are checked first
            // because they *are* ambiguous but also reasonable looking incorrect syntax, so we
            // want to keep their span info to improve diagnostics in these cases in a later stage.
            (true, Some(AssocOp::Multiply)) | // `{ 42 } *foo = bar;` or `{ 42 } * 3`
            (true, Some(AssocOp::Subtract)) | // `{ 42 } -5`
            (true, Some(AssocOp::Add)) // `{ 42 } + 42
            // If the next token is a keyword, then the tokens above *are* unambiguously incorrect:
            // `if x { a } else { b } && if y { c } else { d }`
            if !self.look_ahead(1, |t| t.is_used_keyword()) => {
                // These cases are ambiguous and can't be identified in the parser alone.
                let sp = self.sess.source_map().start_point(self.token.span);
                self.sess.ambiguous_block_expr_parse.borrow_mut().insert(sp, lhs.span);
                false
            }
            (true, Some(AssocOp::LAnd)) => {
                // `{ 42 } &&x` (#61475) or `{ 42 } && if x { 1 } else { 0 }`. Separated from the
                // above due to #74233.
                // These cases are ambiguous and can't be identified in the parser alone.
                let sp = self.sess.source_map().start_point(self.token.span);
                self.sess.ambiguous_block_expr_parse.borrow_mut().insert(sp, lhs.span);
                false
            }
            (true, Some(ref op)) if !op.can_continue_expr_unambiguously() => false,
            (true, Some(_)) => {
                self.error_found_expr_would_be_stmt(lhs);
                true
            }
        }
    }

    /// We've found an expression that would be parsed as a statement,
    /// but the next token implies this should be parsed as an expression.
    /// For example: `if let Some(x) = x { x } else { 0 } / 2`.
    fn error_found_expr_would_be_stmt(&self, lhs: &Expr) {
        let mut err = self.struct_span_err(
            self.token.span,
            &format!("expected expression, found `{}`", pprust::token_to_string(&self.token),),
        );
        err.span_label(self.token.span, "expected expression");
        self.sess.expr_parentheses_needed(&mut err, lhs.span);
        err.emit();
    }

    /// Possibly translate the current token to an associative operator.
    /// The method does not advance the current token.
    ///
    /// Also performs recovery for `and` / `or` which are mistaken for `&&` and `||` respectively.
    fn check_assoc_op(&self) -> Option<Spanned<AssocOp>> {
        let (op, span) = match (AssocOp::from_token(&self.token), self.token.ident()) {
            // When parsing const expressions, stop parsing when encountering `>`.
            (
                Some(
                    AssocOp::ShiftRight
                    | AssocOp::Greater
                    | AssocOp::GreaterEqual
                    | AssocOp::AssignOp(token::BinOpToken::Shr),
                ),
                _,
            ) if self.restrictions.contains(Restrictions::CONST_EXPR) => {
                return None;
            }
            (Some(op), _) => (op, self.token.span),
            (None, Some((Ident { name: sym::and, span }, false))) => {
                self.error_bad_logical_op("and", "&&", "conjunction");
                (AssocOp::LAnd, span)
            }
            (None, Some((Ident { name: sym::or, span }, false))) => {
                self.error_bad_logical_op("or", "||", "disjunction");
                (AssocOp::LOr, span)
            }
            _ => return None,
        };
        Some(source_map::respan(span, op))
    }

    /// Error on `and` and `or` suggesting `&&` and `||` respectively.
    fn error_bad_logical_op(&self, bad: &str, good: &str, english: &str) {
        self.struct_span_err(self.token.span, &format!("`{}` is not a logical operator", bad))
            .span_suggestion_short(
                self.token.span,
                &format!("use `{}` to perform logical {}", good, english),
                good.to_string(),
                Applicability::MachineApplicable,
            )
            .note("unlike in e.g., python and PHP, `&&` and `||` are used for logical operators")
            .emit();
    }

    /// Checks if this expression is a successfully parsed statement.
    fn expr_is_complete(&self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::STMT_EXPR)
            && !classify::expr_requires_semi_to_be_stmt(e)
    }

    /// Parses `x..y`, `x..=y`, and `x..`/`x..=`.
    /// The other two variants are handled in `parse_prefix_range_expr` below.
    fn parse_range_expr(
        &mut self,
        prec: usize,
        lhs: P<Expr>,
        op: AssocOp,
        cur_op_span: Span,
    ) -> PResult<'a, P<Expr>> {
        let rhs = if self.is_at_start_of_range_notation_rhs() {
            Some(self.parse_assoc_expr_with(prec + 1, LhsExpr::NotYetParsed)?)
        } else {
            None
        };
        let rhs_span = rhs.as_ref().map_or(cur_op_span, |x| x.span);
        let span = self.mk_expr_sp(&lhs, lhs.span, rhs_span);
        let limits =
            if op == AssocOp::DotDot { RangeLimits::HalfOpen } else { RangeLimits::Closed };
        let range = self.mk_range(Some(lhs), rhs, limits);
        Ok(self.mk_expr(span, range, AttrVec::new()))
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
    fn parse_prefix_range_expr(&mut self, attrs: Option<AttrWrapper>) -> PResult<'a, P<Expr>> {
        // Check for deprecated `...` syntax.
        if self.token == token::DotDotDot {
            self.err_dotdotdot_syntax(self.token.span);
        }

        debug_assert!(
            [token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token.kind),
            "parse_prefix_range_expr: token {:?} is not DotDot/DotDotEq",
            self.token
        );

        let limits = match self.token.kind {
            token::DotDot => RangeLimits::HalfOpen,
            _ => RangeLimits::Closed,
        };
        let op = AssocOp::from_token(&self.token);
        // FIXME: `parse_prefix_range_expr` is called when the current
        // token is `DotDot`, `DotDotDot`, or `DotDotEq`. If we haven't already
        // parsed attributes, then trying to parse them here will always fail.
        // We should figure out how we want attributes on range expressions to work.
        let attrs = self.parse_or_use_outer_attributes(attrs)?;
        self.collect_tokens_for_expr(attrs, |this, attrs| {
            let lo = this.token.span;
            this.bump();
            let (span, opt_end) = if this.is_at_start_of_range_notation_rhs() {
                // RHS must be parsed with more associativity than the dots.
                this.parse_assoc_expr_with(op.unwrap().precedence() + 1, LhsExpr::NotYetParsed)
                    .map(|x| (lo.to(x.span), Some(x)))?
            } else {
                (lo, None)
            };
            let range = this.mk_range(None, opt_end, limits);
            Ok(this.mk_expr(span, range, attrs.into()))
        })
    }

    /// Parses a prefix-unary-operator expr.
    fn parse_prefix_expr(&mut self, attrs: Option<AttrWrapper>) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(attrs)?;
        let lo = self.token.span;

        macro_rules! make_it {
            ($this:ident, $attrs:expr, |this, _| $body:expr) => {
                $this.collect_tokens_for_expr($attrs, |$this, attrs| {
                    let (hi, ex) = $body?;
                    Ok($this.mk_expr(lo.to(hi), ex, attrs.into()))
                })
            };
        }

        let this = self;

        // Note: when adding new unary operators, don't forget to adjust TokenKind::can_begin_expr()
        match this.token.uninterpolate().kind {
            token::Not => make_it!(this, attrs, |this, _| this.parse_unary_expr(lo, UnOp::Not)), // `!expr`
            token::Tilde => make_it!(this, attrs, |this, _| this.recover_tilde_expr(lo)), // `~expr`
            token::BinOp(token::Minus) => {
                make_it!(this, attrs, |this, _| this.parse_unary_expr(lo, UnOp::Neg))
            } // `-expr`
            token::BinOp(token::Star) => {
                make_it!(this, attrs, |this, _| this.parse_unary_expr(lo, UnOp::Deref))
            } // `*expr`
            token::BinOp(token::And) | token::AndAnd => {
                make_it!(this, attrs, |this, _| this.parse_borrow_expr(lo))
            }
            token::BinOp(token::Plus) if this.look_ahead(1, |tok| tok.is_numeric_lit()) => {
                let mut err = this.struct_span_err(lo, "leading `+` is not supported");
                err.span_label(lo, "unexpected `+`");

                // a block on the LHS might have been intended to be an expression instead
                if let Some(sp) = this.sess.ambiguous_block_expr_parse.borrow().get(&lo) {
                    this.sess.expr_parentheses_needed(&mut err, *sp);
                } else {
                    err.span_suggestion_verbose(
                        lo,
                        "try removing the `+`",
                        "".to_string(),
                        Applicability::MachineApplicable,
                    );
                }
                err.emit();

                this.bump();
                this.parse_prefix_expr(None)
            } // `+expr`
            token::Ident(..) if this.token.is_keyword(kw::Box) => {
                make_it!(this, attrs, |this, _| this.parse_box_expr(lo))
            }
            token::Ident(..) if this.is_mistaken_not_ident_negation() => {
                make_it!(this, attrs, |this, _| this.recover_not_expr(lo))
            }
            _ => return this.parse_dot_or_call_expr(Some(attrs)),
        }
    }

    fn parse_prefix_expr_common(&mut self, lo: Span) -> PResult<'a, (Span, P<Expr>)> {
        self.bump();
        let expr = self.parse_prefix_expr(None);
        let (span, expr) = self.interpolated_or_expr_span(expr)?;
        Ok((lo.to(span), expr))
    }

    fn parse_unary_expr(&mut self, lo: Span, op: UnOp) -> PResult<'a, (Span, ExprKind)> {
        let (span, expr) = self.parse_prefix_expr_common(lo)?;
        Ok((span, self.mk_unary(op, expr)))
    }

    // Recover on `!` suggesting for bitwise negation instead.
    fn recover_tilde_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        self.struct_span_err(lo, "`~` cannot be used as a unary operator")
            .span_suggestion_short(
                lo,
                "use `!` to perform bitwise not",
                "!".to_owned(),
                Applicability::MachineApplicable,
            )
            .emit();

        self.parse_unary_expr(lo, UnOp::Not)
    }

    /// Parse `box expr`.
    fn parse_box_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        let (span, expr) = self.parse_prefix_expr_common(lo)?;
        self.sess.gated_spans.gate(sym::box_syntax, span);
        Ok((span, ExprKind::Box(expr)))
    }

    fn is_mistaken_not_ident_negation(&self) -> bool {
        let token_cannot_continue_expr = |t: &Token| match t.uninterpolate().kind {
            // These tokens can start an expression after `!`, but
            // can't continue an expression after an ident
            token::Ident(name, is_raw) => token::ident_can_begin_expr(name, t.span, is_raw),
            token::Literal(..) | token::Pound => true,
            _ => t.is_whole_expr(),
        };
        self.token.is_ident_named(sym::not) && self.look_ahead(1, token_cannot_continue_expr)
    }

    /// Recover on `not expr` in favor of `!expr`.
    fn recover_not_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        // Emit the error...
        let not_token = self.look_ahead(1, |t| t.clone());
        self.struct_span_err(
            not_token.span,
            &format!("unexpected {} after identifier", super::token_descr(&not_token)),
        )
        .span_suggestion_short(
            // Span the `not` plus trailing whitespace to avoid
            // trailing whitespace after the `!` in our suggestion
            self.sess.source_map().span_until_non_whitespace(lo.to(not_token.span)),
            "use `!` to perform logical negation",
            "!".to_owned(),
            Applicability::MachineApplicable,
        )
        .emit();

        // ...and recover!
        self.parse_unary_expr(lo, UnOp::Not)
    }

    /// Returns the span of expr, if it was not interpolated or the span of the interpolated token.
    fn interpolated_or_expr_span(
        &self,
        expr: PResult<'a, P<Expr>>,
    ) -> PResult<'a, (Span, P<Expr>)> {
        expr.map(|e| {
            (
                match self.prev_token.kind {
                    TokenKind::Interpolated(..) => self.prev_token.span,
                    _ => e.span,
                },
                e,
            )
        })
    }

    fn parse_assoc_op_cast(
        &mut self,
        lhs: P<Expr>,
        lhs_span: Span,
        expr_kind: fn(P<Expr>, P<Ty>) -> ExprKind,
    ) -> PResult<'a, P<Expr>> {
        let mk_expr = |this: &mut Self, lhs: P<Expr>, rhs: P<Ty>| {
            this.mk_expr(
                this.mk_expr_sp(&lhs, lhs_span, rhs.span),
                expr_kind(lhs, rhs),
                AttrVec::new(),
            )
        };

        // Save the state of the parser before parsing type normally, in case there is a
        // LessThan comparison after this cast.
        let parser_snapshot_before_type = self.clone();
        let cast_expr = match self.parse_ty_no_plus() {
            Ok(rhs) => mk_expr(self, lhs, rhs),
            Err(mut type_err) => {
                // Rewind to before attempting to parse the type with generics, to recover
                // from situations like `x as usize < y` in which we first tried to parse
                // `usize < y` as a type with generic arguments.
                let parser_snapshot_after_type = mem::replace(self, parser_snapshot_before_type);

                // Check for typo of `'a: loop { break 'a }` with a missing `'`.
                match (&lhs.kind, &self.token.kind) {
                    (
                        // `foo: `
                        ExprKind::Path(None, ast::Path { segments, .. }),
                        TokenKind::Ident(kw::For | kw::Loop | kw::While, false),
                    ) if segments.len() == 1 => {
                        let snapshot = self.clone();
                        let label = Label {
                            ident: Ident::from_str_and_span(
                                &format!("'{}", segments[0].ident),
                                segments[0].ident.span,
                            ),
                        };
                        match self.parse_labeled_expr(label, AttrVec::new(), false) {
                            Ok(expr) => {
                                type_err.cancel();
                                self.struct_span_err(label.ident.span, "malformed loop label")
                                    .span_suggestion(
                                        label.ident.span,
                                        "use the correct loop label format",
                                        label.ident.to_string(),
                                        Applicability::MachineApplicable,
                                    )
                                    .emit();
                                return Ok(expr);
                            }
                            Err(mut err) => {
                                err.cancel();
                                *self = snapshot;
                            }
                        }
                    }
                    _ => {}
                }

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
                                *self = parser_snapshot_after_type;
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
                        let expr =
                            mk_expr(self, lhs, self.mk_ty(path.span, TyKind::Path(None, path)));

                        self.struct_span_err(self.token.span, &msg)
                            .span_label(
                                self.look_ahead(1, |t| t.span).to(span_after_type),
                                "interpreted as generic arguments",
                            )
                            .span_label(self.token.span, format!("not interpreted as {}", op_noun))
                            .multipart_suggestion(
                                &format!("try {} the cast value", op_verb),
                                vec![
                                    (expr.span.shrink_to_lo(), "(".to_string()),
                                    (expr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MachineApplicable,
                            )
                            .emit();

                        expr
                    }
                    Err(mut path_err) => {
                        // Couldn't parse as a path, return original error and parser state.
                        path_err.cancel();
                        *self = parser_snapshot_after_type;
                        return Err(type_err);
                    }
                }
            }
        };

        self.parse_and_disallow_postfix_after_cast(cast_expr)
    }

    /// Parses a postfix operators such as `.`, `?`, or index (`[]`) after a cast,
    /// then emits an error and returns the newly parsed tree.
    /// The resulting parse tree for `&x as T[0]` has a precedence of `((&x) as T)[0]`.
    fn parse_and_disallow_postfix_after_cast(
        &mut self,
        cast_expr: P<Expr>,
    ) -> PResult<'a, P<Expr>> {
        // Save the memory location of expr before parsing any following postfix operators.
        // This will be compared with the memory location of the output expression.
        // If they different we can assume we parsed another expression because the existing expression is not reallocated.
        let addr_before = &*cast_expr as *const _ as usize;
        let span = cast_expr.span;
        let with_postfix = self.parse_dot_or_call_expr_with_(cast_expr, span)?;
        let changed = addr_before != &*with_postfix as *const _ as usize;

        // Check if an illegal postfix operator has been added after the cast.
        // If the resulting expression is not a cast, or has a different memory location, it is an illegal postfix operator.
        if !matches!(with_postfix.kind, ExprKind::Cast(_, _) | ExprKind::Type(_, _)) || changed {
            let msg = format!(
                "casts cannot be followed by {}",
                match with_postfix.kind {
                    ExprKind::Index(_, _) => "indexing",
                    ExprKind::Try(_) => "?",
                    ExprKind::Field(_, _) => "a field access",
                    ExprKind::MethodCall(_, _, _) => "a method call",
                    ExprKind::Call(_, _) => "a function call",
                    ExprKind::Await(_) => "`.await`",
                    ExprKind::Err => return Ok(with_postfix),
                    _ => unreachable!("parse_dot_or_call_expr_with_ shouldn't produce this"),
                }
            );
            let mut err = self.struct_span_err(span, &msg);
            // If type ascription is "likely an error", the user will already be getting a useful
            // help message, and doesn't need a second.
            if self.last_type_ascription.map_or(false, |last_ascription| last_ascription.1) {
                self.maybe_annotate_with_ascription(&mut err, false);
            } else {
                let suggestions = vec![
                    (span.shrink_to_lo(), "(".to_string()),
                    (span.shrink_to_hi(), ")".to_string()),
                ];
                err.multipart_suggestion(
                    "try surrounding the expression in parentheses",
                    suggestions,
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        };
        Ok(with_postfix)
    }

    fn parse_assoc_op_ascribe(&mut self, lhs: P<Expr>, lhs_span: Span) -> PResult<'a, P<Expr>> {
        let maybe_path = self.could_ascription_be_path(&lhs.kind);
        self.last_type_ascription = Some((self.prev_token.span, maybe_path));
        let lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Type)?;
        self.sess.gated_spans.gate(sym::type_ascription, lhs.span);
        Ok(lhs)
    }

    /// Parse `& mut? <expr>` or `& raw [ const | mut ] <expr>`.
    fn parse_borrow_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        self.expect_and()?;
        let has_lifetime = self.token.is_lifetime() && self.look_ahead(1, |t| t != &token::Colon);
        let lifetime = has_lifetime.then(|| self.expect_lifetime()); // For recovery, see below.
        let (borrow_kind, mutbl) = self.parse_borrow_modifiers(lo);
        let expr = self.parse_prefix_expr(None);
        let (hi, expr) = self.interpolated_or_expr_span(expr)?;
        let span = lo.to(hi);
        if let Some(lt) = lifetime {
            self.error_remove_borrow_lifetime(span, lt.ident.span);
        }
        Ok((span, ExprKind::AddrOf(borrow_kind, mutbl, expr)))
    }

    fn error_remove_borrow_lifetime(&self, span: Span, lt_span: Span) {
        self.struct_span_err(span, "borrow expressions cannot be annotated with lifetimes")
            .span_label(lt_span, "annotated with lifetime here")
            .span_suggestion(
                lt_span,
                "remove the lifetime annotation",
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    /// Parse `mut?` or `raw [ const | mut ]`.
    fn parse_borrow_modifiers(&mut self, lo: Span) -> (ast::BorrowKind, ast::Mutability) {
        if self.check_keyword(kw::Raw) && self.look_ahead(1, Token::is_mutability) {
            // `raw [ const | mut ]`.
            let found_raw = self.eat_keyword(kw::Raw);
            assert!(found_raw);
            let mutability = self.parse_const_or_mut().unwrap();
            self.sess.gated_spans.gate(sym::raw_ref_op, lo.to(self.prev_token.span));
            (ast::BorrowKind::Raw, mutability)
        } else {
            // `mut?`
            (ast::BorrowKind::Ref, self.parse_mutability())
        }
    }

    /// Parses `a.b` or `a(13)` or `a[4]` or just `a`.
    fn parse_dot_or_call_expr(&mut self, attrs: Option<AttrWrapper>) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(attrs)?;
        self.collect_tokens_for_expr(attrs, |this, attrs| {
            let base = this.parse_bottom_expr();
            let (span, base) = this.interpolated_or_expr_span(base)?;
            this.parse_dot_or_call_expr_with(base, span, attrs)
        })
    }

    pub(super) fn parse_dot_or_call_expr_with(
        &mut self,
        e0: P<Expr>,
        lo: Span,
        mut attrs: Vec<ast::Attribute>,
    ) -> PResult<'a, P<Expr>> {
        // Stitch the list of outer attributes onto the return value.
        // A little bit ugly, but the best way given the current code
        // structure
        self.parse_dot_or_call_expr_with_(e0, lo).map(|expr| {
            expr.map(|mut expr| {
                attrs.extend::<Vec<_>>(expr.attrs.into());
                expr.attrs = attrs.into();
                expr
            })
        })
    }

    fn parse_dot_or_call_expr_with_(&mut self, mut e: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        loop {
            if self.eat(&token::Question) {
                // `expr?`
                e = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Try(e), AttrVec::new());
                continue;
            }
            if self.eat(&token::Dot) {
                // expr.f
                e = self.parse_dot_suffix_expr(lo, e)?;
                continue;
            }
            if self.expr_is_complete(&e) {
                return Ok(e);
            }
            e = match self.token.kind {
                token::OpenDelim(token::Paren) => self.parse_fn_call_expr(lo, e),
                token::OpenDelim(token::Bracket) => self.parse_index_expr(lo, e)?,
                _ => return Ok(e),
            }
        }
    }

    fn look_ahead_type_ascription_as_field(&mut self) -> bool {
        self.look_ahead(1, |t| t.is_ident())
            && self.look_ahead(2, |t| t == &token::Colon)
            && self.look_ahead(3, |t| t.can_begin_expr())
    }

    fn parse_dot_suffix_expr(&mut self, lo: Span, base: P<Expr>) -> PResult<'a, P<Expr>> {
        match self.token.uninterpolate().kind {
            token::Ident(..) => self.parse_dot_suffix(base, lo),
            token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) => {
                Ok(self.parse_tuple_field_access_expr(lo, base, symbol, suffix, None))
            }
            token::Literal(token::Lit { kind: token::Float, symbol, suffix }) => {
                Ok(self.parse_tuple_field_access_expr_float(lo, base, symbol, suffix))
            }
            _ => {
                self.error_unexpected_after_dot();
                Ok(base)
            }
        }
    }

    fn error_unexpected_after_dot(&self) {
        // FIXME Could factor this out into non_fatal_unexpected or something.
        let actual = pprust::token_to_string(&self.token);
        self.struct_span_err(self.token.span, &format!("unexpected token: `{}`", actual)).emit();
    }

    // We need an identifier or integer, but the next token is a float.
    // Break the float into components to extract the identifier or integer.
    // FIXME: With current `TokenCursor` it's hard to break tokens into more than 2
    // parts unless those parts are processed immediately. `TokenCursor` should either
    // support pushing "future tokens" (would be also helpful to `break_and_eat`), or
    // we should break everything including floats into more basic proc-macro style
    // tokens in the lexer (probably preferable).
    fn parse_tuple_field_access_expr_float(
        &mut self,
        lo: Span,
        base: P<Expr>,
        float: Symbol,
        suffix: Option<Symbol>,
    ) -> P<Expr> {
        #[derive(Debug)]
        enum FloatComponent {
            IdentLike(String),
            Punct(char),
        }
        use FloatComponent::*;

        let float_str = float.as_str();
        let mut components = Vec::new();
        let mut ident_like = String::new();
        for c in float_str.chars() {
            if c == '_' || c.is_ascii_alphanumeric() {
                ident_like.push(c);
            } else if matches!(c, '.' | '+' | '-') {
                if !ident_like.is_empty() {
                    components.push(IdentLike(mem::take(&mut ident_like)));
                }
                components.push(Punct(c));
            } else {
                panic!("unexpected character in a float token: {:?}", c)
            }
        }
        if !ident_like.is_empty() {
            components.push(IdentLike(ident_like));
        }

        // With proc macros the span can refer to anything, the source may be too short,
        // or too long, or non-ASCII. It only makes sense to break our span into components
        // if its underlying text is identical to our float literal.
        let span = self.token.span;
        let can_take_span_apart =
            || self.span_to_snippet(span).as_deref() == Ok(float_str).as_deref();

        match &*components {
            // 1e2
            [IdentLike(i)] => {
                self.parse_tuple_field_access_expr(lo, base, Symbol::intern(&i), suffix, None)
            }
            // 1.
            [IdentLike(i), Punct('.')] => {
                let (ident_span, dot_span) = if can_take_span_apart() {
                    let (span, ident_len) = (span.data(), BytePos::from_usize(i.len()));
                    let ident_span = span.with_hi(span.lo + ident_len);
                    let dot_span = span.with_lo(span.lo + ident_len);
                    (ident_span, dot_span)
                } else {
                    (span, span)
                };
                assert!(suffix.is_none());
                let symbol = Symbol::intern(&i);
                self.token = Token::new(token::Ident(symbol, false), ident_span);
                let next_token = (Token::new(token::Dot, dot_span), self.token_spacing);
                self.parse_tuple_field_access_expr(lo, base, symbol, None, Some(next_token))
            }
            // 1.2 | 1.2e3
            [IdentLike(i1), Punct('.'), IdentLike(i2)] => {
                let (ident1_span, dot_span, ident2_span) = if can_take_span_apart() {
                    let (span, ident1_len) = (span.data(), BytePos::from_usize(i1.len()));
                    let ident1_span = span.with_hi(span.lo + ident1_len);
                    let dot_span = span
                        .with_lo(span.lo + ident1_len)
                        .with_hi(span.lo + ident1_len + BytePos(1));
                    let ident2_span = self.token.span.with_lo(span.lo + ident1_len + BytePos(1));
                    (ident1_span, dot_span, ident2_span)
                } else {
                    (span, span, span)
                };
                let symbol1 = Symbol::intern(&i1);
                self.token = Token::new(token::Ident(symbol1, false), ident1_span);
                // This needs to be `Spacing::Alone` to prevent regressions.
                // See issue #76399 and PR #76285 for more details
                let next_token1 = (Token::new(token::Dot, dot_span), Spacing::Alone);
                let base1 =
                    self.parse_tuple_field_access_expr(lo, base, symbol1, None, Some(next_token1));
                let symbol2 = Symbol::intern(&i2);
                let next_token2 = Token::new(token::Ident(symbol2, false), ident2_span);
                self.bump_with((next_token2, self.token_spacing)); // `.`
                self.parse_tuple_field_access_expr(lo, base1, symbol2, suffix, None)
            }
            // 1e+ | 1e- (recovered)
            [IdentLike(_), Punct('+' | '-')] |
            // 1e+2 | 1e-2
            [IdentLike(_), Punct('+' | '-'), IdentLike(_)] |
            // 1.2e+3 | 1.2e-3
            [IdentLike(_), Punct('.'), IdentLike(_), Punct('+' | '-'), IdentLike(_)] => {
                // See the FIXME about `TokenCursor` above.
                self.error_unexpected_after_dot();
                base
            }
            _ => panic!("unexpected components in a float token: {:?}", components),
        }
    }

    fn parse_tuple_field_access_expr(
        &mut self,
        lo: Span,
        base: P<Expr>,
        field: Symbol,
        suffix: Option<Symbol>,
        next_token: Option<(Token, Spacing)>,
    ) -> P<Expr> {
        match next_token {
            Some(next_token) => self.bump_with(next_token),
            None => self.bump(),
        }
        let span = self.prev_token.span;
        let field = ExprKind::Field(base, Ident::new(field, span));
        self.expect_no_suffix(span, "a tuple index", suffix);
        self.mk_expr(lo.to(span), field, AttrVec::new())
    }

    /// Parse a function call expression, `expr(...)`.
    fn parse_fn_call_expr(&mut self, lo: Span, fun: P<Expr>) -> P<Expr> {
        let snapshot = if self.token.kind == token::OpenDelim(token::Paren)
            && self.look_ahead_type_ascription_as_field()
        {
            Some((self.clone(), fun.kind.clone()))
        } else {
            None
        };
        let open_paren = self.token.span;

        let mut seq = self.parse_paren_expr_seq().map(|args| {
            self.mk_expr(lo.to(self.prev_token.span), self.mk_call(fun, args), AttrVec::new())
        });
        if let Some(expr) =
            self.maybe_recover_struct_lit_bad_delims(lo, open_paren, &mut seq, snapshot)
        {
            return expr;
        }
        self.recover_seq_parse_error(token::Paren, lo, seq)
    }

    /// If we encounter a parser state that looks like the user has written a `struct` literal with
    /// parentheses instead of braces, recover the parser state and provide suggestions.
    fn maybe_recover_struct_lit_bad_delims(
        &mut self,
        lo: Span,
        open_paren: Span,
        seq: &mut PResult<'a, P<Expr>>,
        snapshot: Option<(Self, ExprKind)>,
    ) -> Option<P<Expr>> {
        match (seq.as_mut(), snapshot) {
            (Err(ref mut err), Some((mut snapshot, ExprKind::Path(None, path)))) => {
                let name = pprust::path_to_string(&path);
                snapshot.bump(); // `(`
                match snapshot.parse_struct_fields(path.clone(), false, token::Paren) {
                    Ok((fields, ..)) if snapshot.eat(&token::CloseDelim(token::Paren)) => {
                        // We have are certain we have `Enum::Foo(a: 3, b: 4)`, suggest
                        // `Enum::Foo { a: 3, b: 4 }` or `Enum::Foo(3, 4)`.
                        *self = snapshot;
                        let close_paren = self.prev_token.span;
                        let span = lo.to(self.prev_token.span);
                        err.cancel();
                        self.struct_span_err(
                            span,
                            "invalid `struct` delimiters or `fn` call arguments",
                        )
                        .multipart_suggestion(
                            &format!("if `{}` is a struct, use braces as delimiters", name),
                            vec![(open_paren, " { ".to_string()), (close_paren, " }".to_string())],
                            Applicability::MaybeIncorrect,
                        )
                        .multipart_suggestion(
                            &format!("if `{}` is a function, use the arguments directly", name),
                            fields
                                .into_iter()
                                .map(|field| (field.span.until(field.expr.span), String::new()))
                                .collect(),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                        return Some(self.mk_expr_err(span));
                    }
                    Ok(_) => {}
                    Err(mut err) => err.emit(),
                }
            }
            _ => {}
        }
        None
    }

    /// Parse an indexing expression `expr[...]`.
    fn parse_index_expr(&mut self, lo: Span, base: P<Expr>) -> PResult<'a, P<Expr>> {
        self.bump(); // `[`
        let index = self.parse_expr()?;
        self.expect(&token::CloseDelim(token::Bracket))?;
        Ok(self.mk_expr(lo.to(self.prev_token.span), self.mk_index(base, index), AttrVec::new()))
    }

    /// Assuming we have just parsed `.`, continue parsing into an expression.
    fn parse_dot_suffix(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        if self.token.uninterpolated_span().rust_2018() && self.eat_keyword(kw::Await) {
            return Ok(self.mk_await_expr(self_arg, lo));
        }

        let fn_span_lo = self.token.span;
        let mut segment = self.parse_path_segment(PathStyle::Expr)?;
        self.check_trailing_angle_brackets(&segment, &[&token::OpenDelim(token::Paren)]);
        self.check_turbofish_missing_angle_brackets(&mut segment);

        if self.check(&token::OpenDelim(token::Paren)) {
            // Method call `expr.f()`
            let mut args = self.parse_paren_expr_seq()?;
            args.insert(0, self_arg);

            let fn_span = fn_span_lo.to(self.prev_token.span);
            let span = lo.to(self.prev_token.span);
            Ok(self.mk_expr(span, ExprKind::MethodCall(segment, args, fn_span), AttrVec::new()))
        } else {
            // Field access `expr.f`
            if let Some(args) = segment.args {
                self.struct_span_err(
                    args.span(),
                    "field expressions cannot have generic arguments",
                )
                .emit();
            }

            let span = lo.to(self.prev_token.span);
            Ok(self.mk_expr(span, ExprKind::Field(self_arg, segment.ident), AttrVec::new()))
        }
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
        // attributes by giving them an empty "already-parsed" list.
        let attrs = AttrVec::new();

        // Note: when adding new syntax here, don't forget to adjust `TokenKind::can_begin_expr()`.
        let lo = self.token.span;
        if let token::Literal(_) = self.token.kind {
            // This match arm is a special-case of the `_` match arm below and
            // could be removed without changing functionality, but it's faster
            // to have it here, especially for programs with large constants.
            self.parse_lit_expr(attrs)
        } else if self.check(&token::OpenDelim(token::Paren)) {
            self.parse_tuple_parens_expr(attrs)
        } else if self.check(&token::OpenDelim(token::Brace)) {
            self.parse_block_expr(None, lo, BlockCheckMode::Default, attrs)
        } else if self.check(&token::BinOp(token::Or)) || self.check(&token::OrOr) {
            self.parse_closure_expr(attrs)
        } else if self.check(&token::OpenDelim(token::Bracket)) {
            self.parse_array_or_repeat_expr(attrs, token::Bracket)
        } else if self.check_path() {
            self.parse_path_start_expr(attrs)
        } else if self.check_keyword(kw::Move) || self.check_keyword(kw::Static) {
            self.parse_closure_expr(attrs)
        } else if self.eat_keyword(kw::If) {
            self.parse_if_expr(attrs)
        } else if self.check_keyword(kw::For) {
            if self.choose_generics_over_qpath(1) {
                // NOTE(Centril, eddyb): DO NOT REMOVE! Beyond providing parser recovery,
                // this is an insurance policy in case we allow qpaths in (tuple-)struct patterns.
                // When `for <Foo as Bar>::Proj in $expr $block` is wanted,
                // you can disambiguate in favor of a pattern with `(...)`.
                self.recover_quantified_closure_expr(attrs)
            } else {
                assert!(self.eat_keyword(kw::For));
                self.parse_for_expr(None, self.prev_token.span, attrs)
            }
        } else if self.eat_keyword(kw::While) {
            self.parse_while_expr(None, self.prev_token.span, attrs)
        } else if let Some(label) = self.eat_label() {
            self.parse_labeled_expr(label, attrs, true)
        } else if self.eat_keyword(kw::Loop) {
            self.parse_loop_expr(None, self.prev_token.span, attrs)
        } else if self.eat_keyword(kw::Continue) {
            let kind = ExprKind::Continue(self.eat_label());
            Ok(self.mk_expr(lo.to(self.prev_token.span), kind, attrs))
        } else if self.eat_keyword(kw::Match) {
            let match_sp = self.prev_token.span;
            self.parse_match_expr(attrs).map_err(|mut err| {
                err.span_label(match_sp, "while parsing this match expression");
                err
            })
        } else if self.eat_keyword(kw::Unsafe) {
            self.parse_block_expr(None, lo, BlockCheckMode::Unsafe(ast::UserProvided), attrs)
        } else if self.check_inline_const(0) {
            self.parse_const_block(lo.to(self.token.span))
        } else if self.is_do_catch_block() {
            self.recover_do_catch(attrs)
        } else if self.is_try_block() {
            self.expect_keyword(kw::Try)?;
            self.parse_try_block(lo, attrs)
        } else if self.eat_keyword(kw::Return) {
            self.parse_return_expr(attrs)
        } else if self.eat_keyword(kw::Break) {
            self.parse_break_expr(attrs)
        } else if self.eat_keyword(kw::Yield) {
            self.parse_yield_expr(attrs)
        } else if self.eat_keyword(kw::Let) {
            self.parse_let_expr(attrs)
        } else if self.eat_keyword(kw::Underscore) {
            self.sess.gated_spans.gate(sym::destructuring_assignment, self.prev_token.span);
            Ok(self.mk_expr(self.prev_token.span, ExprKind::Underscore, attrs))
        } else if !self.unclosed_delims.is_empty() && self.check(&token::Semi) {
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
            Ok(self.mk_expr_err(self.token.span))
        } else if self.token.uninterpolated_span().rust_2018() {
            // `Span::rust_2018()` is somewhat expensive; don't get it repeatedly.
            if self.check_keyword(kw::Async) {
                if self.is_async_block() {
                    // Check for `async {` and `async move {`.
                    self.parse_async_block(attrs)
                } else {
                    self.parse_closure_expr(attrs)
                }
            } else if self.eat_keyword(kw::Await) {
                self.recover_incorrect_await_syntax(lo, self.prev_token.span, attrs)
            } else {
                self.parse_lit_expr(attrs)
            }
        } else {
            self.parse_lit_expr(attrs)
        }
    }

    fn parse_lit_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        match self.parse_opt_lit() {
            Some(literal) => {
                let expr = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Lit(literal), attrs);
                self.maybe_recover_from_bad_qpath(expr, true)
            }
            None => self.try_macro_suggestion(),
        }
    }

    fn parse_tuple_parens_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        self.expect(&token::OpenDelim(token::Paren))?;
        let (es, trailing_comma) = match self.parse_seq_to_end(
            &token::CloseDelim(token::Paren),
            SeqSep::trailing_allowed(token::Comma),
            |p| p.parse_expr_catch_underscore(),
        ) {
            Ok(x) => x,
            Err(err) => return Ok(self.recover_seq_parse_error(token::Paren, lo, Err(err))),
        };
        let kind = if es.len() == 1 && !trailing_comma {
            // `(e)` is parenthesized `e`.
            ExprKind::Paren(es.into_iter().next().unwrap())
        } else {
            // `(e,)` is a tuple with only one field, `e`.
            ExprKind::Tup(es)
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    fn parse_array_or_repeat_expr(
        &mut self,
        attrs: AttrVec,
        close_delim: token::DelimToken,
    ) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        self.bump(); // `[` or other open delim

        let close = &token::CloseDelim(close_delim);
        let kind = if self.eat(close) {
            // Empty vector
            ExprKind::Array(Vec::new())
        } else {
            // Non-empty vector
            let first_expr = self.parse_expr()?;
            if self.eat(&token::Semi) {
                // Repeating array syntax: `[ 0; 512 ]`
                let count = self.parse_anon_const_expr()?;
                self.expect(close)?;
                ExprKind::Repeat(first_expr, count)
            } else if self.eat(&token::Comma) {
                // Vector with two or more elements.
                let sep = SeqSep::trailing_allowed(token::Comma);
                let (remaining_exprs, _) = self.parse_seq_to_end(close, sep, |p| p.parse_expr())?;
                let mut exprs = vec![first_expr];
                exprs.extend(remaining_exprs);
                ExprKind::Array(exprs)
            } else {
                // Vector with one element
                self.expect(close)?;
                ExprKind::Array(vec![first_expr])
            }
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    fn parse_path_start_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let (qself, path) = if self.eat_lt() {
            let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
            (Some(qself), path)
        } else {
            (None, self.parse_path(PathStyle::Expr)?)
        };
        let lo = path.span;

        // `!`, as an operator, is prefix, so we know this isn't that.
        let (hi, kind) = if self.eat(&token::Not) {
            // MACRO INVOCATION expression
            if qself.is_some() {
                self.struct_span_err(path.span, "macros cannot use qualified paths").emit();
            }
            let mac = MacCall {
                path,
                args: self.parse_mac_args()?,
                prior_type_ascription: self.last_type_ascription,
            };
            (self.prev_token.span, ExprKind::MacCall(mac))
        } else if self.check(&token::OpenDelim(token::Brace)) {
            if let Some(expr) = self.maybe_parse_struct_expr(qself.as_ref(), &path, &attrs) {
                if qself.is_some() {
                    self.sess.gated_spans.gate(sym::more_qualified_paths, path.span);
                }
                return expr;
            } else {
                (path.span, ExprKind::Path(qself, path))
            }
        } else {
            (path.span, ExprKind::Path(qself, path))
        };

        let expr = self.mk_expr(lo.to(hi), kind, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    /// Parse `'label: $expr`. The label is already parsed.
    fn parse_labeled_expr(
        &mut self,
        label: Label,
        attrs: AttrVec,
        consume_colon: bool,
    ) -> PResult<'a, P<Expr>> {
        let lo = label.ident.span;
        let label = Some(label);
        let ate_colon = self.eat(&token::Colon);
        let expr = if self.eat_keyword(kw::While) {
            self.parse_while_expr(label, lo, attrs)
        } else if self.eat_keyword(kw::For) {
            self.parse_for_expr(label, lo, attrs)
        } else if self.eat_keyword(kw::Loop) {
            self.parse_loop_expr(label, lo, attrs)
        } else if self.check(&token::OpenDelim(token::Brace)) || self.token.is_whole_block() {
            self.parse_block_expr(label, lo, BlockCheckMode::Default, attrs)
        } else {
            let msg = "expected `while`, `for`, `loop` or `{` after a label";
            self.struct_span_err(self.token.span, msg).span_label(self.token.span, msg).emit();
            // Continue as an expression in an effort to recover on `'label: non_block_expr`.
            self.parse_expr()
        }?;

        if !ate_colon && consume_colon {
            self.error_labeled_expr_must_be_followed_by_colon(lo, expr.span);
        }

        Ok(expr)
    }

    fn error_labeled_expr_must_be_followed_by_colon(&self, lo: Span, span: Span) {
        self.struct_span_err(span, "labeled expression must be followed by `:`")
            .span_label(lo, "the label")
            .span_suggestion_short(
                lo.shrink_to_hi(),
                "add `:` after the label",
                ": ".to_string(),
                Applicability::MachineApplicable,
            )
            .note("labels are used before loops and blocks, allowing e.g., `break 'label` to them")
            .emit();
    }

    /// Recover on the syntax `do catch { ... }` suggesting `try { ... }` instead.
    fn recover_do_catch(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        self.bump(); // `do`
        self.bump(); // `catch`

        let span_dc = lo.to(self.prev_token.span);
        self.struct_span_err(span_dc, "found removed `do catch` syntax")
            .span_suggestion(
                span_dc,
                "replace with the new syntax",
                "try".to_string(),
                Applicability::MachineApplicable,
            )
            .note("following RFC #2388, the new non-placeholder syntax is `try`")
            .emit();

        self.parse_try_block(lo, attrs)
    }

    /// Parse an expression if the token can begin one.
    fn parse_expr_opt(&mut self) -> PResult<'a, Option<P<Expr>>> {
        Ok(if self.token.can_begin_expr() { Some(self.parse_expr()?) } else { None })
    }

    /// Parse `"return" expr?`.
    fn parse_return_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let kind = ExprKind::Ret(self.parse_expr_opt()?);
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    /// Parse `"break" (('label (:? expr)?) | expr?)` with `"break"` token already eaten.
    /// If the label is followed immediately by a `:` token, the label and `:` are
    /// parsed as part of the expression (i.e. a labeled loop). The language team has
    /// decided in #87026 to require parentheses as a visual aid to avoid confusion if
    /// the break expression of an unlabeled break is a labeled loop (as in
    /// `break 'lbl: loop {}`); a labeled break with an unlabeled loop as its value
    /// expression only gets a warning for compatibility reasons; and a labeled break
    /// with a labeled loop does not even get a warning because there is no ambiguity.
    fn parse_break_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let mut label = self.eat_label();
        let kind = if label.is_some() && self.token == token::Colon {
            // The value expression can be a labeled loop, see issue #86948, e.g.:
            // `loop { break 'label: loop { break 'label 42; }; }`
            let lexpr = self.parse_labeled_expr(label.take().unwrap(), AttrVec::new(), true)?;
            self.struct_span_err(
                lexpr.span,
                "parentheses are required around this expression to avoid confusion with a labeled break expression",
            )
            .multipart_suggestion(
                "wrap the expression in parentheses",
                vec![
                    (lexpr.span.shrink_to_lo(), "(".to_string()),
                    (lexpr.span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MachineApplicable,
            )
            .emit();
            Some(lexpr)
        } else if self.token != token::OpenDelim(token::Brace)
            || !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
        {
            let expr = self.parse_expr_opt()?;
            if let Some(ref expr) = expr {
                if label.is_some()
                    && matches!(
                        expr.kind,
                        ExprKind::While(_, _, None)
                            | ExprKind::ForLoop(_, _, _, None)
                            | ExprKind::Loop(_, None)
                            | ExprKind::Block(_, None)
                    )
                {
                    self.sess.buffer_lint_with_diagnostic(
                        BREAK_WITH_LABEL_AND_LOOP,
                        lo.to(expr.span),
                        ast::CRATE_NODE_ID,
                        "this labeled break expression is easy to confuse with an unlabeled break with a labeled value expression",
                        BuiltinLintDiagnostics::BreakWithLabelAndLoop(expr.span),
                    );
                }
            }
            expr
        } else {
            None
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Break(label, kind), attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    /// Parse `"yield" expr?`.
    fn parse_yield_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let kind = ExprKind::Yield(self.parse_expr_opt()?);
        let span = lo.to(self.prev_token.span);
        self.sess.gated_spans.gate(sym::generators, span);
        let expr = self.mk_expr(span, kind, attrs);
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
            },
            None => Err(None),
        }
    }

    pub(super) fn parse_lit(&mut self) -> PResult<'a, Lit> {
        self.parse_opt_lit().ok_or_else(|| {
            let msg = format!("unexpected token: {}", super::token_descr(&self.token));
            self.struct_span_err(self.token.span, &msg)
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
                if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) =
                    next_token.kind
                {
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
                self.error_float_lits_must_have_int_part(&token);
            }
        }

        let token = recovered.as_ref().unwrap_or(&self.token);
        match Lit::from_token(token) {
            Ok(lit) => {
                self.bump();
                Some(lit)
            }
            Err(LitError::NotLiteral) => None,
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

    fn error_float_lits_must_have_int_part(&self, token: &Token) {
        self.struct_span_err(token.span, "float literals must have an integer part")
            .span_suggestion(
                token.span,
                "must have an integer part",
                pprust::token_to_string(token).into(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    fn report_lit_error(&self, err: LitError, lit: token::Lit, span: Span) {
        // Checks if `s` looks like i32 or u1234 etc.
        fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
            s.len() > 1 && s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
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
                    let msg = format!("invalid suffix `{}` for number literal", suf);
                    self.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)")
                        .emit();
                }
            }
            LitError::InvalidFloatSuffix => {
                let suf = suffix.expect("suffix error with no suffix").as_str();
                if looks_like_width_suffix(&['f'], &suf) {
                    // If it looks like a width, try to be helpful.
                    let msg = format!("invalid width `{}` for float literal", &suf[1..]);
                    self.struct_span_err(span, &msg).help("valid widths are 32 and 64").emit();
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
                self.struct_span_err(span, "integer literal is too large").emit();
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
                let mut err = self
                    .sess
                    .span_diagnostic
                    .struct_span_warn(sp, &format!("suffixes on {} are invalid", kind));
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
                    "see issue #60210 <https://github.com/rust-lang/rust/issues/60210> \
                     for more information",
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
    /// Keep this in sync with `Token::can_begin_literal_maybe_minus`.
    pub fn parse_literal_maybe_minus(&mut self) -> PResult<'a, P<Expr>> {
        maybe_whole_expr!(self);

        let lo = self.token.span;
        let minus_present = self.eat(&token::BinOp(token::Minus));
        let lit = self.parse_lit()?;
        let expr = self.mk_expr(lit.span, ExprKind::Lit(lit), AttrVec::new());

        if minus_present {
            Ok(self.mk_expr(
                lo.to(self.prev_token.span),
                self.mk_unary(UnOp::Neg, expr),
                AttrVec::new(),
            ))
        } else {
            Ok(expr)
        }
    }

    fn is_array_like_block(&mut self) -> bool {
        self.look_ahead(1, |t| matches!(t.kind, TokenKind::Ident(..) | TokenKind::Literal(_)))
            && self.look_ahead(2, |t| t == &token::Comma)
            && self.look_ahead(3, |t| t.can_begin_expr())
    }

    /// Emits a suggestion if it looks like the user meant an array but
    /// accidentally used braces, causing the code to be interpreted as a block
    /// expression.
    fn maybe_suggest_brackets_instead_of_braces(
        &mut self,
        lo: Span,
        attrs: AttrVec,
    ) -> Option<P<Expr>> {
        let mut snapshot = self.clone();
        match snapshot.parse_array_or_repeat_expr(attrs, token::Brace) {
            Ok(arr) => {
                let hi = snapshot.prev_token.span;
                self.struct_span_err(
                    arr.span,
                    "this code is interpreted as a block expression, not an array",
                )
                .multipart_suggestion(
                    "try using [] instead of {}",
                    vec![(lo, "[".to_owned()), (hi, "]".to_owned())],
                    Applicability::MaybeIncorrect,
                )
                .note("to define an array, one would use square brackets instead of curly braces")
                .emit();

                *self = snapshot;
                Some(self.mk_expr_err(arr.span))
            }
            Err(mut e) => {
                e.cancel();
                None
            }
        }
    }

    /// Parses a block or unsafe block.
    pub(super) fn parse_block_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        blk_mode: BlockCheckMode,
        mut attrs: AttrVec,
    ) -> PResult<'a, P<Expr>> {
        if self.is_array_like_block() {
            if let Some(arr) = self.maybe_suggest_brackets_instead_of_braces(lo, attrs.clone()) {
                return Ok(arr);
            }
        }

        if let Some(label) = opt_label {
            self.sess.gated_spans.gate(sym::label_break_value, label.ident.span);
        }

        if self.token.is_whole_block() {
            self.struct_span_err(self.token.span, "cannot use a `block` macro fragment here")
                .span_label(lo.to(self.token.span), "the `block` fragment is within this context")
                .emit();
        }

        let (inner_attrs, blk) = self.parse_block_common(lo, blk_mode)?;
        attrs.extend(inner_attrs);
        Ok(self.mk_expr(blk.span, ExprKind::Block(blk, opt_label), attrs))
    }

    /// Recover on an explicitly quantified closure expression, e.g., `for<'a> |x: &'a u8| *x + 1`.
    fn recover_quantified_closure_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        let _ = self.parse_late_bound_lifetime_defs()?;
        let span_for = lo.to(self.prev_token.span);
        let closure = self.parse_closure_expr(attrs)?;

        self.struct_span_err(span_for, "cannot introduce explicit parameters for a closure")
            .span_label(closure.span, "the parameters are attached to this closure")
            .span_suggestion(
                span_for,
                "remove the parameters",
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();

        Ok(self.mk_expr_err(lo.to(closure.span)))
    }

    /// Parses a closure expression (e.g., `move |args| expr`).
    fn parse_closure_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        let movability =
            if self.eat_keyword(kw::Static) { Movability::Static } else { Movability::Movable };

        let asyncness = if self.token.uninterpolated_span().rust_2018() {
            self.parse_asyncness()
        } else {
            Async::No
        };

        let capture_clause = self.parse_capture_clause()?;
        let decl = self.parse_fn_block_decl()?;
        let decl_hi = self.prev_token.span;
        let mut body = match decl.output {
            FnRetTy::Default(_) => {
                let restrictions = self.restrictions - Restrictions::STMT_EXPR;
                self.parse_expr_res(restrictions, None)?
            }
            _ => {
                // If an explicit return type is given, require a block to appear (RFC 968).
                let body_lo = self.token.span;
                self.parse_block_expr(None, body_lo, BlockCheckMode::Default, AttrVec::new())?
            }
        };

        if let Async::Yes { span, .. } = asyncness {
            // Feature-gate `async ||` closures.
            self.sess.gated_spans.gate(sym::async_closure, span);
        }

        if self.token.kind == TokenKind::Semi && self.token_cursor.frame.delim == DelimToken::Paren
        {
            // It is likely that the closure body is a block but where the
            // braces have been removed. We will recover and eat the next
            // statements later in the parsing process.
            body = self.mk_expr_err(body.span);
        }

        let body_span = body.span;

        let closure = self.mk_expr(
            lo.to(body.span),
            ExprKind::Closure(capture_clause, asyncness, movability, decl, body, lo.to(decl_hi)),
            attrs,
        );

        // Disable recovery for closure body
        let spans =
            ClosureSpans { whole_closure: closure.span, closing_pipe: decl_hi, body: body_span };
        self.current_closure = Some(spans);

        Ok(closure)
    }

    /// Parses an optional `move` prefix to a closure-like construct.
    fn parse_capture_clause(&mut self) -> PResult<'a, CaptureBy> {
        if self.eat_keyword(kw::Move) {
            // Check for `move async` and recover
            if self.check_keyword(kw::Async) {
                let move_async_span = self.token.span.with_lo(self.prev_token.span.data().lo);
                Err(self.incorrect_move_async_order_found(move_async_span))
            } else {
                Ok(CaptureBy::Value)
            }
        } else {
            Ok(CaptureBy::Ref)
        }
    }

    /// Parses the `|arg, arg|` header of a closure.
    fn parse_fn_block_decl(&mut self) -> PResult<'a, P<FnDecl>> {
        let inputs = if self.eat(&token::OrOr) {
            Vec::new()
        } else {
            self.expect(&token::BinOp(token::Or))?;
            let args = self
                .parse_seq_to_before_tokens(
                    &[&token::BinOp(token::Or), &token::OrOr],
                    SeqSep::trailing_allowed(token::Comma),
                    TokenExpectType::NoExpect,
                    |p| p.parse_fn_block_param(),
                )?
                .0;
            self.expect_or()?;
            args
        };
        let output =
            self.parse_ret_ty(AllowPlus::Yes, RecoverQPath::Yes, RecoverReturnSign::Yes)?;

        Ok(P(FnDecl { inputs, output }))
    }

    /// Parses a parameter in a closure header (e.g., `|arg, arg|`).
    fn parse_fn_block_param(&mut self) -> PResult<'a, Param> {
        let lo = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
            let pat = this.parse_pat_no_top_alt(PARAM_EXPECTED)?;
            let ty = if this.eat(&token::Colon) {
                this.parse_ty()?
            } else {
                this.mk_ty(this.prev_token.span, TyKind::Infer)
            };

            Ok((
                Param {
                    attrs: attrs.into(),
                    ty,
                    pat,
                    span: lo.to(this.token.span),
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                TrailingToken::MaybeComma,
            ))
        })
    }

    /// Parses an `if` expression (`if` token already eaten).
    fn parse_if_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let cond = self.parse_cond_expr()?;

        // Verify that the parsed `if` condition makes sense as a condition. If it is a block, then
        // verify that the last statement is either an implicit return (no `;`) or an explicit
        // return. This won't catch blocks with an explicit `return`, but that would be caught by
        // the dead code lint.
        let thn = if self.eat_keyword(kw::Else) || !cond.returns() {
            self.error_missing_if_cond(lo, cond.span)
        } else {
            let attrs = self.parse_outer_attributes()?.take_for_recovery(); // For recovery.
            let not_block = self.token != token::OpenDelim(token::Brace);
            let block = self.parse_block().map_err(|mut err| {
                if not_block {
                    err.span_label(lo, "this `if` expression has a condition, but no block");
                    if let ExprKind::Binary(_, _, ref right) = cond.kind {
                        if let ExprKind::Block(_, _) = right.kind {
                            err.help("maybe you forgot the right operand of the condition?");
                        }
                    }
                }
                err
            })?;
            self.error_on_if_block_attrs(lo, false, block.span, &attrs);
            block
        };
        let els = if self.eat_keyword(kw::Else) { Some(self.parse_else_expr()?) } else { None };
        Ok(self.mk_expr(lo.to(self.prev_token.span), ExprKind::If(cond, thn, els), attrs))
    }

    fn error_missing_if_cond(&self, lo: Span, span: Span) -> P<ast::Block> {
        let sp = self.sess.source_map().next_point(lo);
        self.struct_span_err(sp, "missing condition for `if` expression")
            .span_label(sp, "expected if condition here")
            .emit();
        self.mk_block_err(span)
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
    fn parse_let_expr(&mut self, attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let pat = self.parse_pat_allow_top_alt(None, RecoverComma::Yes, RecoverColon::Yes)?;
        self.expect(&token::Eq)?;
        let expr = self.with_res(self.restrictions | Restrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_assoc_expr_with(1 + prec_let_scrutinee_needs_par(), None.into())
        })?;
        let span = lo.to(expr.span);
        self.sess.gated_spans.gate(sym::let_chains, span);
        Ok(self.mk_expr(span, ExprKind::Let(pat, expr, span), attrs))
    }

    /// Parses an `else { ... }` expression (`else` token already eaten).
    fn parse_else_expr(&mut self) -> PResult<'a, P<Expr>> {
        let ctx_span = self.prev_token.span; // `else`
        let attrs = self.parse_outer_attributes()?.take_for_recovery(); // For recovery.
        let expr = if self.eat_keyword(kw::If) {
            self.parse_if_expr(AttrVec::new())?
        } else {
            let blk = self.parse_block()?;
            self.mk_expr(blk.span, ExprKind::Block(blk, None), AttrVec::new())
        };
        self.error_on_if_block_attrs(ctx_span, true, expr.span, &attrs);
        Ok(expr)
    }

    fn error_on_if_block_attrs(
        &self,
        ctx_span: Span,
        is_ctx_else: bool,
        branch_span: Span,
        attrs: &[ast::Attribute],
    ) {
        let (span, last) = match attrs {
            [] => return,
            [x0 @ xn] | [x0, .., xn] => (x0.span.to(xn.span), xn.span),
        };
        let ctx = if is_ctx_else { "else" } else { "if" };
        self.struct_span_err(last, "outer attributes are not allowed on `if` and `else` branches")
            .span_label(branch_span, "the attributes are attached to this branch")
            .span_label(ctx_span, format!("the branch belongs to this `{}`", ctx))
            .span_suggestion(
                span,
                "remove the attributes",
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    /// Parses `for <src_pat> in <src_expr> <src_loop_block>` (`for` token already eaten).
    fn parse_for_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        mut attrs: AttrVec,
    ) -> PResult<'a, P<Expr>> {
        // Record whether we are about to parse `for (`.
        // This is used below for recovery in case of `for ( $stuff ) $block`
        // in which case we will suggest `for $stuff $block`.
        let begin_paren = match self.token.kind {
            token::OpenDelim(token::Paren) => Some(self.token.span),
            _ => None,
        };

        let pat = self.parse_pat_allow_top_alt(None, RecoverComma::Yes, RecoverColon::Yes)?;
        if !self.eat_keyword(kw::In) {
            self.error_missing_in_for_loop();
        }
        self.check_for_for_in_in_typo(self.prev_token.span);
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;

        let pat = self.recover_parens_around_for_head(pat, begin_paren);

        let (iattrs, loop_block) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);

        let kind = ExprKind::ForLoop(pat, expr, loop_block, opt_label);
        Ok(self.mk_expr(lo.to(self.prev_token.span), kind, attrs))
    }

    fn error_missing_in_for_loop(&mut self) {
        let (span, msg, sugg) = if self.token.is_ident_named(sym::of) {
            // Possibly using JS syntax (#75311).
            let span = self.token.span;
            self.bump();
            (span, "try using `in` here instead", "in")
        } else {
            (self.prev_token.span.between(self.token.span), "try adding `in` here", " in ")
        };
        self.struct_span_err(span, "missing `in` in `for` loop")
            .span_suggestion_short(
                span,
                msg,
                sugg.into(),
                // Has been misleading, at least in the past (closed Issue #48492).
                Applicability::MaybeIncorrect,
            )
            .emit();
    }

    /// Parses a `while` or `while let` expression (`while` token already eaten).
    fn parse_while_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        mut attrs: AttrVec,
    ) -> PResult<'a, P<Expr>> {
        let cond = self.parse_cond_expr()?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        Ok(self.mk_expr(lo.to(self.prev_token.span), ExprKind::While(cond, body, opt_label), attrs))
    }

    /// Parses `loop { ... }` (`loop` token already eaten).
    fn parse_loop_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        mut attrs: AttrVec,
    ) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        Ok(self.mk_expr(lo.to(self.prev_token.span), ExprKind::Loop(body, opt_label), attrs))
    }

    fn eat_label(&mut self) -> Option<Label> {
        self.token.lifetime().map(|ident| {
            self.bump();
            Label { ident }
        })
    }

    /// Parses a `match ... { ... }` expression (`match` token already eaten).
    fn parse_match_expr(&mut self, mut attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let match_span = self.prev_token.span;
        let lo = self.prev_token.span;
        let scrutinee = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        if let Err(mut e) = self.expect(&token::OpenDelim(token::Brace)) {
            if self.token == token::Semi {
                e.span_suggestion_short(
                    match_span,
                    "try removing this `match`",
                    String::new(),
                    Applicability::MaybeIncorrect, // speculative
                );
            }
            return Err(e);
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
                    return Ok(self.mk_expr(span, ExprKind::Match(scrutinee, arms), attrs));
                }
            }
        }
        let hi = self.token.span;
        self.bump();
        Ok(self.mk_expr(lo.to(hi), ExprKind::Match(scrutinee, arms), attrs))
    }

    /// Attempt to recover from match arm body with statements and no surrounding braces.
    fn parse_arm_body_missing_braces(
        &mut self,
        first_expr: &P<Expr>,
        arrow_span: Span,
    ) -> Option<P<Expr>> {
        if self.token.kind != token::Semi {
            return None;
        }
        let start_snapshot = self.clone();
        let semi_sp = self.token.span;
        self.bump(); // `;`
        let mut stmts =
            vec![self.mk_stmt(first_expr.span, ast::StmtKind::Expr(first_expr.clone()))];
        let err = |this: &mut Parser<'_>, stmts: Vec<ast::Stmt>| {
            let span = stmts[0].span.to(stmts[stmts.len() - 1].span);
            let mut err = this.struct_span_err(span, "`match` arm body without braces");
            let (these, s, are) =
                if stmts.len() > 1 { ("these", "s", "are") } else { ("this", "", "is") };
            err.span_label(
                span,
                &format!(
                    "{these} statement{s} {are} not surrounded by a body",
                    these = these,
                    s = s,
                    are = are
                ),
            );
            err.span_label(arrow_span, "while parsing the `match` arm starting here");
            if stmts.len() > 1 {
                err.multipart_suggestion(
                    &format!("surround the statement{} with a body", s),
                    vec![
                        (span.shrink_to_lo(), "{ ".to_string()),
                        (span.shrink_to_hi(), " }".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            } else {
                err.span_suggestion(
                    semi_sp,
                    "use a comma to end a `match` arm expression",
                    ",".to_string(),
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
            this.mk_expr_err(span)
        };
        // We might have either a `,` -> `;` typo, or a block without braces. We need
        // a more subtle parsing strategy.
        loop {
            if self.token.kind == token::CloseDelim(token::Brace) {
                // We have reached the closing brace of the `match` expression.
                return Some(err(self, stmts));
            }
            if self.token.kind == token::Comma {
                *self = start_snapshot;
                return None;
            }
            let pre_pat_snapshot = self.clone();
            match self.parse_pat_no_top_alt(None) {
                Ok(_pat) => {
                    if self.token.kind == token::FatArrow {
                        // Reached arm end.
                        *self = pre_pat_snapshot;
                        return Some(err(self, stmts));
                    }
                }
                Err(mut err) => {
                    err.cancel();
                }
            }

            *self = pre_pat_snapshot;
            match self.parse_stmt_without_recovery(true, ForceCollect::No) {
                // Consume statements for as long as possible.
                Ok(Some(stmt)) => {
                    stmts.push(stmt);
                }
                Ok(None) => {
                    *self = start_snapshot;
                    break;
                }
                // We couldn't parse either yet another statement missing it's
                // enclosing block nor the next arm's pattern or closing brace.
                Err(mut stmt_err) => {
                    stmt_err.cancel();
                    *self = start_snapshot;
                    break;
                }
            }
        }
        None
    }

    pub(super) fn parse_arm(&mut self) -> PResult<'a, Arm> {
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
            let lo = this.token.span;
            let pat = this.parse_pat_allow_top_alt(None, RecoverComma::Yes, RecoverColon::Yes)?;
            let guard = if this.eat_keyword(kw::If) {
                let if_span = this.prev_token.span;
                let cond = this.parse_expr()?;
                if let ExprKind::Let(..) = cond.kind {
                    // Remove the last feature gating of a `let` expression since it's stable.
                    this.sess.gated_spans.ungate_last(sym::let_chains, cond.span);
                    let span = if_span.to(cond.span);
                    this.sess.gated_spans.gate(sym::if_let_guard, span);
                }
                Some(cond)
            } else {
                None
            };
            let arrow_span = this.token.span;
            this.expect(&token::FatArrow)?;
            let arm_start_span = this.token.span;

            let expr = this.parse_expr_res(Restrictions::STMT_EXPR, None).map_err(|mut err| {
                err.span_label(arrow_span, "while parsing the `match` arm starting here");
                err
            })?;

            let require_comma = classify::expr_requires_semi_to_be_stmt(&expr)
                && this.token != token::CloseDelim(token::Brace);

            let hi = this.prev_token.span;

            if require_comma {
                let sm = this.sess.source_map();
                if let Some(body) = this.parse_arm_body_missing_braces(&expr, arrow_span) {
                    let span = body.span;
                    return Ok((
                        ast::Arm {
                            attrs: attrs.into(),
                            pat,
                            guard,
                            body,
                            span,
                            id: DUMMY_NODE_ID,
                            is_placeholder: false,
                        },
                        TrailingToken::None,
                    ));
                }
                this.expect_one_of(&[token::Comma], &[token::CloseDelim(token::Brace)]).map_err(
                    |mut err| {
                        match (sm.span_to_lines(expr.span), sm.span_to_lines(arm_start_span)) {
                            (Ok(ref expr_lines), Ok(ref arm_start_lines))
                                if arm_start_lines.lines[0].end_col
                                    == expr_lines.lines[0].end_col
                                    && expr_lines.lines.len() == 2
                                    && this.token == token::FatArrow =>
                            {
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
                                    arm_start_span.shrink_to_hi(),
                                    "missing a comma here to end this `match` arm",
                                    ",".to_owned(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            _ => {
                                err.span_label(
                                    arrow_span,
                                    "while parsing the `match` arm starting here",
                                );
                            }
                        }
                        err
                    },
                )?;
            } else {
                this.eat(&token::Comma);
            }

            Ok((
                ast::Arm {
                    attrs: attrs.into(),
                    pat,
                    guard,
                    body: expr,
                    span: lo.to(hi),
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                TrailingToken::None,
            ))
        })
    }

    /// Parses a `try {...}` expression (`try` token already eaten).
    fn parse_try_block(&mut self, span_lo: Span, mut attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        if self.eat_keyword(kw::Catch) {
            let mut error = self.struct_span_err(
                self.prev_token.span,
                "keyword `catch` cannot follow a `try` block",
            );
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
        self.token.is_keyword(kw::Do)
            && self.is_keyword_ahead(1, &[kw::Catch])
            && self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))
            && !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_try_block(&self) -> bool {
        self.token.is_keyword(kw::Try)
            && self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace))
            && self.token.uninterpolated_span().rust_2018()
    }

    /// Parses an `async move? {...}` expression.
    fn parse_async_block(&mut self, mut attrs: AttrVec) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        self.expect_keyword(kw::Async)?;
        let capture_clause = self.parse_capture_clause()?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let kind = ExprKind::Async(capture_clause, DUMMY_NODE_ID, body);
        Ok(self.mk_expr(lo.to(self.prev_token.span), kind, attrs))
    }

    fn is_async_block(&self) -> bool {
        self.token.is_keyword(kw::Async)
            && ((
                // `async move {`
                self.is_keyword_ahead(1, &[kw::Move])
                    && self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))
            ) || (
                // `async {`
                self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace))
            ))
    }

    fn is_certainly_not_a_block(&self) -> bool {
        self.look_ahead(1, |t| t.is_ident())
            && (
                // `{ ident, ` cannot start a block.
                self.look_ahead(2, |t| t == &token::Comma)
                    || self.look_ahead(2, |t| t == &token::Colon)
                        && (
                            // `{ ident: token, ` cannot start a block.
                            self.look_ahead(4, |t| t == &token::Comma) ||
                // `{ ident: ` cannot start a block unless it's a type ascription `ident: Type`.
                self.look_ahead(3, |t| !t.can_begin_type())
                        )
            )
    }

    fn maybe_parse_struct_expr(
        &mut self,
        qself: Option<&ast::QSelf>,
        path: &ast::Path,
        attrs: &AttrVec,
    ) -> Option<PResult<'a, P<Expr>>> {
        let struct_allowed = !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
        if struct_allowed || self.is_certainly_not_a_block() {
            if let Err(err) = self.expect(&token::OpenDelim(token::Brace)) {
                return Some(Err(err));
            }
            let expr = self.parse_struct_expr(qself.cloned(), path.clone(), attrs.clone(), true);
            if let (Ok(expr), false) = (&expr, struct_allowed) {
                // This is a struct literal, but we don't can't accept them here.
                self.error_struct_lit_not_allowed_here(path.span, expr.span);
            }
            return Some(expr);
        }
        None
    }

    fn error_struct_lit_not_allowed_here(&self, lo: Span, sp: Span) {
        self.struct_span_err(sp, "struct literals are not allowed here")
            .multipart_suggestion(
                "surround the struct literal with parentheses",
                vec![(lo.shrink_to_lo(), "(".to_string()), (sp.shrink_to_hi(), ")".to_string())],
                Applicability::MachineApplicable,
            )
            .emit();
    }

    pub(super) fn parse_struct_fields(
        &mut self,
        pth: ast::Path,
        recover: bool,
        close_delim: token::DelimToken,
    ) -> PResult<'a, (Vec<ExprField>, ast::StructRest, bool)> {
        let mut fields = Vec::new();
        let mut base = ast::StructRest::None;
        let mut recover_async = false;

        let mut async_block_err = |e: &mut DiagnosticBuilder<'_>, span: Span| {
            recover_async = true;
            e.span_label(span, "`async` blocks are only allowed in Rust 2018 or later");
            e.help(&format!("set `edition = \"{}\"` in `Cargo.toml`", LATEST_STABLE_EDITION));
            e.note("for more on editions, read https://doc.rust-lang.org/edition-guide");
        };

        while self.token != token::CloseDelim(close_delim) {
            if self.eat(&token::DotDot) {
                let exp_span = self.prev_token.span;
                // We permit `.. }` on the left-hand side of a destructuring assignment.
                if self.check(&token::CloseDelim(close_delim)) {
                    self.sess.gated_spans.gate(sym::destructuring_assignment, self.prev_token.span);
                    base = ast::StructRest::Rest(self.prev_token.span.shrink_to_hi());
                    break;
                }
                match self.parse_expr() {
                    Ok(e) => base = ast::StructRest::Base(e),
                    Err(mut e) if recover => {
                        e.emit();
                        self.recover_stmt();
                    }
                    Err(e) => return Err(e),
                }
                self.recover_struct_comma_after_dotdot(exp_span);
                break;
            }

            let recovery_field = self.find_struct_error_after_field_looking_code();
            let parsed_field = match self.parse_expr_field() {
                Ok(f) => Some(f),
                Err(mut e) => {
                    if pth == kw::Async {
                        async_block_err(&mut e, pth.span);
                    } else {
                        e.span_label(pth.span, "while parsing this struct");
                    }
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
                    None
                }
            };

            match self.expect_one_of(&[token::Comma], &[token::CloseDelim(close_delim)]) {
                Ok(_) => {
                    if let Some(f) = parsed_field.or(recovery_field) {
                        // Only include the field if there's no parse error for the field name.
                        fields.push(f);
                    }
                }
                Err(mut e) => {
                    if pth == kw::Async {
                        async_block_err(&mut e, pth.span);
                    } else {
                        e.span_label(pth.span, "while parsing this struct");
                        if let Some(f) = recovery_field {
                            fields.push(f);
                            e.span_suggestion(
                                self.prev_token.span.shrink_to_hi(),
                                "try adding a comma",
                                ",".into(),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                    if !recover {
                        return Err(e);
                    }
                    e.emit();
                    self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                    self.eat(&token::Comma);
                }
            }
        }
        Ok((fields, base, recover_async))
    }

    /// Precondition: already parsed the '{'.
    pub(super) fn parse_struct_expr(
        &mut self,
        qself: Option<ast::QSelf>,
        pth: ast::Path,
        attrs: AttrVec,
        recover: bool,
    ) -> PResult<'a, P<Expr>> {
        let lo = pth.span;
        let (fields, base, recover_async) =
            self.parse_struct_fields(pth.clone(), recover, token::Brace)?;
        let span = lo.to(self.token.span);
        self.expect(&token::CloseDelim(token::Brace))?;
        let expr = if recover_async {
            ExprKind::Err
        } else {
            ExprKind::Struct(P(ast::StructExpr { qself, path: pth, fields, rest: base }))
        };
        Ok(self.mk_expr(span, expr, attrs))
    }

    /// Use in case of error after field-looking code: `S { foo: () with a }`.
    fn find_struct_error_after_field_looking_code(&self) -> Option<ExprField> {
        match self.token.ident() {
            Some((ident, is_raw))
                if (is_raw || !ident.is_reserved())
                    && self.look_ahead(1, |t| *t == token::Colon) =>
            {
                Some(ast::ExprField {
                    ident,
                    span: self.token.span,
                    expr: self.mk_expr_err(self.token.span),
                    is_shorthand: false,
                    attrs: AttrVec::new(),
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                })
            }
            _ => None,
        }
    }

    fn recover_struct_comma_after_dotdot(&mut self, span: Span) {
        if self.token != token::Comma {
            return;
        }
        self.struct_span_err(
            span.to(self.prev_token.span),
            "cannot use a comma after the base struct",
        )
        .span_suggestion_short(
            self.token.span,
            "remove this comma",
            String::new(),
            Applicability::MachineApplicable,
        )
        .note("the base struct must always be the last field")
        .emit();
        self.recover_stmt();
    }

    /// Parses `ident (COLON expr)?`.
    fn parse_expr_field(&mut self) -> PResult<'a, ExprField> {
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
            let lo = this.token.span;

            // Check if a colon exists one ahead. This means we're parsing a fieldname.
            let is_shorthand = !this.look_ahead(1, |t| t == &token::Colon || t == &token::Eq);
            let (ident, expr) = if is_shorthand {
                // Mimic `x: x` for the `x` field shorthand.
                let ident = this.parse_ident_common(false)?;
                let path = ast::Path::from_ident(ident);
                (ident, this.mk_expr(ident.span, ExprKind::Path(None, path), AttrVec::new()))
            } else {
                let ident = this.parse_field_name()?;
                this.error_on_eq_field_init(ident);
                this.bump(); // `:`
                (ident, this.parse_expr()?)
            };

            Ok((
                ast::ExprField {
                    ident,
                    span: lo.to(expr.span),
                    expr,
                    is_shorthand,
                    attrs: attrs.into(),
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                TrailingToken::MaybeComma,
            ))
        })
    }

    /// Check for `=`. This means the source incorrectly attempts to
    /// initialize a field with an eq rather than a colon.
    fn error_on_eq_field_init(&self, field_name: Ident) {
        if self.token != token::Eq {
            return;
        }

        self.struct_span_err(self.token.span, "expected `:`, found `=`")
            .span_suggestion(
                field_name.span.shrink_to_hi().to(self.token.span),
                "replace equals symbol with a colon",
                ":".to_string(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    fn err_dotdotdot_syntax(&self, span: Span) {
        self.struct_span_err(span, "unexpected token: `...`")
            .span_suggestion(
                span,
                "use `..` for an exclusive range",
                "..".to_owned(),
                Applicability::MaybeIncorrect,
            )
            .span_suggestion(
                span,
                "or `..=` for an inclusive range",
                "..=".to_owned(),
                Applicability::MaybeIncorrect,
            )
            .emit();
    }

    fn err_larrow_operator(&self, span: Span) {
        self.struct_span_err(span, "unexpected token: `<-`")
            .span_suggestion(
                span,
                "if you meant to write a comparison against a negative value, add a \
             space in between `<` and `-`",
                "< -".to_string(),
                Applicability::MaybeIncorrect,
            )
            .emit();
    }

    fn mk_assign_op(&self, binop: BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ExprKind {
        ExprKind::AssignOp(binop, lhs, rhs)
    }

    fn mk_range(
        &mut self,
        start: Option<P<Expr>>,
        end: Option<P<Expr>>,
        limits: RangeLimits,
    ) -> ExprKind {
        if end.is_none() && limits == RangeLimits::Closed {
            self.inclusive_range_with_incorrect_end(self.prev_token.span);
            ExprKind::Err
        } else {
            ExprKind::Range(start, end, limits)
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

    fn mk_await_expr(&mut self, self_arg: P<Expr>, lo: Span) -> P<Expr> {
        let span = lo.to(self.prev_token.span);
        let await_expr = self.mk_expr(span, ExprKind::Await(self_arg), AttrVec::new());
        self.recover_from_await_method_call();
        await_expr
    }

    crate fn mk_expr(&self, span: Span, kind: ExprKind, attrs: AttrVec) -> P<Expr> {
        P(Expr { kind, span, attrs, id: DUMMY_NODE_ID, tokens: None })
    }

    pub(super) fn mk_expr_err(&self, span: Span) -> P<Expr> {
        self.mk_expr(span, ExprKind::Err, AttrVec::new())
    }

    /// Create expression span ensuring the span of the parent node
    /// is larger than the span of lhs and rhs, including the attributes.
    fn mk_expr_sp(&self, lhs: &P<Expr>, lhs_span: Span, rhs_span: Span) -> Span {
        lhs.attrs
            .iter()
            .find(|a| a.style == AttrStyle::Outer)
            .map_or(lhs_span, |a| a.span)
            .to(rhs_span)
    }

    fn collect_tokens_for_expr(
        &mut self,
        attrs: AttrWrapper,
        f: impl FnOnce(&mut Self, Vec<ast::Attribute>) -> PResult<'a, P<Expr>>,
    ) -> PResult<'a, P<Expr>> {
        self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
            let res = f(this, attrs)?;
            let trailing = if this.restrictions.contains(Restrictions::STMT_EXPR)
                && this.token.kind == token::Semi
            {
                TrailingToken::Semi
            } else {
                // FIXME - pass this through from the place where we know
                // we need a comma, rather than assuming that `#[attr] expr,`
                // always captures a trailing comma
                TrailingToken::MaybeComma
            };
            Ok((res, trailing))
        })
    }
}
