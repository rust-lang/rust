// ignore-tidy-filelength

use core::mem;
use core::ops::{Bound, ControlFlow};

use ast::mut_visit::{self, MutVisitor};
use ast::token::IdentIsRaw;
use ast::{CoroutineKind, ForLoopKind, GenBlockKind, MatchKind, Pat, Path, PathSegment, Recovered};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, InvisibleOrigin, MetaVarKind, Token, TokenKind};
use rustc_ast::tokenstream::TokenTree;
use rustc_ast::util::case::Case;
use rustc_ast::util::classify;
use rustc_ast::util::parser::{AssocOp, ExprPrecedence, Fixity, prec_let_scrutinee_needs_par};
use rustc_ast::visit::{Visitor, walk_expr};
use rustc_ast::{
    self as ast, AnonConst, Arm, AssignOp, AssignOpKind, AttrStyle, AttrVec, BinOp, BinOpKind,
    BlockCheckMode, CaptureBy, ClosureBinder, DUMMY_NODE_ID, Expr, ExprField, ExprKind, FnDecl,
    FnRetTy, Label, MacCall, MetaItemLit, Movability, Param, RangeLimits, StmtKind, Ty, TyKind,
    UnOp, UnsafeBinderCastKind, YieldKind,
};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{Applicability, Diag, PResult, StashKey, Subdiagnostic};
use rustc_literal_escaper::unescape_char;
use rustc_macros::Subdiagnostic;
use rustc_session::errors::{ExprParenthesesNeeded, report_lit_error};
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::BREAK_WITH_LABEL_AND_LOOP;
use rustc_span::edition::Edition;
use rustc_span::source_map::{self, Spanned};
use rustc_span::{BytePos, ErrorGuaranteed, Ident, Pos, Span, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::instrument;

use super::diagnostics::SnapshotParser;
use super::pat::{CommaRecoveryMode, Expected, RecoverColon, RecoverComma};
use super::ty::{AllowPlus, RecoverQPath, RecoverReturnSign};
use super::{
    AttrWrapper, BlockMode, ClosureSpans, ExpTokenPair, ForceCollect, Parser, PathStyle,
    Restrictions, SemiColonMode, SeqSep, TokenType, Trailing, UsePreAttrPos,
};
use crate::{errors, exp, maybe_recover_from_interpolated_ty_qpath};

#[derive(Debug)]
pub(super) enum DestructuredFloat {
    /// 1e2
    Single(Symbol, Span),
    /// 1.
    TrailingDot(Symbol, Span, Span),
    /// 1.2 | 1.2e3
    MiddleDot(Symbol, Span, Span, Symbol, Span),
    /// Invalid
    Error,
}

impl<'a> Parser<'a> {
    /// Parses an expression.
    #[inline]
    pub fn parse_expr(&mut self) -> PResult<'a, P<Expr>> {
        self.current_closure.take();

        let attrs = self.parse_outer_attributes()?;
        self.parse_expr_res(Restrictions::empty(), attrs).map(|res| res.0)
    }

    /// Parses an expression, forcing tokens to be collected.
    pub fn parse_expr_force_collect(&mut self) -> PResult<'a, P<Expr>> {
        self.current_closure.take();

        // If the expression is associative (e.g. `1 + 2`), then any preceding
        // outer attribute actually belongs to the first inner sub-expression.
        // In which case we must use the pre-attr pos to include the attribute
        // in the collected tokens for the outer expression.
        let pre_attr_pos = self.collect_pos();
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens(
            Some(pre_attr_pos),
            AttrWrapper::empty(),
            ForceCollect::Yes,
            |this, _empty_attrs| {
                let (expr, is_assoc) = this.parse_expr_res(Restrictions::empty(), attrs)?;
                let use_pre_attr_pos =
                    if is_assoc { UsePreAttrPos::Yes } else { UsePreAttrPos::No };
                Ok((expr, Trailing::No, use_pre_attr_pos))
            },
        )
    }

    pub fn parse_expr_anon_const(&mut self) -> PResult<'a, AnonConst> {
        self.parse_expr().map(|value| AnonConst { id: DUMMY_NODE_ID, value })
    }

    fn parse_expr_catch_underscore(&mut self, restrictions: Restrictions) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_outer_attributes()?;
        match self.parse_expr_res(restrictions, attrs) {
            Ok((expr, _)) => Ok(expr),
            Err(err) => match self.token.ident() {
                Some((Ident { name: kw::Underscore, .. }, IdentIsRaw::No))
                    if self.may_recover() && self.look_ahead(1, |t| t == &token::Comma) =>
                {
                    // Special-case handling of `foo(_, _, _)`
                    let guar = err.emit();
                    self.bump();
                    Ok(self.mk_expr(self.prev_token.span, ExprKind::Err(guar)))
                }
                _ => Err(err),
            },
        }
    }

    /// Parses a sequence of expressions delimited by parentheses.
    fn parse_expr_paren_seq(&mut self) -> PResult<'a, ThinVec<P<Expr>>> {
        self.parse_paren_comma_seq(|p| p.parse_expr_catch_underscore(Restrictions::empty()))
            .map(|(r, _)| r)
    }

    /// Parses an expression, subject to the given restrictions.
    #[inline]
    pub(super) fn parse_expr_res(
        &mut self,
        r: Restrictions,
        attrs: AttrWrapper,
    ) -> PResult<'a, (P<Expr>, bool)> {
        self.with_res(r, |this| this.parse_expr_assoc_with(Bound::Unbounded, attrs))
    }

    /// Parses an associative expression with operators of at least `min_prec` precedence.
    /// The `bool` in the return value indicates if it was an assoc expr, i.e. with an operator
    /// followed by a subexpression (e.g. `1 + 2`).
    pub(super) fn parse_expr_assoc_with(
        &mut self,
        min_prec: Bound<ExprPrecedence>,
        attrs: AttrWrapper,
    ) -> PResult<'a, (P<Expr>, bool)> {
        let lhs = if self.token.is_range_separator() {
            return self.parse_expr_prefix_range(attrs).map(|res| (res, false));
        } else {
            self.parse_expr_prefix(attrs)?
        };
        self.parse_expr_assoc_rest_with(min_prec, false, lhs)
    }

    /// Parses the rest of an associative expression (i.e. the part after the lhs) with operators
    /// of at least `min_prec` precedence. The `bool` in the return value indicates if something
    /// was actually parsed.
    pub(super) fn parse_expr_assoc_rest_with(
        &mut self,
        min_prec: Bound<ExprPrecedence>,
        starts_stmt: bool,
        mut lhs: P<Expr>,
    ) -> PResult<'a, (P<Expr>, bool)> {
        let mut parsed_something = false;
        if !self.should_continue_as_assoc_expr(&lhs) {
            return Ok((lhs, parsed_something));
        }

        self.expected_token_types.insert(TokenType::Operator);
        while let Some(op) = self.check_assoc_op() {
            let lhs_span = self.interpolated_or_expr_span(&lhs);
            let cur_op_span = self.token.span;
            let restrictions = if op.node.is_assign_like() {
                self.restrictions & Restrictions::NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            let prec = op.node.precedence();
            if match min_prec {
                Bound::Included(min_prec) => prec < min_prec,
                Bound::Excluded(min_prec) => prec <= min_prec,
                Bound::Unbounded => false,
            } {
                break;
            }
            // Check for deprecated `...` syntax
            if self.token == token::DotDotDot && op.node == AssocOp::Range(RangeLimits::Closed) {
                self.err_dotdotdot_syntax(self.token.span);
            }

            if self.token == token::LArrow {
                self.err_larrow_operator(self.token.span);
            }

            parsed_something = true;
            self.bump();
            if op.node.is_comparison() {
                if let Some(expr) = self.check_no_chained_comparison(&lhs, &op)? {
                    return Ok((expr, parsed_something));
                }
            }

            // Look for JS' `===` and `!==` and recover
            if let AssocOp::Binary(bop @ BinOpKind::Eq | bop @ BinOpKind::Ne) = op.node
                && self.token == token::Eq
                && self.prev_token.span.hi() == self.token.span.lo()
            {
                let sp = op.span.to(self.token.span);
                let sugg = bop.as_str().into();
                let invalid = format!("{sugg}=");
                self.dcx().emit_err(errors::InvalidComparisonOperator {
                    span: sp,
                    invalid: invalid.clone(),
                    sub: errors::InvalidComparisonOperatorSub::Correctable {
                        span: sp,
                        invalid,
                        correct: sugg,
                    },
                });
                self.bump();
            }

            // Look for PHP's `<>` and recover
            if op.node == AssocOp::Binary(BinOpKind::Lt)
                && self.token == token::Gt
                && self.prev_token.span.hi() == self.token.span.lo()
            {
                let sp = op.span.to(self.token.span);
                self.dcx().emit_err(errors::InvalidComparisonOperator {
                    span: sp,
                    invalid: "<>".into(),
                    sub: errors::InvalidComparisonOperatorSub::Correctable {
                        span: sp,
                        invalid: "<>".into(),
                        correct: "!=".into(),
                    },
                });
                self.bump();
            }

            // Look for C++'s `<=>` and recover
            if op.node == AssocOp::Binary(BinOpKind::Le)
                && self.token == token::Gt
                && self.prev_token.span.hi() == self.token.span.lo()
            {
                let sp = op.span.to(self.token.span);
                self.dcx().emit_err(errors::InvalidComparisonOperator {
                    span: sp,
                    invalid: "<=>".into(),
                    sub: errors::InvalidComparisonOperatorSub::Spaceship(sp),
                });
                self.bump();
            }

            if self.prev_token == token::Plus
                && self.token == token::Plus
                && self.prev_token.span.between(self.token.span).is_empty()
            {
                let op_span = self.prev_token.span.to(self.token.span);
                // Eat the second `+`
                self.bump();
                lhs = self.recover_from_postfix_increment(lhs, op_span, starts_stmt)?;
                continue;
            }

            if self.prev_token == token::Minus
                && self.token == token::Minus
                && self.prev_token.span.between(self.token.span).is_empty()
                && !self.look_ahead(1, |tok| tok.can_begin_expr())
            {
                let op_span = self.prev_token.span.to(self.token.span);
                // Eat the second `-`
                self.bump();
                lhs = self.recover_from_postfix_decrement(lhs, op_span, starts_stmt)?;
                continue;
            }

            let op = op.node;
            // Special cases:
            if op == AssocOp::Cast {
                lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Cast)?;
                continue;
            } else if let AssocOp::Range(limits) = op {
                // If we didn't have to handle `x..`/`x..=`, it would be pretty easy to
                // generalise it to the Fixity::None code.
                lhs = self.parse_expr_range(prec, lhs, limits, cur_op_span)?;
                break;
            }

            let min_prec = match op.fixity() {
                Fixity::Right => Bound::Included(prec),
                Fixity::Left | Fixity::None => Bound::Excluded(prec),
            };
            let (rhs, _) = self.with_res(restrictions - Restrictions::STMT_EXPR, |this| {
                let attrs = this.parse_outer_attributes()?;
                this.parse_expr_assoc_with(min_prec, attrs)
            })?;

            let span = self.mk_expr_sp(&lhs, lhs_span, rhs.span);
            lhs = match op {
                AssocOp::Binary(ast_op) => {
                    let binary = self.mk_binary(source_map::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(span, binary)
                }
                AssocOp::Assign => self.mk_expr(span, ExprKind::Assign(lhs, rhs, cur_op_span)),
                AssocOp::AssignOp(aop) => {
                    let aopexpr = self.mk_assign_op(source_map::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(span, aopexpr)
                }
                AssocOp::Cast | AssocOp::Range(_) => {
                    self.dcx().span_bug(span, "AssocOp should have been handled by special case")
                }
            };
        }

        Ok((lhs, parsed_something))
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
            (true, Some(AssocOp::Binary(
                BinOpKind::Mul | // `{ 42 } *foo = bar;` or `{ 42 } * 3`
                BinOpKind::Sub | // `{ 42 } -5`
                BinOpKind::Add | // `{ 42 } + 42` (unary plus)
                BinOpKind::And | // `{ 42 } &&x` (#61475) or `{ 42 } && if x { 1 } else { 0 }`
                BinOpKind::Or | // `{ 42 } || 42` ("logical or" or closure)
                BinOpKind::BitOr // `{ 42 } | 42` or `{ 42 } |x| 42`
            ))) => {
                // These cases are ambiguous and can't be identified in the parser alone.
                //
                // Bitwise AND is left out because guessing intent is hard. We can make
                // suggestions based on the assumption that double-refs are rarely intentional,
                // and closures are distinct enough that they don't get mixed up with their
                // return value.
                let sp = self.psess.source_map().start_point(self.token.span);
                self.psess.ambiguous_block_expr_parse.borrow_mut().insert(sp, lhs.span);
                false
            }
            (true, Some(op)) if !op.can_continue_expr_unambiguously() => false,
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
        self.dcx().emit_err(errors::FoundExprWouldBeStmt {
            span: self.token.span,
            token: self.token,
            suggestion: ExprParenthesesNeeded::surrounding(lhs.span),
        });
    }

    /// Possibly translate the current token to an associative operator.
    /// The method does not advance the current token.
    ///
    /// Also performs recovery for `and` / `or` which are mistaken for `&&` and `||` respectively.
    pub(super) fn check_assoc_op(&self) -> Option<Spanned<AssocOp>> {
        let (op, span) = match (AssocOp::from_token(&self.token), self.token.ident()) {
            // When parsing const expressions, stop parsing when encountering `>`.
            (
                Some(
                    AssocOp::Binary(BinOpKind::Shr | BinOpKind::Gt | BinOpKind::Ge)
                    | AssocOp::AssignOp(AssignOpKind::ShrAssign),
                ),
                _,
            ) if self.restrictions.contains(Restrictions::CONST_EXPR) => {
                return None;
            }
            // When recovering patterns as expressions, stop parsing when encountering an
            // assignment `=`, an alternative `|`, or a range `..`.
            (
                Some(
                    AssocOp::Assign
                    | AssocOp::AssignOp(_)
                    | AssocOp::Binary(BinOpKind::BitOr)
                    | AssocOp::Range(_),
                ),
                _,
            ) if self.restrictions.contains(Restrictions::IS_PAT) => {
                return None;
            }
            (Some(op), _) => (op, self.token.span),
            (None, Some((Ident { name: sym::and, span }, IdentIsRaw::No)))
                if self.may_recover() =>
            {
                self.dcx().emit_err(errors::InvalidLogicalOperator {
                    span: self.token.span,
                    incorrect: "and".into(),
                    sub: errors::InvalidLogicalOperatorSub::Conjunction(self.token.span),
                });
                (AssocOp::Binary(BinOpKind::And), span)
            }
            (None, Some((Ident { name: sym::or, span }, IdentIsRaw::No))) if self.may_recover() => {
                self.dcx().emit_err(errors::InvalidLogicalOperator {
                    span: self.token.span,
                    incorrect: "or".into(),
                    sub: errors::InvalidLogicalOperatorSub::Disjunction(self.token.span),
                });
                (AssocOp::Binary(BinOpKind::Or), span)
            }
            _ => return None,
        };
        Some(source_map::respan(span, op))
    }

    /// Checks if this expression is a successfully parsed statement.
    fn expr_is_complete(&self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::STMT_EXPR) && classify::expr_is_complete(e)
    }

    /// Parses `x..y`, `x..=y`, and `x..`/`x..=`.
    /// The other two variants are handled in `parse_prefix_range_expr` below.
    fn parse_expr_range(
        &mut self,
        prec: ExprPrecedence,
        lhs: P<Expr>,
        limits: RangeLimits,
        cur_op_span: Span,
    ) -> PResult<'a, P<Expr>> {
        let rhs = if self.is_at_start_of_range_notation_rhs() {
            let maybe_lt = self.token;
            let attrs = self.parse_outer_attributes()?;
            Some(
                self.parse_expr_assoc_with(Bound::Excluded(prec), attrs)
                    .map_err(|err| self.maybe_err_dotdotlt_syntax(maybe_lt, err))?
                    .0,
            )
        } else {
            None
        };
        let rhs_span = rhs.as_ref().map_or(cur_op_span, |x| x.span);
        let span = self.mk_expr_sp(&lhs, lhs.span, rhs_span);
        let range = self.mk_range(Some(lhs), rhs, limits);
        Ok(self.mk_expr(span, range))
    }

    fn is_at_start_of_range_notation_rhs(&self) -> bool {
        if self.token.can_begin_expr() {
            // Parse `for i in 1.. { }` as infinite loop, not as `for i in (1..{})`.
            if self.token == token::OpenBrace {
                return !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
            }
            true
        } else {
            false
        }
    }

    /// Parses prefix-forms of range notation: `..expr`, `..`, `..=expr`.
    fn parse_expr_prefix_range(&mut self, attrs: AttrWrapper) -> PResult<'a, P<Expr>> {
        if !attrs.is_empty() {
            let err = errors::DotDotRangeAttribute { span: self.token.span };
            self.dcx().emit_err(err);
        }

        // Check for deprecated `...` syntax.
        if self.token == token::DotDotDot {
            self.err_dotdotdot_syntax(self.token.span);
        }

        debug_assert!(
            self.token.is_range_separator(),
            "parse_prefix_range_expr: token {:?} is not DotDot/DotDotEq",
            self.token
        );

        let limits = match self.token.kind {
            token::DotDot => RangeLimits::HalfOpen,
            _ => RangeLimits::Closed,
        };
        let op = AssocOp::from_token(&self.token);
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens_for_expr(attrs, |this, attrs| {
            let lo = this.token.span;
            let maybe_lt = this.look_ahead(1, |t| t.clone());
            this.bump();
            let (span, opt_end) = if this.is_at_start_of_range_notation_rhs() {
                // RHS must be parsed with more associativity than the dots.
                let attrs = this.parse_outer_attributes()?;
                this.parse_expr_assoc_with(Bound::Excluded(op.unwrap().precedence()), attrs)
                    .map(|(x, _)| (lo.to(x.span), Some(x)))
                    .map_err(|err| this.maybe_err_dotdotlt_syntax(maybe_lt, err))?
            } else {
                (lo, None)
            };
            let range = this.mk_range(None, opt_end, limits);
            Ok(this.mk_expr_with_attrs(span, range, attrs))
        })
    }

    /// Parses a prefix-unary-operator expr.
    fn parse_expr_prefix(&mut self, attrs: AttrWrapper) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        macro_rules! make_it {
            ($this:ident, $attrs:expr, |this, _| $body:expr) => {
                $this.collect_tokens_for_expr($attrs, |$this, attrs| {
                    let (hi, ex) = $body?;
                    Ok($this.mk_expr_with_attrs(lo.to(hi), ex, attrs))
                })
            };
        }

        let this = self;

        // Note: when adding new unary operators, don't forget to adjust TokenKind::can_begin_expr()
        match this.token.uninterpolate().kind {
            // `!expr`
            token::Bang => make_it!(this, attrs, |this, _| this.parse_expr_unary(lo, UnOp::Not)),
            // `~expr`
            token::Tilde => make_it!(this, attrs, |this, _| this.recover_tilde_expr(lo)),
            // `-expr`
            token::Minus => {
                make_it!(this, attrs, |this, _| this.parse_expr_unary(lo, UnOp::Neg))
            }
            // `*expr`
            token::Star => {
                make_it!(this, attrs, |this, _| this.parse_expr_unary(lo, UnOp::Deref))
            }
            // `&expr` and `&&expr`
            token::And | token::AndAnd => {
                make_it!(this, attrs, |this, _| this.parse_expr_borrow(lo))
            }
            // `+lit`
            token::Plus if this.look_ahead(1, |tok| tok.is_numeric_lit()) => {
                let mut err = errors::LeadingPlusNotSupported {
                    span: lo,
                    remove_plus: None,
                    add_parentheses: None,
                };

                // a block on the LHS might have been intended to be an expression instead
                if let Some(sp) = this.psess.ambiguous_block_expr_parse.borrow().get(&lo) {
                    err.add_parentheses = Some(ExprParenthesesNeeded::surrounding(*sp));
                } else {
                    err.remove_plus = Some(lo);
                }
                this.dcx().emit_err(err);

                this.bump();
                let attrs = this.parse_outer_attributes()?;
                this.parse_expr_prefix(attrs)
            }
            // Recover from `++x`:
            token::Plus if this.look_ahead(1, |t| *t == token::Plus) => {
                let starts_stmt =
                    this.prev_token == token::Semi || this.prev_token == token::CloseBrace;
                let pre_span = this.token.span.to(this.look_ahead(1, |t| t.span));
                // Eat both `+`s.
                this.bump();
                this.bump();

                let operand_expr = this.parse_expr_dot_or_call(attrs)?;
                this.recover_from_prefix_increment(operand_expr, pre_span, starts_stmt)
            }
            token::Ident(..) if this.token.is_keyword(kw::Box) => {
                make_it!(this, attrs, |this, _| this.parse_expr_box(lo))
            }
            token::Ident(..) if this.may_recover() && this.is_mistaken_not_ident_negation() => {
                make_it!(this, attrs, |this, _| this.recover_not_expr(lo))
            }
            _ => return this.parse_expr_dot_or_call(attrs),
        }
    }

    fn parse_expr_prefix_common(&mut self, lo: Span) -> PResult<'a, (Span, P<Expr>)> {
        self.bump();
        let attrs = self.parse_outer_attributes()?;
        let expr = if self.token.is_range_separator() {
            self.parse_expr_prefix_range(attrs)
        } else {
            self.parse_expr_prefix(attrs)
        }?;
        let span = self.interpolated_or_expr_span(&expr);
        Ok((lo.to(span), expr))
    }

    fn parse_expr_unary(&mut self, lo: Span, op: UnOp) -> PResult<'a, (Span, ExprKind)> {
        let (span, expr) = self.parse_expr_prefix_common(lo)?;
        Ok((span, self.mk_unary(op, expr)))
    }

    /// Recover on `~expr` in favor of `!expr`.
    fn recover_tilde_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        self.dcx().emit_err(errors::TildeAsUnaryOperator(lo));

        self.parse_expr_unary(lo, UnOp::Not)
    }

    /// Parse `box expr` - this syntax has been removed, but we still parse this
    /// for now to provide a more useful error
    fn parse_expr_box(&mut self, box_kw: Span) -> PResult<'a, (Span, ExprKind)> {
        let (span, expr) = self.parse_expr_prefix_common(box_kw)?;
        // Make a multipart suggestion instead of `span_to_snippet` in case source isn't available
        let box_kw_and_lo = box_kw.until(self.interpolated_or_expr_span(&expr));
        let hi = span.shrink_to_hi();
        let sugg = errors::AddBoxNew { box_kw_and_lo, hi };
        let guar = self.dcx().emit_err(errors::BoxSyntaxRemoved { span, sugg });
        Ok((span, ExprKind::Err(guar)))
    }

    fn is_mistaken_not_ident_negation(&self) -> bool {
        let token_cannot_continue_expr = |t: &Token| match t.uninterpolate().kind {
            // These tokens can start an expression after `!`, but
            // can't continue an expression after an ident
            token::Ident(name, is_raw) => token::ident_can_begin_expr(name, t.span, is_raw),
            token::Literal(..) | token::Pound => true,
            _ => t.is_metavar_expr(),
        };
        self.token.is_ident_named(sym::not) && self.look_ahead(1, token_cannot_continue_expr)
    }

    /// Recover on `not expr` in favor of `!expr`.
    fn recover_not_expr(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        let negated_token = self.look_ahead(1, |t| *t);

        let sub_diag = if negated_token.is_numeric_lit() {
            errors::NotAsNegationOperatorSub::SuggestNotBitwise
        } else if negated_token.is_bool_lit() {
            errors::NotAsNegationOperatorSub::SuggestNotLogical
        } else {
            errors::NotAsNegationOperatorSub::SuggestNotDefault
        };

        self.dcx().emit_err(errors::NotAsNegationOperator {
            negated: negated_token.span,
            negated_desc: super::token_descr(&negated_token),
            // Span the `not` plus trailing whitespace to avoid
            // trailing whitespace after the `!` in our suggestion
            sub: sub_diag(
                self.psess.source_map().span_until_non_whitespace(lo.to(negated_token.span)),
            ),
        });

        self.parse_expr_unary(lo, UnOp::Not)
    }

    /// Returns the span of expr if it was not interpolated, or the span of the interpolated token.
    fn interpolated_or_expr_span(&self, expr: &Expr) -> Span {
        match self.prev_token.kind {
            token::NtIdent(..) | token::NtLifetime(..) => self.prev_token.span,
            token::CloseInvisible(InvisibleOrigin::MetaVar(_)) => {
                // `expr.span` is the interpolated span, because invisible open
                // and close delims both get marked with the same span, one
                // that covers the entire thing between them. (See
                // `rustc_expand::mbe::transcribe::transcribe`.)
                self.prev_token.span
            }
            _ => expr.span,
        }
    }

    fn parse_assoc_op_cast(
        &mut self,
        lhs: P<Expr>,
        lhs_span: Span,
        expr_kind: fn(P<Expr>, P<Ty>) -> ExprKind,
    ) -> PResult<'a, P<Expr>> {
        let mk_expr = |this: &mut Self, lhs: P<Expr>, rhs: P<Ty>| {
            this.mk_expr(this.mk_expr_sp(&lhs, lhs_span, rhs.span), expr_kind(lhs, rhs))
        };

        // Save the state of the parser before parsing type normally, in case there is a
        // LessThan comparison after this cast.
        let parser_snapshot_before_type = self.clone();
        let cast_expr = match self.parse_as_cast_ty() {
            Ok(rhs) => mk_expr(self, lhs, rhs),
            Err(type_err) => {
                if !self.may_recover() {
                    return Err(type_err);
                }

                // Rewind to before attempting to parse the type with generics, to recover
                // from situations like `x as usize < y` in which we first tried to parse
                // `usize < y` as a type with generic arguments.
                let parser_snapshot_after_type = mem::replace(self, parser_snapshot_before_type);

                // Check for typo of `'a: loop { break 'a }` with a missing `'`.
                match (&lhs.kind, &self.token.kind) {
                    (
                        // `foo: `
                        ExprKind::Path(None, ast::Path { segments, .. }),
                        token::Ident(kw::For | kw::Loop | kw::While, IdentIsRaw::No),
                    ) if let [segment] = segments.as_slice() => {
                        let snapshot = self.create_snapshot_for_diagnostic();
                        let label = Label {
                            ident: Ident::from_str_and_span(
                                &format!("'{}", segment.ident),
                                segment.ident.span,
                            ),
                        };
                        match self.parse_expr_labeled(label, false) {
                            Ok(expr) => {
                                type_err.cancel();
                                self.dcx().emit_err(errors::MalformedLoopLabel {
                                    span: label.ident.span,
                                    suggestion: label.ident.span.shrink_to_lo(),
                                });
                                return Ok(expr);
                            }
                            Err(err) => {
                                err.cancel();
                                self.restore_snapshot(snapshot);
                            }
                        }
                    }
                    _ => {}
                }

                match self.parse_path(PathStyle::Expr) {
                    Ok(path) => {
                        let span_after_type = parser_snapshot_after_type.token.span;
                        let expr = mk_expr(
                            self,
                            lhs,
                            self.mk_ty(path.span, TyKind::Path(None, path.clone())),
                        );

                        let args_span = self.look_ahead(1, |t| t.span).to(span_after_type);
                        let suggestion = errors::ComparisonOrShiftInterpretedAsGenericSugg {
                            left: expr.span.shrink_to_lo(),
                            right: expr.span.shrink_to_hi(),
                        };

                        match self.token.kind {
                            token::Lt => {
                                self.dcx().emit_err(errors::ComparisonInterpretedAsGeneric {
                                    comparison: self.token.span,
                                    r#type: path,
                                    args: args_span,
                                    suggestion,
                                })
                            }
                            token::Shl => self.dcx().emit_err(errors::ShiftInterpretedAsGeneric {
                                shift: self.token.span,
                                r#type: path,
                                args: args_span,
                                suggestion,
                            }),
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

                        // Keep `x as usize` as an expression in AST and continue parsing.
                        expr
                    }
                    Err(path_err) => {
                        // Couldn't parse as a path, return original error and parser state.
                        path_err.cancel();
                        *self = parser_snapshot_after_type;
                        return Err(type_err);
                    }
                }
            }
        };

        // Try to parse a postfix operator such as `.`, `?`, or index (`[]`)
        // after a cast. If one is present, emit an error then return a valid
        // parse tree; For something like `&x as T[0]` will be as if it was
        // written `((&x) as T)[0]`.

        let span = cast_expr.span;

        let with_postfix = self.parse_expr_dot_or_call_with(AttrVec::new(), cast_expr, span)?;

        // Check if an illegal postfix operator has been added after the cast.
        // If the resulting expression is not a cast, it is an illegal postfix operator.
        if !matches!(with_postfix.kind, ExprKind::Cast(_, _)) {
            let msg = format!(
                "cast cannot be followed by {}",
                match with_postfix.kind {
                    ExprKind::Index(..) => "indexing",
                    ExprKind::Try(_) => "`?`",
                    ExprKind::Field(_, _) => "a field access",
                    ExprKind::MethodCall(_) => "a method call",
                    ExprKind::Call(_, _) => "a function call",
                    ExprKind::Await(_, _) => "`.await`",
                    ExprKind::Use(_, _) => "`.use`",
                    ExprKind::Match(_, _, MatchKind::Postfix) => "a postfix match",
                    ExprKind::Err(_) => return Ok(with_postfix),
                    _ => unreachable!("parse_dot_or_call_expr_with_ shouldn't produce this"),
                }
            );
            let mut err = self.dcx().struct_span_err(span, msg);

            let suggest_parens = |err: &mut Diag<'_>| {
                let suggestions = vec![
                    (span.shrink_to_lo(), "(".to_string()),
                    (span.shrink_to_hi(), ")".to_string()),
                ];
                err.multipart_suggestion(
                    "try surrounding the expression in parentheses",
                    suggestions,
                    Applicability::MachineApplicable,
                );
            };

            suggest_parens(&mut err);

            err.emit();
        };
        Ok(with_postfix)
    }

    /// Parse `& mut? <expr>` or `& raw [ const | mut ] <expr>`.
    fn parse_expr_borrow(&mut self, lo: Span) -> PResult<'a, (Span, ExprKind)> {
        self.expect_and()?;
        let has_lifetime = self.token.is_lifetime() && self.look_ahead(1, |t| t != &token::Colon);
        let lifetime = has_lifetime.then(|| self.expect_lifetime()); // For recovery, see below.
        let (borrow_kind, mutbl) = self.parse_borrow_modifiers();
        let attrs = self.parse_outer_attributes()?;
        let expr = if self.token.is_range_separator() {
            self.parse_expr_prefix_range(attrs)
        } else {
            self.parse_expr_prefix(attrs)
        }?;
        let hi = self.interpolated_or_expr_span(&expr);
        let span = lo.to(hi);
        if let Some(lt) = lifetime {
            self.error_remove_borrow_lifetime(span, lt.ident.span.until(expr.span));
        }

        // Add expected tokens if we parsed `&raw` as an expression.
        // This will make sure we see "expected `const`, `mut`", and
        // guides recovery in case we write `&raw expr`.
        if borrow_kind == ast::BorrowKind::Ref
            && mutbl == ast::Mutability::Not
            && matches!(&expr.kind, ExprKind::Path(None, p) if *p == kw::Raw)
        {
            self.expected_token_types.insert(TokenType::KwMut);
            self.expected_token_types.insert(TokenType::KwConst);
        }

        Ok((span, ExprKind::AddrOf(borrow_kind, mutbl, expr)))
    }

    fn error_remove_borrow_lifetime(&self, span: Span, lt_span: Span) {
        self.dcx().emit_err(errors::LifetimeInBorrowExpression { span, lifetime_span: lt_span });
    }

    /// Parse `mut?` or `raw [ const | mut ]`.
    fn parse_borrow_modifiers(&mut self) -> (ast::BorrowKind, ast::Mutability) {
        if self.check_keyword(exp!(Raw)) && self.look_ahead(1, Token::is_mutability) {
            // `raw [ const | mut ]`.
            let found_raw = self.eat_keyword(exp!(Raw));
            assert!(found_raw);
            let mutability = self.parse_const_or_mut().unwrap();
            (ast::BorrowKind::Raw, mutability)
        } else {
            // `mut?`
            (ast::BorrowKind::Ref, self.parse_mutability())
        }
    }

    /// Parses `a.b` or `a(13)` or `a[4]` or just `a`.
    fn parse_expr_dot_or_call(&mut self, attrs: AttrWrapper) -> PResult<'a, P<Expr>> {
        self.collect_tokens_for_expr(attrs, |this, attrs| {
            let base = this.parse_expr_bottom()?;
            let span = this.interpolated_or_expr_span(&base);
            this.parse_expr_dot_or_call_with(attrs, base, span)
        })
    }

    pub(super) fn parse_expr_dot_or_call_with(
        &mut self,
        mut attrs: ast::AttrVec,
        mut e: P<Expr>,
        lo: Span,
    ) -> PResult<'a, P<Expr>> {
        let mut res = ensure_sufficient_stack(|| {
            loop {
                let has_question =
                    if self.prev_token == TokenKind::Ident(kw::Return, IdentIsRaw::No) {
                        // We are using noexpect here because we don't expect a `?` directly after
                        // a `return` which could be suggested otherwise.
                        self.eat_noexpect(&token::Question)
                    } else {
                        self.eat(exp!(Question))
                    };
                if has_question {
                    // `expr?`
                    e = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Try(e));
                    continue;
                }
                let has_dot = if self.prev_token == TokenKind::Ident(kw::Return, IdentIsRaw::No) {
                    // We are using noexpect here because we don't expect a `.` directly after
                    // a `return` which could be suggested otherwise.
                    self.eat_noexpect(&token::Dot)
                } else if self.token == TokenKind::RArrow && self.may_recover() {
                    // Recovery for `expr->suffix`.
                    self.bump();
                    let span = self.prev_token.span;
                    self.dcx().emit_err(errors::ExprRArrowCall { span });
                    true
                } else {
                    self.eat(exp!(Dot))
                };
                if has_dot {
                    // expr.f
                    e = self.parse_dot_suffix_expr(lo, e)?;
                    continue;
                }
                if self.expr_is_complete(&e) {
                    return Ok(e);
                }
                e = match self.token.kind {
                    token::OpenParen => self.parse_expr_fn_call(lo, e),
                    token::OpenBracket => self.parse_expr_index(lo, e)?,
                    _ => return Ok(e),
                }
            }
        });

        // Stitch the list of outer attributes onto the return value. A little
        // bit ugly, but the best way given the current code structure.
        if !attrs.is_empty()
            && let Ok(expr) = &mut res
        {
            mem::swap(&mut expr.attrs, &mut attrs);
            expr.attrs.extend(attrs)
        }
        res
    }

    pub(super) fn parse_dot_suffix_expr(
        &mut self,
        lo: Span,
        base: P<Expr>,
    ) -> PResult<'a, P<Expr>> {
        // At this point we've consumed something like `expr.` and `self.token` holds the token
        // after the dot.
        match self.token.uninterpolate().kind {
            token::Ident(..) => self.parse_dot_suffix(base, lo),
            token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) => {
                let ident_span = self.token.span;
                self.bump();
                Ok(self.mk_expr_tuple_field_access(lo, ident_span, base, symbol, suffix))
            }
            token::Literal(token::Lit { kind: token::Float, symbol, suffix }) => {
                Ok(match self.break_up_float(symbol, self.token.span) {
                    // 1e2
                    DestructuredFloat::Single(sym, _sp) => {
                        // `foo.1e2`: a single complete dot access, fully consumed. We end up with
                        // the `1e2` token in `self.prev_token` and the following token in
                        // `self.token`.
                        let ident_span = self.token.span;
                        self.bump();
                        self.mk_expr_tuple_field_access(lo, ident_span, base, sym, suffix)
                    }
                    // 1.
                    DestructuredFloat::TrailingDot(sym, ident_span, dot_span) => {
                        // `foo.1.`: a single complete dot access and the start of another.
                        // We end up with the `sym` (`1`) token in `self.prev_token` and a dot in
                        // `self.token`.
                        assert!(suffix.is_none());
                        self.token = Token::new(token::Ident(sym, IdentIsRaw::No), ident_span);
                        self.bump_with((Token::new(token::Dot, dot_span), self.token_spacing));
                        self.mk_expr_tuple_field_access(lo, ident_span, base, sym, None)
                    }
                    // 1.2 | 1.2e3
                    DestructuredFloat::MiddleDot(
                        sym1,
                        ident1_span,
                        _dot_span,
                        sym2,
                        ident2_span,
                    ) => {
                        // `foo.1.2` (or `foo.1.2e3`): two complete dot accesses. We end up with
                        // the `sym2` (`2` or `2e3`) token in `self.prev_token` and the following
                        // token in `self.token`.
                        let next_token2 =
                            Token::new(token::Ident(sym2, IdentIsRaw::No), ident2_span);
                        self.bump_with((next_token2, self.token_spacing));
                        self.bump();
                        let base1 =
                            self.mk_expr_tuple_field_access(lo, ident1_span, base, sym1, None);
                        self.mk_expr_tuple_field_access(lo, ident2_span, base1, sym2, suffix)
                    }
                    DestructuredFloat::Error => base,
                })
            }
            _ => {
                self.error_unexpected_after_dot();
                Ok(base)
            }
        }
    }

    fn error_unexpected_after_dot(&self) {
        let actual = super::token_descr(&self.token);
        let span = self.token.span;
        let sm = self.psess.source_map();
        let (span, actual) = match (&self.token.kind, self.subparser_name) {
            (token::Eof, Some(_)) if let Ok(snippet) = sm.span_to_snippet(sm.next_point(span)) => {
                (span.shrink_to_hi(), format!("`{}`", snippet))
            }
            (token::CloseInvisible(InvisibleOrigin::MetaVar(_)), _) => {
                // No need to report an error. This case will only occur when parsing a pasted
                // metavariable, and we should have emitted an error when parsing the macro call in
                // the first place. E.g. in this code:
                // ```
                // macro_rules! m { ($e:expr) => { $e }; }
                //
                // fn main() {
                //     let f = 1;
                //     m!(f.);
                // }
                // ```
                // we'll get an error "unexpected token: `)` when parsing the `m!(f.)`, so we don't
                // want to issue a second error when parsing the expansion `«f.»` (where `«`/`»`
                // represent the invisible delimiters).
                self.dcx().span_delayed_bug(span, "bad dot expr in metavariable");
                return;
            }
            _ => (span, actual),
        };
        self.dcx().emit_err(errors::UnexpectedTokenAfterDot { span, actual });
    }

    /// We need an identifier or integer, but the next token is a float.
    /// Break the float into components to extract the identifier or integer.
    ///
    /// See also [`TokenKind::break_two_token_op`] which does similar splitting of `>>` into `>`.
    //
    // FIXME: With current `TokenCursor` it's hard to break tokens into more than 2
    //  parts unless those parts are processed immediately. `TokenCursor` should either
    //  support pushing "future tokens" (would be also helpful to `break_and_eat`), or
    //  we should break everything including floats into more basic proc-macro style
    //  tokens in the lexer (probably preferable).
    pub(super) fn break_up_float(&self, float: Symbol, span: Span) -> DestructuredFloat {
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
                panic!("unexpected character in a float token: {c:?}")
            }
        }
        if !ident_like.is_empty() {
            components.push(IdentLike(ident_like));
        }

        // With proc macros the span can refer to anything, the source may be too short,
        // or too long, or non-ASCII. It only makes sense to break our span into components
        // if its underlying text is identical to our float literal.
        let can_take_span_apart =
            || self.span_to_snippet(span).as_deref() == Ok(float_str).as_deref();

        match &*components {
            // 1e2
            [IdentLike(i)] => {
                DestructuredFloat::Single(Symbol::intern(i), span)
            }
            // 1.
            [IdentLike(left), Punct('.')] => {
                let (left_span, dot_span) = if can_take_span_apart() {
                    let left_span = span.with_hi(span.lo() + BytePos::from_usize(left.len()));
                    let dot_span = span.with_lo(left_span.hi());
                    (left_span, dot_span)
                } else {
                    (span, span)
                };
                let left = Symbol::intern(left);
                DestructuredFloat::TrailingDot(left, left_span, dot_span)
            }
            // 1.2 | 1.2e3
            [IdentLike(left), Punct('.'), IdentLike(right)] => {
                let (left_span, dot_span, right_span) = if can_take_span_apart() {
                    let left_span = span.with_hi(span.lo() + BytePos::from_usize(left.len()));
                    let dot_span = span.with_lo(left_span.hi()).with_hi(left_span.hi() + BytePos(1));
                    let right_span = span.with_lo(dot_span.hi());
                    (left_span, dot_span, right_span)
                } else {
                    (span, span, span)
                };
                let left = Symbol::intern(left);
                let right = Symbol::intern(right);
                DestructuredFloat::MiddleDot(left, left_span, dot_span, right, right_span)
            }
            // 1e+ | 1e- (recovered)
            [IdentLike(_), Punct('+' | '-')] |
            // 1e+2 | 1e-2
            [IdentLike(_), Punct('+' | '-'), IdentLike(_)] |
            // 1.2e+ | 1.2e-
            [IdentLike(_), Punct('.'), IdentLike(_), Punct('+' | '-')] |
            // 1.2e+3 | 1.2e-3
            [IdentLike(_), Punct('.'), IdentLike(_), Punct('+' | '-'), IdentLike(_)] => {
                // See the FIXME about `TokenCursor` above.
                self.error_unexpected_after_dot();
                DestructuredFloat::Error
            }
            _ => panic!("unexpected components in a float token: {components:?}"),
        }
    }

    /// Parse the field access used in offset_of, matched by `$(e:expr)+`.
    /// Currently returns a list of idents. However, it should be possible in
    /// future to also do array indices, which might be arbitrary expressions.
    fn parse_floating_field_access(&mut self) -> PResult<'a, Vec<Ident>> {
        let mut fields = Vec::new();
        let mut trailing_dot = None;

        loop {
            // This is expected to use a metavariable $(args:expr)+, but the builtin syntax
            // could be called directly. Calling `parse_expr` allows this function to only
            // consider `Expr`s.
            let expr = self.parse_expr()?;
            let mut current = &expr;
            let start_idx = fields.len();
            loop {
                match current.kind {
                    ExprKind::Field(ref left, right) => {
                        // Field access is read right-to-left.
                        fields.insert(start_idx, right);
                        trailing_dot = None;
                        current = left;
                    }
                    // Parse this both to give helpful error messages and to
                    // verify it can be done with this parser setup.
                    ExprKind::Index(ref left, ref _right, span) => {
                        self.dcx().emit_err(errors::ArrayIndexInOffsetOf(span));
                        current = left;
                    }
                    ExprKind::Lit(token::Lit {
                        kind: token::Float | token::Integer,
                        symbol,
                        suffix,
                    }) => {
                        if let Some(suffix) = suffix {
                            self.expect_no_tuple_index_suffix(current.span, suffix);
                        }
                        match self.break_up_float(symbol, current.span) {
                            // 1e2
                            DestructuredFloat::Single(sym, sp) => {
                                trailing_dot = None;
                                fields.insert(start_idx, Ident::new(sym, sp));
                            }
                            // 1.
                            DestructuredFloat::TrailingDot(sym, sym_span, dot_span) => {
                                assert!(suffix.is_none());
                                trailing_dot = Some(dot_span);
                                fields.insert(start_idx, Ident::new(sym, sym_span));
                            }
                            // 1.2 | 1.2e3
                            DestructuredFloat::MiddleDot(
                                symbol1,
                                span1,
                                _dot_span,
                                symbol2,
                                span2,
                            ) => {
                                trailing_dot = None;
                                fields.insert(start_idx, Ident::new(symbol2, span2));
                                fields.insert(start_idx, Ident::new(symbol1, span1));
                            }
                            DestructuredFloat::Error => {
                                trailing_dot = None;
                                fields.insert(start_idx, Ident::new(symbol, self.prev_token.span));
                            }
                        }
                        break;
                    }
                    ExprKind::Path(None, Path { ref segments, .. }) => {
                        match &segments[..] {
                            [PathSegment { ident, args: None, .. }] => {
                                trailing_dot = None;
                                fields.insert(start_idx, *ident)
                            }
                            _ => {
                                self.dcx().emit_err(errors::InvalidOffsetOf(current.span));
                                break;
                            }
                        }
                        break;
                    }
                    _ => {
                        self.dcx().emit_err(errors::InvalidOffsetOf(current.span));
                        break;
                    }
                }
            }

            if self.token.kind.close_delim().is_some() || self.token.kind == token::Comma {
                break;
            } else if trailing_dot.is_none() {
                // This loop should only repeat if there is a trailing dot.
                self.dcx().emit_err(errors::InvalidOffsetOf(self.token.span));
                break;
            }
        }
        if let Some(dot) = trailing_dot {
            self.dcx().emit_err(errors::InvalidOffsetOf(dot));
        }
        Ok(fields.into_iter().collect())
    }

    fn mk_expr_tuple_field_access(
        &self,
        lo: Span,
        ident_span: Span,
        base: P<Expr>,
        field: Symbol,
        suffix: Option<Symbol>,
    ) -> P<Expr> {
        if let Some(suffix) = suffix {
            self.expect_no_tuple_index_suffix(ident_span, suffix);
        }
        self.mk_expr(lo.to(ident_span), ExprKind::Field(base, Ident::new(field, ident_span)))
    }

    /// Parse a function call expression, `expr(...)`.
    fn parse_expr_fn_call(&mut self, lo: Span, fun: P<Expr>) -> P<Expr> {
        let snapshot = if self.token == token::OpenParen {
            Some((self.create_snapshot_for_diagnostic(), fun.kind.clone()))
        } else {
            None
        };
        let open_paren = self.token.span;

        let seq = self
            .parse_expr_paren_seq()
            .map(|args| self.mk_expr(lo.to(self.prev_token.span), self.mk_call(fun, args)));
        match self.maybe_recover_struct_lit_bad_delims(lo, open_paren, seq, snapshot) {
            Ok(expr) => expr,
            Err(err) => self.recover_seq_parse_error(exp!(OpenParen), exp!(CloseParen), lo, err),
        }
    }

    /// If we encounter a parser state that looks like the user has written a `struct` literal with
    /// parentheses instead of braces, recover the parser state and provide suggestions.
    #[instrument(skip(self, seq, snapshot), level = "trace")]
    fn maybe_recover_struct_lit_bad_delims(
        &mut self,
        lo: Span,
        open_paren: Span,
        seq: PResult<'a, P<Expr>>,
        snapshot: Option<(SnapshotParser<'a>, ExprKind)>,
    ) -> PResult<'a, P<Expr>> {
        match (self.may_recover(), seq, snapshot) {
            (true, Err(err), Some((mut snapshot, ExprKind::Path(None, path)))) => {
                snapshot.bump(); // `(`
                match snapshot.parse_struct_fields(path.clone(), false, exp!(CloseParen)) {
                    Ok((fields, ..)) if snapshot.eat(exp!(CloseParen)) => {
                        // We are certain we have `Enum::Foo(a: 3, b: 4)`, suggest
                        // `Enum::Foo { a: 3, b: 4 }` or `Enum::Foo(3, 4)`.
                        self.restore_snapshot(snapshot);
                        let close_paren = self.prev_token.span;
                        let span = lo.to(close_paren);
                        // filter shorthand fields
                        let fields: Vec<_> =
                            fields.into_iter().filter(|field| !field.is_shorthand).collect();

                        let guar = if !fields.is_empty() &&
                            // `token.kind` should not be compared here.
                            // This is because the `snapshot.token.kind` is treated as the same as
                            // that of the open delim in `TokenTreesReader::parse_token_tree`, even
                            // if they are different.
                            self.span_to_snippet(close_paren).is_ok_and(|snippet| snippet == ")")
                        {
                            err.cancel();
                            self.dcx()
                                .create_err(errors::ParenthesesWithStructFields {
                                    span,
                                    r#type: path,
                                    braces_for_struct: errors::BracesForStructLiteral {
                                        first: open_paren,
                                        second: close_paren,
                                    },
                                    no_fields_for_fn: errors::NoFieldsForFnCall {
                                        fields: fields
                                            .into_iter()
                                            .map(|field| field.span.until(field.expr.span))
                                            .collect(),
                                    },
                                })
                                .emit()
                        } else {
                            err.emit()
                        };
                        Ok(self.mk_expr_err(span, guar))
                    }
                    Ok(_) => Err(err),
                    Err(err2) => {
                        err2.cancel();
                        Err(err)
                    }
                }
            }
            (_, seq, _) => seq,
        }
    }

    /// Parse an indexing expression `expr[...]`.
    fn parse_expr_index(&mut self, lo: Span, base: P<Expr>) -> PResult<'a, P<Expr>> {
        let prev_span = self.prev_token.span;
        let open_delim_span = self.token.span;
        self.bump(); // `[`
        let index = self.parse_expr()?;
        self.suggest_missing_semicolon_before_array(prev_span, open_delim_span)?;
        self.expect(exp!(CloseBracket))?;
        Ok(self.mk_expr(
            lo.to(self.prev_token.span),
            self.mk_index(base, index, open_delim_span.to(self.prev_token.span)),
        ))
    }

    /// Assuming we have just parsed `.`, continue parsing into an expression.
    fn parse_dot_suffix(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        if self.token_uninterpolated_span().at_least_rust_2018() && self.eat_keyword(exp!(Await)) {
            return Ok(self.mk_await_expr(self_arg, lo));
        }

        if self.eat_keyword(exp!(Use)) {
            let use_span = self.prev_token.span;
            self.psess.gated_spans.gate(sym::ergonomic_clones, use_span);
            return Ok(self.mk_use_expr(self_arg, lo));
        }

        // Post-fix match
        if self.eat_keyword(exp!(Match)) {
            let match_span = self.prev_token.span;
            self.psess.gated_spans.gate(sym::postfix_match, match_span);
            return self.parse_match_block(lo, match_span, self_arg, MatchKind::Postfix);
        }

        // Parse a postfix `yield`.
        if self.eat_keyword(exp!(Yield)) {
            let yield_span = self.prev_token.span;
            self.psess.gated_spans.gate(sym::yield_expr, yield_span);
            return Ok(
                self.mk_expr(lo.to(yield_span), ExprKind::Yield(YieldKind::Postfix(self_arg)))
            );
        }

        let fn_span_lo = self.token.span;
        let mut seg = self.parse_path_segment(PathStyle::Expr, None)?;
        self.check_trailing_angle_brackets(&seg, &[exp!(OpenParen)]);
        self.check_turbofish_missing_angle_brackets(&mut seg);

        if self.check(exp!(OpenParen)) {
            // Method call `expr.f()`
            let args = self.parse_expr_paren_seq()?;
            let fn_span = fn_span_lo.to(self.prev_token.span);
            let span = lo.to(self.prev_token.span);
            Ok(self.mk_expr(
                span,
                ExprKind::MethodCall(Box::new(ast::MethodCall {
                    seg,
                    receiver: self_arg,
                    args,
                    span: fn_span,
                })),
            ))
        } else {
            // Field access `expr.f`
            let span = lo.to(self.prev_token.span);
            if let Some(args) = seg.args {
                // See `StashKey::GenericInFieldExpr` for more info on why we stash this.
                self.dcx()
                    .create_err(errors::FieldExpressionWithGeneric(args.span()))
                    .stash(seg.ident.span, StashKey::GenericInFieldExpr);
            }

            Ok(self.mk_expr(span, ExprKind::Field(self_arg, seg.ident)))
        }
    }

    /// At the bottom (top?) of the precedence hierarchy,
    /// Parses things like parenthesized exprs, macros, `return`, etc.
    ///
    /// N.B., this does not parse outer attributes, and is private because it only works
    /// correctly if called from `parse_expr_dot_or_call`.
    fn parse_expr_bottom(&mut self) -> PResult<'a, P<Expr>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);

        let span = self.token.span;
        if let Some(expr) = self.eat_metavar_seq_with_matcher(
            |mv_kind| matches!(mv_kind, MetaVarKind::Expr { .. }),
            |this| {
                // Force collection (as opposed to just `parse_expr`) is required to avoid the
                // attribute duplication seen in #138478.
                let expr = this.parse_expr_force_collect();
                // FIXME(nnethercote) Sometimes with expressions we get a trailing comma, possibly
                // related to the FIXME in `collect_tokens_for_expr`. Examples are the multi-line
                // `assert_eq!` calls involving arguments annotated with `#[rustfmt::skip]` in
                // `compiler/rustc_index/src/bit_set/tests.rs`.
                if this.token.kind == token::Comma {
                    this.bump();
                }
                expr
            },
        ) {
            return Ok(expr);
        } else if let Some(lit) =
            self.eat_metavar_seq(MetaVarKind::Literal, |this| this.parse_literal_maybe_minus())
        {
            return Ok(lit);
        } else if let Some(block) =
            self.eat_metavar_seq(MetaVarKind::Block, |this| this.parse_block())
        {
            return Ok(self.mk_expr(span, ExprKind::Block(block, None)));
        } else if let Some(path) =
            self.eat_metavar_seq(MetaVarKind::Path, |this| this.parse_path(PathStyle::Type))
        {
            return Ok(self.mk_expr(span, ExprKind::Path(None, path)));
        }

        // Outer attributes are already parsed and will be
        // added to the return value after the fact.

        let restrictions = self.restrictions;
        self.with_res(restrictions - Restrictions::ALLOW_LET, |this| {
            // Note: adding new syntax here? Don't forget to adjust `TokenKind::can_begin_expr()`.
            let lo = this.token.span;
            if let token::Literal(_) = this.token.kind {
                // This match arm is a special-case of the `_` match arm below and
                // could be removed without changing functionality, but it's faster
                // to have it here, especially for programs with large constants.
                this.parse_expr_lit()
            } else if this.check(exp!(OpenParen)) {
                this.parse_expr_tuple_parens(restrictions)
            } else if this.check(exp!(OpenBrace)) {
                this.parse_expr_block(None, lo, BlockCheckMode::Default)
            } else if this.check(exp!(Or)) || this.check(exp!(OrOr)) {
                this.parse_expr_closure().map_err(|mut err| {
                    // If the input is something like `if a { 1 } else { 2 } | if a { 3 } else { 4 }`
                    // then suggest parens around the lhs.
                    if let Some(sp) = this.psess.ambiguous_block_expr_parse.borrow().get(&lo) {
                        err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
                    }
                    err
                })
            } else if this.check(exp!(OpenBracket)) {
                this.parse_expr_array_or_repeat(exp!(CloseBracket))
            } else if this.is_builtin() {
                this.parse_expr_builtin()
            } else if this.check_path() {
                this.parse_expr_path_start()
            } else if this.check_keyword(exp!(Move))
                || this.check_keyword(exp!(Use))
                || this.check_keyword(exp!(Static))
                || this.check_const_closure()
            {
                this.parse_expr_closure()
            } else if this.eat_keyword(exp!(If)) {
                this.parse_expr_if()
            } else if this.check_keyword(exp!(For)) {
                if this.choose_generics_over_qpath(1) {
                    this.parse_expr_closure()
                } else {
                    assert!(this.eat_keyword(exp!(For)));
                    this.parse_expr_for(None, lo)
                }
            } else if this.eat_keyword(exp!(While)) {
                this.parse_expr_while(None, lo)
            } else if let Some(label) = this.eat_label() {
                this.parse_expr_labeled(label, true)
            } else if this.eat_keyword(exp!(Loop)) {
                this.parse_expr_loop(None, lo).map_err(|mut err| {
                    err.span_label(lo, "while parsing this `loop` expression");
                    err
                })
            } else if this.eat_keyword(exp!(Match)) {
                this.parse_expr_match().map_err(|mut err| {
                    err.span_label(lo, "while parsing this `match` expression");
                    err
                })
            } else if this.eat_keyword(exp!(Unsafe)) {
                this.parse_expr_block(None, lo, BlockCheckMode::Unsafe(ast::UserProvided)).map_err(
                    |mut err| {
                        err.span_label(lo, "while parsing this `unsafe` expression");
                        err
                    },
                )
            } else if this.check_inline_const(0) {
                this.parse_const_block(lo, false)
            } else if this.may_recover() && this.is_do_catch_block() {
                this.recover_do_catch()
            } else if this.is_try_block() {
                this.expect_keyword(exp!(Try))?;
                this.parse_try_block(lo)
            } else if this.eat_keyword(exp!(Return)) {
                this.parse_expr_return()
            } else if this.eat_keyword(exp!(Continue)) {
                this.parse_expr_continue(lo)
            } else if this.eat_keyword(exp!(Break)) {
                this.parse_expr_break()
            } else if this.eat_keyword(exp!(Yield)) {
                this.parse_expr_yield()
            } else if this.is_do_yeet() {
                this.parse_expr_yeet()
            } else if this.eat_keyword(exp!(Become)) {
                this.parse_expr_become()
            } else if this.check_keyword(exp!(Let)) {
                this.parse_expr_let(restrictions)
            } else if this.eat_keyword(exp!(Underscore)) {
                Ok(this.mk_expr(this.prev_token.span, ExprKind::Underscore))
            } else if this.token_uninterpolated_span().at_least_rust_2018() {
                // `Span::at_least_rust_2018()` is somewhat expensive; don't get it repeatedly.
                let at_async = this.check_keyword(exp!(Async));
                // check for `gen {}` and `gen move {}`
                // or `async gen {}` and `async gen move {}`
                // FIXME: (async) gen closures aren't yet parsed.
                // FIXME(gen_blocks): Parse `gen async` and suggest swap
                if this.token_uninterpolated_span().at_least_rust_2024()
                    && this.is_gen_block(kw::Gen, at_async as usize)
                {
                    this.parse_gen_block()
                // Check for `async {` and `async move {`,
                } else if this.is_gen_block(kw::Async, 0) {
                    this.parse_gen_block()
                } else if at_async {
                    this.parse_expr_closure()
                } else if this.eat_keyword_noexpect(kw::Await) {
                    this.recover_incorrect_await_syntax(lo)
                } else {
                    this.parse_expr_lit()
                }
            } else {
                this.parse_expr_lit()
            }
        })
    }

    fn parse_expr_lit(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        match self.parse_opt_token_lit() {
            Some((token_lit, _)) => {
                let expr = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Lit(token_lit));
                self.maybe_recover_from_bad_qpath(expr)
            }
            None => self.try_macro_suggestion(),
        }
    }

    fn parse_expr_tuple_parens(&mut self, restrictions: Restrictions) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        self.expect(exp!(OpenParen))?;
        let (es, trailing_comma) = match self.parse_seq_to_end(
            exp!(CloseParen),
            SeqSep::trailing_allowed(exp!(Comma)),
            |p| p.parse_expr_catch_underscore(restrictions.intersection(Restrictions::ALLOW_LET)),
        ) {
            Ok(x) => x,
            Err(err) => {
                return Ok(self.recover_seq_parse_error(
                    exp!(OpenParen),
                    exp!(CloseParen),
                    lo,
                    err,
                ));
            }
        };
        let kind = if es.len() == 1 && matches!(trailing_comma, Trailing::No) {
            // `(e)` is parenthesized `e`.
            ExprKind::Paren(es.into_iter().next().unwrap())
        } else {
            // `(e,)` is a tuple with only one field, `e`.
            ExprKind::Tup(es)
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    fn parse_expr_array_or_repeat(&mut self, close: ExpTokenPair<'_>) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        self.bump(); // `[` or other open delim

        let kind = if self.eat(close) {
            // Empty vector
            ExprKind::Array(ThinVec::new())
        } else {
            // Non-empty vector
            let first_expr = self.parse_expr()?;
            if self.eat(exp!(Semi)) {
                // Repeating array syntax: `[ 0; 512 ]`
                let count = self.parse_expr_anon_const()?;
                self.expect(close)?;
                ExprKind::Repeat(first_expr, count)
            } else if self.eat(exp!(Comma)) {
                // Vector with two or more elements.
                let sep = SeqSep::trailing_allowed(exp!(Comma));
                let (mut exprs, _) = self.parse_seq_to_end(close, sep, |p| p.parse_expr())?;
                exprs.insert(0, first_expr);
                ExprKind::Array(exprs)
            } else {
                // Vector with one element
                self.expect(close)?;
                ExprKind::Array(thin_vec![first_expr])
            }
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    fn parse_expr_path_start(&mut self) -> PResult<'a, P<Expr>> {
        let maybe_eq_tok = self.prev_token;
        let (qself, path) = if self.eat_lt() {
            let lt_span = self.prev_token.span;
            let (qself, path) = self.parse_qpath(PathStyle::Expr).map_err(|mut err| {
                // Suggests using '<=' if there is an error parsing qpath when the previous token
                // is an '=' token. Only emits suggestion if the '<' token and '=' token are
                // directly adjacent (i.e. '=<')
                if maybe_eq_tok == TokenKind::Eq && maybe_eq_tok.span.hi() == lt_span.lo() {
                    let eq_lt = maybe_eq_tok.span.to(lt_span);
                    err.span_suggestion(eq_lt, "did you mean", "<=", Applicability::Unspecified);
                }
                err
            })?;
            (Some(qself), path)
        } else {
            (None, self.parse_path(PathStyle::Expr)?)
        };

        // `!`, as an operator, is prefix, so we know this isn't that.
        let (span, kind) = if self.eat(exp!(Bang)) {
            // MACRO INVOCATION expression
            if qself.is_some() {
                self.dcx().emit_err(errors::MacroInvocationWithQualifiedPath(path.span));
            }
            let lo = path.span;
            let mac = P(MacCall { path, args: self.parse_delim_args()? });
            (lo.to(self.prev_token.span), ExprKind::MacCall(mac))
        } else if self.check(exp!(OpenBrace))
            && let Some(expr) = self.maybe_parse_struct_expr(&qself, &path)
        {
            if qself.is_some() {
                self.psess.gated_spans.gate(sym::more_qualified_paths, path.span);
            }
            return expr;
        } else {
            (path.span, ExprKind::Path(qself, path))
        };

        let expr = self.mk_expr(span, kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `'label: $expr`. The label is already parsed.
    pub(super) fn parse_expr_labeled(
        &mut self,
        label_: Label,
        mut consume_colon: bool,
    ) -> PResult<'a, P<Expr>> {
        let lo = label_.ident.span;
        let label = Some(label_);
        let ate_colon = self.eat(exp!(Colon));
        let tok_sp = self.token.span;
        let expr = if self.eat_keyword(exp!(While)) {
            self.parse_expr_while(label, lo)
        } else if self.eat_keyword(exp!(For)) {
            self.parse_expr_for(label, lo)
        } else if self.eat_keyword(exp!(Loop)) {
            self.parse_expr_loop(label, lo)
        } else if self.check_noexpect(&token::OpenBrace) || self.token.is_metavar_block() {
            self.parse_expr_block(label, lo, BlockCheckMode::Default)
        } else if !ate_colon
            && self.may_recover()
            && (self.token.kind.close_delim().is_some() || self.token.is_punct())
            && could_be_unclosed_char_literal(label_.ident)
        {
            let (lit, _) =
                self.recover_unclosed_char(label_.ident, Parser::mk_token_lit_char, |self_| {
                    self_.dcx().create_err(errors::UnexpectedTokenAfterLabel {
                        span: self_.token.span,
                        remove_label: None,
                        enclose_in_block: None,
                    })
                });
            consume_colon = false;
            Ok(self.mk_expr(lo, ExprKind::Lit(lit)))
        } else if !ate_colon
            && (self.check_noexpect(&TokenKind::Comma) || self.check_noexpect(&TokenKind::Gt))
        {
            // We're probably inside of a `Path<'a>` that needs a turbofish
            let guar = self.dcx().emit_err(errors::UnexpectedTokenAfterLabel {
                span: self.token.span,
                remove_label: None,
                enclose_in_block: None,
            });
            consume_colon = false;
            Ok(self.mk_expr_err(lo, guar))
        } else {
            let mut err = errors::UnexpectedTokenAfterLabel {
                span: self.token.span,
                remove_label: None,
                enclose_in_block: None,
            };

            // Continue as an expression in an effort to recover on `'label: non_block_expr`.
            let expr = self.parse_expr().map(|expr| {
                let span = expr.span;

                let found_labeled_breaks = {
                    struct FindLabeledBreaksVisitor;

                    impl<'ast> Visitor<'ast> for FindLabeledBreaksVisitor {
                        type Result = ControlFlow<()>;
                        fn visit_expr(&mut self, ex: &'ast Expr) -> ControlFlow<()> {
                            if let ExprKind::Break(Some(_label), _) = ex.kind {
                                ControlFlow::Break(())
                            } else {
                                walk_expr(self, ex)
                            }
                        }
                    }

                    FindLabeledBreaksVisitor.visit_expr(&expr).is_break()
                };

                // Suggestion involves adding a labeled block.
                //
                // If there are no breaks that may use this label, suggest removing the label and
                // recover to the unmodified expression.
                if !found_labeled_breaks {
                    err.remove_label = Some(lo.until(span));

                    return expr;
                }

                err.enclose_in_block = Some(errors::UnexpectedTokenAfterLabelSugg {
                    left: span.shrink_to_lo(),
                    right: span.shrink_to_hi(),
                });

                // Replace `'label: non_block_expr` with `'label: {non_block_expr}` in order to suppress future errors about `break 'label`.
                let stmt = self.mk_stmt(span, StmtKind::Expr(expr));
                let blk = self.mk_block(thin_vec![stmt], BlockCheckMode::Default, span);
                self.mk_expr(span, ExprKind::Block(blk, label))
            });

            self.dcx().emit_err(err);
            expr
        }?;

        if !ate_colon && consume_colon {
            self.dcx().emit_err(errors::RequireColonAfterLabeledExpression {
                span: expr.span,
                label: lo,
                label_end: lo.between(tok_sp),
            });
        }

        Ok(expr)
    }

    /// Emit an error when a char is parsed as a lifetime or label because of a missing quote.
    pub(super) fn recover_unclosed_char<L>(
        &self,
        ident: Ident,
        mk_lit_char: impl FnOnce(Symbol, Span) -> L,
        err: impl FnOnce(&Self) -> Diag<'a>,
    ) -> L {
        assert!(could_be_unclosed_char_literal(ident));
        self.dcx()
            .try_steal_modify_and_emit_err(ident.span, StashKey::LifetimeIsChar, |err| {
                err.span_suggestion_verbose(
                    ident.span.shrink_to_hi(),
                    "add `'` to close the char literal",
                    "'",
                    Applicability::MaybeIncorrect,
                );
            })
            .unwrap_or_else(|| {
                err(self)
                    .with_span_suggestion_verbose(
                        ident.span.shrink_to_hi(),
                        "add `'` to close the char literal",
                        "'",
                        Applicability::MaybeIncorrect,
                    )
                    .emit()
            });
        let name = ident.without_first_quote().name;
        mk_lit_char(name, ident.span)
    }

    /// Recover on the syntax `do catch { ... }` suggesting `try { ... }` instead.
    fn recover_do_catch(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        self.bump(); // `do`
        self.bump(); // `catch`

        let span = lo.to(self.prev_token.span);
        self.dcx().emit_err(errors::DoCatchSyntaxRemoved { span });

        self.parse_try_block(lo)
    }

    /// Parse an expression if the token can begin one.
    fn parse_expr_opt(&mut self) -> PResult<'a, Option<P<Expr>>> {
        Ok(if self.token.can_begin_expr() { Some(self.parse_expr()?) } else { None })
    }

    /// Parse `"return" expr?`.
    fn parse_expr_return(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let kind = ExprKind::Ret(self.parse_expr_opt()?);
        let expr = self.mk_expr(lo.to(self.prev_token.span), kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `"do" "yeet" expr?`.
    fn parse_expr_yeet(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        self.bump(); // `do`
        self.bump(); // `yeet`

        let kind = ExprKind::Yeet(self.parse_expr_opt()?);

        let span = lo.to(self.prev_token.span);
        self.psess.gated_spans.gate(sym::yeet_expr, span);
        let expr = self.mk_expr(span, kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `"become" expr`, with `"become"` token already eaten.
    fn parse_expr_become(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let kind = ExprKind::Become(self.parse_expr()?);
        let span = lo.to(self.prev_token.span);
        self.psess.gated_spans.gate(sym::explicit_tail_calls, span);
        let expr = self.mk_expr(span, kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `"break" (('label (:? expr)?) | expr?)` with `"break"` token already eaten.
    /// If the label is followed immediately by a `:` token, the label and `:` are
    /// parsed as part of the expression (i.e. a labeled loop). The language team has
    /// decided in #87026 to require parentheses as a visual aid to avoid confusion if
    /// the break expression of an unlabeled break is a labeled loop (as in
    /// `break 'lbl: loop {}`); a labeled break with an unlabeled loop as its value
    /// expression only gets a warning for compatibility reasons; and a labeled break
    /// with a labeled loop does not even get a warning because there is no ambiguity.
    fn parse_expr_break(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let mut label = self.eat_label();
        let kind = if self.token == token::Colon
            && let Some(label) = label.take()
        {
            // The value expression can be a labeled loop, see issue #86948, e.g.:
            // `loop { break 'label: loop { break 'label 42; }; }`
            let lexpr = self.parse_expr_labeled(label, true)?;
            self.dcx().emit_err(errors::LabeledLoopInBreak {
                span: lexpr.span,
                sub: errors::WrapInParentheses::Expression {
                    left: lexpr.span.shrink_to_lo(),
                    right: lexpr.span.shrink_to_hi(),
                },
            });
            Some(lexpr)
        } else if self.token != token::OpenBrace
            || !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
        {
            let mut expr = self.parse_expr_opt()?;
            if let Some(expr) = &mut expr {
                if label.is_some()
                    && match &expr.kind {
                        ExprKind::While(_, _, None)
                        | ExprKind::ForLoop { label: None, .. }
                        | ExprKind::Loop(_, None, _) => true,
                        ExprKind::Block(block, None) => {
                            matches!(block.rules, BlockCheckMode::Default)
                        }
                        _ => false,
                    }
                {
                    self.psess.buffer_lint(
                        BREAK_WITH_LABEL_AND_LOOP,
                        lo.to(expr.span),
                        ast::CRATE_NODE_ID,
                        BuiltinLintDiag::BreakWithLabelAndLoop(expr.span),
                    );
                }

                // Recover `break label aaaaa`
                if self.may_recover()
                    && let ExprKind::Path(None, p) = &expr.kind
                    && let [segment] = &*p.segments
                    && let &ast::PathSegment { ident, args: None, .. } = segment
                    && let Some(next) = self.parse_expr_opt()?
                {
                    label = Some(self.recover_ident_into_label(ident));
                    *expr = next;
                }
            }

            expr
        } else {
            None
        };
        let expr = self.mk_expr(lo.to(self.prev_token.span), ExprKind::Break(label, kind));
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `"continue" label?`.
    fn parse_expr_continue(&mut self, lo: Span) -> PResult<'a, P<Expr>> {
        let mut label = self.eat_label();

        // Recover `continue label` -> `continue 'label`
        if self.may_recover()
            && label.is_none()
            && let Some((ident, _)) = self.token.ident()
        {
            self.bump();
            label = Some(self.recover_ident_into_label(ident));
        }

        let kind = ExprKind::Continue(label);
        Ok(self.mk_expr(lo.to(self.prev_token.span), kind))
    }

    /// Parse `"yield" expr?`.
    fn parse_expr_yield(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        let kind = ExprKind::Yield(YieldKind::Prefix(self.parse_expr_opt()?));
        let span = lo.to(self.prev_token.span);
        self.psess.gated_spans.gate(sym::yield_expr, span);
        let expr = self.mk_expr(span, kind);
        self.maybe_recover_from_bad_qpath(expr)
    }

    /// Parse `builtin # ident(args,*)`.
    fn parse_expr_builtin(&mut self) -> PResult<'a, P<Expr>> {
        self.parse_builtin(|this, lo, ident| {
            Ok(match ident.name {
                sym::offset_of => Some(this.parse_expr_offset_of(lo)?),
                sym::type_ascribe => Some(this.parse_expr_type_ascribe(lo)?),
                sym::wrap_binder => {
                    Some(this.parse_expr_unsafe_binder_cast(lo, UnsafeBinderCastKind::Wrap)?)
                }
                sym::unwrap_binder => {
                    Some(this.parse_expr_unsafe_binder_cast(lo, UnsafeBinderCastKind::Unwrap)?)
                }
                _ => None,
            })
        })
    }

    pub(crate) fn parse_builtin<T>(
        &mut self,
        parse: impl FnOnce(&mut Parser<'a>, Span, Ident) -> PResult<'a, Option<T>>,
    ) -> PResult<'a, T> {
        let lo = self.token.span;

        self.bump(); // `builtin`
        self.bump(); // `#`

        let Some((ident, IdentIsRaw::No)) = self.token.ident() else {
            let err = self.dcx().create_err(errors::ExpectedBuiltinIdent { span: self.token.span });
            return Err(err);
        };
        self.psess.gated_spans.gate(sym::builtin_syntax, ident.span);
        self.bump();

        self.expect(exp!(OpenParen))?;
        let ret = if let Some(res) = parse(self, lo, ident)? {
            Ok(res)
        } else {
            let err = self.dcx().create_err(errors::UnknownBuiltinConstruct {
                span: lo.to(ident.span),
                name: ident,
            });
            return Err(err);
        };
        self.expect(exp!(CloseParen))?;

        ret
    }

    /// Built-in macro for `offset_of!` expressions.
    pub(crate) fn parse_expr_offset_of(&mut self, lo: Span) -> PResult<'a, P<Expr>> {
        let container = self.parse_ty()?;
        self.expect(exp!(Comma))?;

        let fields = self.parse_floating_field_access()?;
        let trailing_comma = self.eat_noexpect(&TokenKind::Comma);

        if let Err(mut e) = self.expect_one_of(&[], &[exp!(CloseParen)]) {
            if trailing_comma {
                e.note("unexpected third argument to offset_of");
            } else {
                e.note("offset_of expects dot-separated field and variant names");
            }
            e.emit();
        }

        // Eat tokens until the macro call ends.
        if self.may_recover() {
            while !self.token.kind.is_close_delim_or_eof() {
                self.bump();
            }
        }

        let span = lo.to(self.token.span);
        Ok(self.mk_expr(span, ExprKind::OffsetOf(container, fields)))
    }

    /// Built-in macro for type ascription expressions.
    pub(crate) fn parse_expr_type_ascribe(&mut self, lo: Span) -> PResult<'a, P<Expr>> {
        let expr = self.parse_expr()?;
        self.expect(exp!(Comma))?;
        let ty = self.parse_ty()?;
        let span = lo.to(self.token.span);
        Ok(self.mk_expr(span, ExprKind::Type(expr, ty)))
    }

    pub(crate) fn parse_expr_unsafe_binder_cast(
        &mut self,
        lo: Span,
        kind: UnsafeBinderCastKind,
    ) -> PResult<'a, P<Expr>> {
        let expr = self.parse_expr()?;
        let ty = if self.eat(exp!(Comma)) { Some(self.parse_ty()?) } else { None };
        let span = lo.to(self.token.span);
        Ok(self.mk_expr(span, ExprKind::UnsafeBinderCast(kind, expr, ty)))
    }

    /// Returns a string literal if the next token is a string literal.
    /// In case of error returns `Some(lit)` if the next token is a literal with a wrong kind,
    /// and returns `None` if the next token is not literal at all.
    pub fn parse_str_lit(&mut self) -> Result<ast::StrLit, Option<MetaItemLit>> {
        match self.parse_opt_meta_item_lit() {
            Some(lit) => match lit.kind {
                ast::LitKind::Str(symbol_unescaped, style) => Ok(ast::StrLit {
                    style,
                    symbol: lit.symbol,
                    suffix: lit.suffix,
                    span: lit.span,
                    symbol_unescaped,
                }),
                _ => Err(Some(lit)),
            },
            None => Err(None),
        }
    }

    pub(crate) fn mk_token_lit_char(name: Symbol, span: Span) -> (token::Lit, Span) {
        (token::Lit { symbol: name, suffix: None, kind: token::Char }, span)
    }

    fn mk_meta_item_lit_char(name: Symbol, span: Span) -> MetaItemLit {
        ast::MetaItemLit {
            symbol: name,
            suffix: None,
            kind: ast::LitKind::Char(name.as_str().chars().next().unwrap_or('_')),
            span,
        }
    }

    fn handle_missing_lit<L>(
        &mut self,
        mk_lit_char: impl FnOnce(Symbol, Span) -> L,
    ) -> PResult<'a, L> {
        let token = self.token;
        let err = |self_: &Self| {
            let msg = format!("unexpected token: {}", super::token_descr(&token));
            self_.dcx().struct_span_err(token.span, msg)
        };
        // On an error path, eagerly consider a lifetime to be an unclosed character lit, if that
        // makes sense.
        if let Some((ident, IdentIsRaw::No)) = self.token.lifetime()
            && could_be_unclosed_char_literal(ident)
        {
            let lt = self.expect_lifetime();
            Ok(self.recover_unclosed_char(lt.ident, mk_lit_char, err))
        } else {
            Err(err(self))
        }
    }

    pub(super) fn parse_token_lit(&mut self) -> PResult<'a, (token::Lit, Span)> {
        self.parse_opt_token_lit()
            .ok_or(())
            .or_else(|()| self.handle_missing_lit(Parser::mk_token_lit_char))
    }

    pub(super) fn parse_meta_item_lit(&mut self) -> PResult<'a, MetaItemLit> {
        self.parse_opt_meta_item_lit()
            .ok_or(())
            .or_else(|()| self.handle_missing_lit(Parser::mk_meta_item_lit_char))
    }

    fn recover_after_dot(&mut self) {
        if self.token == token::Dot {
            // Attempt to recover `.4` as `0.4`. We don't currently have any syntax where
            // dot would follow an optional literal, so we do this unconditionally.
            let recovered = self.look_ahead(1, |next_token| {
                // If it's an integer that looks like a float, then recover as such.
                //
                // We will never encounter the exponent part of a floating
                // point literal here, since there's no use of the exponent
                // syntax that also constitutes a valid integer, so we need
                // not check for that.
                if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) =
                    next_token.kind
                    && suffix.is_none_or(|s| s == sym::f32 || s == sym::f64)
                    && symbol.as_str().chars().all(|c| c.is_numeric() || c == '_')
                    && self.token.span.hi() == next_token.span.lo()
                {
                    let s = String::from("0.") + symbol.as_str();
                    let kind = TokenKind::lit(token::Float, Symbol::intern(&s), suffix);
                    Some(Token::new(kind, self.token.span.to(next_token.span)))
                } else {
                    None
                }
            });
            if let Some(recovered) = recovered {
                self.dcx().emit_err(errors::FloatLiteralRequiresIntegerPart {
                    span: recovered.span,
                    suggestion: recovered.span.shrink_to_lo(),
                });
                self.bump();
                self.token = recovered;
            }
        }
    }

    /// Keep this in sync with `Token::can_begin_literal_maybe_minus` and
    /// `Lit::from_token` (excluding unary negation).
    fn eat_token_lit(&mut self) -> Option<token::Lit> {
        let check_expr = |expr: P<Expr>| {
            if let ast::ExprKind::Lit(token_lit) = expr.kind {
                Some(token_lit)
            } else if let ast::ExprKind::Unary(UnOp::Neg, inner) = &expr.kind
                && let ast::Expr { kind: ast::ExprKind::Lit(_), .. } = **inner
            {
                None
            } else {
                panic!("unexpected reparsed expr/literal: {:?}", expr.kind);
            }
        };
        match self.token.uninterpolate().kind {
            token::Ident(name, IdentIsRaw::No) if name.is_bool_lit() => {
                self.bump();
                Some(token::Lit::new(token::Bool, name, None))
            }
            token::Literal(token_lit) => {
                self.bump();
                Some(token_lit)
            }
            token::OpenInvisible(InvisibleOrigin::MetaVar(MetaVarKind::Literal)) => {
                let lit = self
                    .eat_metavar_seq(MetaVarKind::Literal, |this| this.parse_literal_maybe_minus())
                    .expect("metavar seq literal");
                check_expr(lit)
            }
            token::OpenInvisible(InvisibleOrigin::MetaVar(
                mv_kind @ MetaVarKind::Expr { can_begin_literal_maybe_minus: true, .. },
            )) => {
                let expr = self
                    .eat_metavar_seq(mv_kind, |this| this.parse_expr())
                    .expect("metavar seq expr");
                check_expr(expr)
            }
            _ => None,
        }
    }

    /// Matches `lit = true | false | token_lit`.
    /// Returns `None` if the next token is not a literal.
    fn parse_opt_token_lit(&mut self) -> Option<(token::Lit, Span)> {
        self.recover_after_dot();
        let span = self.token.span;
        self.eat_token_lit().map(|token_lit| (token_lit, span))
    }

    /// Matches `lit = true | false | token_lit`.
    /// Returns `None` if the next token is not a literal.
    fn parse_opt_meta_item_lit(&mut self) -> Option<MetaItemLit> {
        self.recover_after_dot();
        let span = self.token.span;
        let uninterpolated_span = self.token_uninterpolated_span();
        self.eat_token_lit().map(|token_lit| {
            match MetaItemLit::from_token_lit(token_lit, span) {
                Ok(lit) => lit,
                Err(err) => {
                    let guar = report_lit_error(&self.psess, err, token_lit, uninterpolated_span);
                    // Pack possible quotes and prefixes from the original literal into
                    // the error literal's symbol so they can be pretty-printed faithfully.
                    let suffixless_lit = token::Lit::new(token_lit.kind, token_lit.symbol, None);
                    let symbol = Symbol::intern(&suffixless_lit.to_string());
                    let token_lit = token::Lit::new(token::Err(guar), symbol, token_lit.suffix);
                    MetaItemLit::from_token_lit(token_lit, uninterpolated_span).unwrap()
                }
            }
        })
    }

    pub(super) fn expect_no_tuple_index_suffix(&self, span: Span, suffix: Symbol) {
        if [sym::i32, sym::u32, sym::isize, sym::usize].contains(&suffix) {
            // #59553: warn instead of reject out of hand to allow the fix to percolate
            // through the ecosystem when people fix their macros
            self.dcx().emit_warn(errors::InvalidLiteralSuffixOnTupleIndex {
                span,
                suffix,
                exception: true,
            });
        } else {
            self.dcx().emit_err(errors::InvalidLiteralSuffixOnTupleIndex {
                span,
                suffix,
                exception: false,
            });
        }
    }

    /// Matches `'-' lit | lit` (cf. `ast_validation::AstValidator::check_expr_within_pat`).
    /// Keep this in sync with `Token::can_begin_literal_maybe_minus`.
    pub fn parse_literal_maybe_minus(&mut self) -> PResult<'a, P<Expr>> {
        if let Some(expr) = self.eat_metavar_seq_with_matcher(
            |mv_kind| matches!(mv_kind, MetaVarKind::Expr { .. }),
            |this| {
                // FIXME(nnethercote) The `expr` case should only match if
                // `e` is an `ExprKind::Lit` or an `ExprKind::Unary` containing
                // an `UnOp::Neg` and an `ExprKind::Lit`, like how
                // `can_begin_literal_maybe_minus` works. But this method has
                // been over-accepting for a long time, and to make that change
                // here requires also changing some `parse_literal_maybe_minus`
                // call sites to accept additional expression kinds. E.g.
                // `ExprKind::Path` must be accepted when parsing range
                // patterns. That requires some care. So for now, we continue
                // being less strict here than we should be.
                this.parse_expr()
            },
        ) {
            return Ok(expr);
        } else if let Some(lit) =
            self.eat_metavar_seq(MetaVarKind::Literal, |this| this.parse_literal_maybe_minus())
        {
            return Ok(lit);
        }

        let lo = self.token.span;
        let minus_present = self.eat(exp!(Minus));
        let (token_lit, span) = self.parse_token_lit()?;
        let expr = self.mk_expr(span, ExprKind::Lit(token_lit));

        if minus_present {
            Ok(self.mk_expr(lo.to(self.prev_token.span), self.mk_unary(UnOp::Neg, expr)))
        } else {
            Ok(expr)
        }
    }

    fn is_array_like_block(&mut self) -> bool {
        self.token.kind == TokenKind::OpenBrace
            && self
                .look_ahead(1, |t| matches!(t.kind, TokenKind::Ident(..) | TokenKind::Literal(_)))
            && self.look_ahead(2, |t| t == &token::Comma)
            && self.look_ahead(3, |t| t.can_begin_expr())
    }

    /// Emits a suggestion if it looks like the user meant an array but
    /// accidentally used braces, causing the code to be interpreted as a block
    /// expression.
    fn maybe_suggest_brackets_instead_of_braces(&mut self, lo: Span) -> Option<P<Expr>> {
        let mut snapshot = self.create_snapshot_for_diagnostic();
        match snapshot.parse_expr_array_or_repeat(exp!(CloseBrace)) {
            Ok(arr) => {
                let guar = self.dcx().emit_err(errors::ArrayBracketsInsteadOfBraces {
                    span: arr.span,
                    sub: errors::ArrayBracketsInsteadOfBracesSugg {
                        left: lo,
                        right: snapshot.prev_token.span,
                    },
                });

                self.restore_snapshot(snapshot);
                Some(self.mk_expr_err(arr.span, guar))
            }
            Err(e) => {
                e.cancel();
                None
            }
        }
    }

    fn suggest_missing_semicolon_before_array(
        &self,
        prev_span: Span,
        open_delim_span: Span,
    ) -> PResult<'a, ()> {
        if !self.may_recover() {
            return Ok(());
        }

        if self.token == token::Comma {
            if !self.psess.source_map().is_multiline(prev_span.until(self.token.span)) {
                return Ok(());
            }
            let mut snapshot = self.create_snapshot_for_diagnostic();
            snapshot.bump();
            match snapshot.parse_seq_to_before_end(
                exp!(CloseBracket),
                SeqSep::trailing_allowed(exp!(Comma)),
                |p| p.parse_expr(),
            ) {
                Ok(_)
                    // When the close delim is `)`, `token.kind` is expected to be `token::CloseParen`,
                    // but the actual `token.kind` is `token::CloseBracket`.
                    // This is because the `token.kind` of the close delim is treated as the same as
                    // that of the open delim in `TokenTreesReader::parse_token_tree`, even if the delimiters of them are different.
                    // Therefore, `token.kind` should not be compared here.
                    if snapshot
                        .span_to_snippet(snapshot.token.span)
                        .is_ok_and(|snippet| snippet == "]") =>
                {
                    return Err(self.dcx().create_err(errors::MissingSemicolonBeforeArray {
                        open_delim: open_delim_span,
                        semicolon: prev_span.shrink_to_hi(),
                    }));
                }
                Ok(_) => (),
                Err(err) => err.cancel(),
            }
        }
        Ok(())
    }

    /// Parses a block or unsafe block.
    pub(super) fn parse_expr_block(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        blk_mode: BlockCheckMode,
    ) -> PResult<'a, P<Expr>> {
        if self.may_recover() && self.is_array_like_block() {
            if let Some(arr) = self.maybe_suggest_brackets_instead_of_braces(lo) {
                return Ok(arr);
            }
        }

        if self.token.is_metavar_block() {
            self.dcx().emit_err(errors::InvalidBlockMacroSegment {
                span: self.token.span,
                context: lo.to(self.token.span),
                wrap: errors::WrapInExplicitBlock {
                    lo: self.token.span.shrink_to_lo(),
                    hi: self.token.span.shrink_to_hi(),
                },
            });
        }

        let (attrs, blk) = self.parse_block_common(lo, blk_mode, None)?;
        Ok(self.mk_expr_with_attrs(blk.span, ExprKind::Block(blk, opt_label), attrs))
    }

    /// Parse a block which takes no attributes and has no label
    fn parse_simple_block(&mut self) -> PResult<'a, P<Expr>> {
        let blk = self.parse_block()?;
        Ok(self.mk_expr(blk.span, ExprKind::Block(blk, None)))
    }

    /// Parses a closure expression (e.g., `move |args| expr`).
    fn parse_expr_closure(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;

        let before = self.prev_token;
        let binder = if self.check_keyword(exp!(For)) {
            let lo = self.token.span;
            let (lifetime_defs, _) = self.parse_late_bound_lifetime_defs()?;
            let span = lo.to(self.prev_token.span);

            self.psess.gated_spans.gate(sym::closure_lifetime_binder, span);

            ClosureBinder::For { span, generic_params: lifetime_defs }
        } else {
            ClosureBinder::NotPresent
        };

        let constness = self.parse_closure_constness();

        let movability =
            if self.eat_keyword(exp!(Static)) { Movability::Static } else { Movability::Movable };

        let coroutine_kind = if self.token_uninterpolated_span().at_least_rust_2018() {
            self.parse_coroutine_kind(Case::Sensitive)
        } else {
            None
        };

        if let ClosureBinder::NotPresent = binder
            && coroutine_kind.is_some()
        {
            // coroutine closures and generators can have the same qualifiers, so we might end up
            // in here if there is a missing `|` but also no `{`. Adjust the expectations in that case.
            self.expected_token_types.insert(TokenType::OpenBrace);
        }

        let capture_clause = self.parse_capture_clause()?;
        let (fn_decl, fn_arg_span) = self.parse_fn_block_decl()?;
        let decl_hi = self.prev_token.span;
        let mut body = match &fn_decl.output {
            // No return type.
            FnRetTy::Default(_) => {
                let restrictions =
                    self.restrictions - Restrictions::STMT_EXPR - Restrictions::ALLOW_LET;
                let prev = self.prev_token;
                let token = self.token;
                let attrs = self.parse_outer_attributes()?;
                match self.parse_expr_res(restrictions, attrs) {
                    Ok((expr, _)) => expr,
                    Err(err) => self.recover_closure_body(err, before, prev, token, lo, decl_hi)?,
                }
            }
            // Explicit return type (`->`) needs block `-> T { }`.
            FnRetTy::Ty(ty) => self.parse_closure_block_body(ty.span)?,
        };

        match coroutine_kind {
            Some(CoroutineKind::Async { .. }) => {}
            Some(CoroutineKind::Gen { span, .. }) | Some(CoroutineKind::AsyncGen { span, .. }) => {
                // Feature-gate `gen ||` and `async gen ||` closures.
                // FIXME(gen_blocks): This perhaps should be a different gate.
                self.psess.gated_spans.gate(sym::gen_blocks, span);
            }
            None => {}
        }

        if self.token == TokenKind::Semi
            && let Some(last) = self.token_cursor.stack.last()
            && let Some(TokenTree::Delimited(_, _, Delimiter::Parenthesis, _)) = last.curr()
            && self.may_recover()
        {
            // It is likely that the closure body is a block but where the
            // braces have been removed. We will recover and eat the next
            // statements later in the parsing process.
            body = self.mk_expr_err(
                body.span,
                self.dcx().span_delayed_bug(body.span, "recovered a closure body as a block"),
            );
        }

        let body_span = body.span;

        let closure = self.mk_expr(
            lo.to(body.span),
            ExprKind::Closure(Box::new(ast::Closure {
                binder,
                capture_clause,
                constness,
                coroutine_kind,
                movability,
                fn_decl,
                body,
                fn_decl_span: lo.to(decl_hi),
                fn_arg_span,
            })),
        );

        // Disable recovery for closure body
        let spans =
            ClosureSpans { whole_closure: closure.span, closing_pipe: decl_hi, body: body_span };
        self.current_closure = Some(spans);

        Ok(closure)
    }

    /// If an explicit return type is given, require a block to appear (RFC 968).
    fn parse_closure_block_body(&mut self, ret_span: Span) -> PResult<'a, P<Expr>> {
        if self.may_recover()
            && self.token.can_begin_expr()
            && self.token.kind != TokenKind::OpenBrace
            && !self.token.is_metavar_block()
        {
            let snapshot = self.create_snapshot_for_diagnostic();
            let restrictions =
                self.restrictions - Restrictions::STMT_EXPR - Restrictions::ALLOW_LET;
            let tok = self.token.clone();
            match self.parse_expr_res(restrictions, AttrWrapper::empty()) {
                Ok((expr, _)) => {
                    let descr = super::token_descr(&tok);
                    let mut diag = self
                        .dcx()
                        .struct_span_err(tok.span, format!("expected `{{`, found {descr}"));
                    diag.span_label(
                        ret_span,
                        "explicit return type requires closure body to be enclosed in braces",
                    );
                    diag.multipart_suggestion_verbose(
                        "wrap the expression in curly braces",
                        vec![
                            (expr.span.shrink_to_lo(), "{ ".to_string()),
                            (expr.span.shrink_to_hi(), " }".to_string()),
                        ],
                        Applicability::MachineApplicable,
                    );
                    diag.emit();
                    return Ok(expr);
                }
                Err(diag) => {
                    diag.cancel();
                    self.restore_snapshot(snapshot);
                }
            }
        }

        let body_lo = self.token.span;
        self.parse_expr_block(None, body_lo, BlockCheckMode::Default)
    }

    /// Parses an optional `move` or `use` prefix to a closure-like construct.
    fn parse_capture_clause(&mut self) -> PResult<'a, CaptureBy> {
        if self.eat_keyword(exp!(Move)) {
            let move_kw_span = self.prev_token.span;
            // Check for `move async` and recover
            if self.check_keyword(exp!(Async)) {
                let move_async_span = self.token.span.with_lo(self.prev_token.span.data().lo);
                Err(self
                    .dcx()
                    .create_err(errors::AsyncMoveOrderIncorrect { span: move_async_span }))
            } else {
                Ok(CaptureBy::Value { move_kw: move_kw_span })
            }
        } else if self.eat_keyword(exp!(Use)) {
            let use_kw_span = self.prev_token.span;
            self.psess.gated_spans.gate(sym::ergonomic_clones, use_kw_span);
            // Check for `use async` and recover
            if self.check_keyword(exp!(Async)) {
                let use_async_span = self.token.span.with_lo(self.prev_token.span.data().lo);
                Err(self.dcx().create_err(errors::AsyncUseOrderIncorrect { span: use_async_span }))
            } else {
                Ok(CaptureBy::Use { use_kw: use_kw_span })
            }
        } else {
            Ok(CaptureBy::Ref)
        }
    }

    /// Parses the `|arg, arg|` header of a closure.
    fn parse_fn_block_decl(&mut self) -> PResult<'a, (P<FnDecl>, Span)> {
        let arg_start = self.token.span.lo();

        let inputs = if self.eat(exp!(OrOr)) {
            ThinVec::new()
        } else {
            self.expect(exp!(Or))?;
            let args = self
                .parse_seq_to_before_tokens(
                    &[exp!(Or)],
                    &[&token::OrOr],
                    SeqSep::trailing_allowed(exp!(Comma)),
                    |p| p.parse_fn_block_param(),
                )?
                .0;
            self.expect_or()?;
            args
        };
        let arg_span = self.prev_token.span.with_lo(arg_start);
        let output =
            self.parse_ret_ty(AllowPlus::Yes, RecoverQPath::Yes, RecoverReturnSign::Yes)?;

        Ok((P(FnDecl { inputs, output }), arg_span))
    }

    /// Parses a parameter in a closure header (e.g., `|arg, arg|`).
    fn parse_fn_block_param(&mut self) -> PResult<'a, Param> {
        let lo = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let pat = this.parse_pat_no_top_alt(Some(Expected::ParameterName), None)?;
            let ty = if this.eat(exp!(Colon)) {
                this.parse_ty()?
            } else {
                this.mk_ty(pat.span, TyKind::Infer)
            };

            Ok((
                Param {
                    attrs,
                    ty,
                    pat,
                    span: lo.to(this.prev_token.span),
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                Trailing::from(this.token == token::Comma),
                UsePreAttrPos::No,
            ))
        })
    }

    /// Parses an `if` expression (`if` token already eaten).
    fn parse_expr_if(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.prev_token.span;
        // Scoping code checks the top level edition of the `if`; let's match it here.
        // The `CondChecker` also checks the edition of the `let` itself, just to make sure.
        let let_chains_policy = LetChainsPolicy::EditionDependent { current_edition: lo.edition() };
        let cond = self.parse_expr_cond(let_chains_policy)?;
        self.parse_if_after_cond(lo, cond)
    }

    fn parse_if_after_cond(&mut self, lo: Span, mut cond: P<Expr>) -> PResult<'a, P<Expr>> {
        let cond_span = cond.span;
        // Tries to interpret `cond` as either a missing expression if it's a block,
        // or as an unfinished expression if it's a binop and the RHS is a block.
        // We could probably add more recoveries here too...
        let mut recover_block_from_condition = |this: &mut Self| {
            let block = match &mut cond.kind {
                ExprKind::Binary(Spanned { span: binop_span, .. }, _, right)
                    if let ExprKind::Block(_, None) = right.kind =>
                {
                    let guar = this.dcx().emit_err(errors::IfExpressionMissingThenBlock {
                        if_span: lo,
                        missing_then_block_sub:
                            errors::IfExpressionMissingThenBlockSub::UnfinishedCondition(
                                cond_span.shrink_to_lo().to(*binop_span),
                            ),
                        let_else_sub: None,
                    });
                    std::mem::replace(right, this.mk_expr_err(binop_span.shrink_to_hi(), guar))
                }
                ExprKind::Block(_, None) => {
                    let guar = this.dcx().emit_err(errors::IfExpressionMissingCondition {
                        if_span: lo.with_neighbor(cond.span).shrink_to_hi(),
                        block_span: self.psess.source_map().start_point(cond_span),
                    });
                    std::mem::replace(&mut cond, this.mk_expr_err(cond_span.shrink_to_hi(), guar))
                }
                _ => {
                    return None;
                }
            };
            if let ExprKind::Block(block, _) = &block.kind {
                Some(block.clone())
            } else {
                unreachable!()
            }
        };
        // Parse then block
        let thn = if self.token.is_keyword(kw::Else) {
            if let Some(block) = recover_block_from_condition(self) {
                block
            } else {
                let let_else_sub = matches!(cond.kind, ExprKind::Let(..))
                    .then(|| errors::IfExpressionLetSomeSub { if_span: lo.until(cond_span) });

                let guar = self.dcx().emit_err(errors::IfExpressionMissingThenBlock {
                    if_span: lo,
                    missing_then_block_sub: errors::IfExpressionMissingThenBlockSub::AddThenBlock(
                        cond_span.shrink_to_hi(),
                    ),
                    let_else_sub,
                });
                self.mk_block_err(cond_span.shrink_to_hi(), guar)
            }
        } else {
            let attrs = self.parse_outer_attributes()?; // For recovery.
            let maybe_fatarrow = self.token;
            let block = if self.check(exp!(OpenBrace)) {
                self.parse_block()?
            } else if let Some(block) = recover_block_from_condition(self) {
                block
            } else {
                self.error_on_extra_if(&cond)?;
                // Parse block, which will always fail, but we can add a nice note to the error
                self.parse_block().map_err(|mut err| {
                        if self.prev_token == token::Semi
                            && self.token == token::AndAnd
                            && let maybe_let = self.look_ahead(1, |t| t.clone())
                            && maybe_let.is_keyword(kw::Let)
                        {
                            err.span_suggestion(
                                self.prev_token.span,
                                "consider removing this semicolon to parse the `let` as part of the same chain",
                                "",
                                Applicability::MachineApplicable,
                            ).span_note(
                                self.token.span.to(maybe_let.span),
                                "you likely meant to continue parsing the let-chain starting here",
                            );
                        } else {
                            // Look for usages of '=>' where '>=' might be intended
                            if maybe_fatarrow == token::FatArrow {
                                err.span_suggestion(
                                    maybe_fatarrow.span,
                                    "you might have meant to write a \"greater than or equal to\" comparison",
                                    ">=",
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            err.span_note(
                                cond_span,
                                "the `if` expression is missing a block after this condition",
                            );
                        }
                        err
                    })?
            };
            self.error_on_if_block_attrs(lo, false, block.span, attrs);
            block
        };
        let els = if self.eat_keyword(exp!(Else)) { Some(self.parse_expr_else()?) } else { None };
        Ok(self.mk_expr(lo.to(self.prev_token.span), ExprKind::If(cond, thn, els)))
    }

    /// Parses the condition of a `if` or `while` expression.
    ///
    /// The specified `edition` in `let_chains_policy` should be that of the whole `if` construct,
    /// i.e. the same span we use to later decide whether the drop behaviour should be that of
    /// edition `..=2021` or that of `2024..`.
    // Public because it is used in rustfmt forks such as https://github.com/tucant/rustfmt/blob/30c83df9e1db10007bdd16dafce8a86b404329b2/src/parse/macros/html.rs#L57 for custom if expressions.
    pub fn parse_expr_cond(&mut self, let_chains_policy: LetChainsPolicy) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_outer_attributes()?;
        let (mut cond, _) =
            self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL | Restrictions::ALLOW_LET, attrs)?;

        CondChecker::new(self, let_chains_policy).visit_expr(&mut cond);

        Ok(cond)
    }

    /// Parses a `let $pat = $expr` pseudo-expression.
    fn parse_expr_let(&mut self, restrictions: Restrictions) -> PResult<'a, P<Expr>> {
        let recovered = if !restrictions.contains(Restrictions::ALLOW_LET) {
            let err = errors::ExpectedExpressionFoundLet {
                span: self.token.span,
                reason: ForbiddenLetReason::OtherForbidden,
                missing_let: None,
                comparison: None,
            };
            if self.prev_token == token::Or {
                // This was part of a closure, the that part of the parser recover.
                return Err(self.dcx().create_err(err));
            } else {
                Recovered::Yes(self.dcx().emit_err(err))
            }
        } else {
            Recovered::No
        };
        self.bump(); // Eat `let` token
        let lo = self.prev_token.span;
        let pat = self.parse_pat_no_top_guard(
            None,
            RecoverComma::Yes,
            RecoverColon::Yes,
            CommaRecoveryMode::LikelyTuple,
        )?;
        if self.token == token::EqEq {
            self.dcx().emit_err(errors::ExpectedEqForLetExpr {
                span: self.token.span,
                sugg_span: self.token.span,
            });
            self.bump();
        } else {
            self.expect(exp!(Eq))?;
        }
        let attrs = self.parse_outer_attributes()?;
        let (expr, _) =
            self.parse_expr_assoc_with(Bound::Excluded(prec_let_scrutinee_needs_par()), attrs)?;
        let span = lo.to(expr.span);
        Ok(self.mk_expr(span, ExprKind::Let(pat, expr, span, recovered)))
    }

    /// Parses an `else { ... }` expression (`else` token already eaten).
    fn parse_expr_else(&mut self) -> PResult<'a, P<Expr>> {
        let else_span = self.prev_token.span; // `else`
        let attrs = self.parse_outer_attributes()?; // For recovery.
        let expr = if self.eat_keyword(exp!(If)) {
            ensure_sufficient_stack(|| self.parse_expr_if())?
        } else if self.check(exp!(OpenBrace)) {
            self.parse_simple_block()?
        } else {
            let snapshot = self.create_snapshot_for_diagnostic();
            let first_tok = super::token_descr(&self.token);
            let first_tok_span = self.token.span;
            match self.parse_expr() {
                Ok(cond)
                // Try to guess the difference between a "condition-like" vs
                // "statement-like" expression.
                //
                // We are seeing the following code, in which $cond is neither
                // ExprKind::Block nor ExprKind::If (the 2 cases wherein this
                // would be valid syntax).
                //
                //     if ... {
                //     } else $cond
                //
                // If $cond is "condition-like" such as ExprKind::Binary, we
                // want to suggest inserting `if`.
                //
                //     if ... {
                //     } else if a == b {
                //            ^^
                //     }
                //
                // We account for macro calls that were meant as conditions as well.
                //
                //     if ... {
                //     } else if macro! { foo bar } {
                //            ^^
                //     }
                //
                // If $cond is "statement-like" such as ExprKind::While then we
                // want to suggest wrapping in braces.
                //
                //     if ... {
                //     } else {
                //            ^
                //         while true {}
                //     }
                //     ^
                    if self.check(exp!(OpenBrace))
                        && (classify::expr_requires_semi_to_be_stmt(&cond)
                            || matches!(cond.kind, ExprKind::MacCall(..)))
                    =>
                {
                    self.dcx().emit_err(errors::ExpectedElseBlock {
                        first_tok_span,
                        first_tok,
                        else_span,
                        condition_start: cond.span.shrink_to_lo(),
                    });
                    self.parse_if_after_cond(cond.span.shrink_to_lo(), cond)?
                }
                Err(e) => {
                    e.cancel();
                    self.restore_snapshot(snapshot);
                    self.parse_simple_block()?
                },
                Ok(_) => {
                    self.restore_snapshot(snapshot);
                    self.parse_simple_block()?
                },
            }
        };
        self.error_on_if_block_attrs(else_span, true, expr.span, attrs);
        Ok(expr)
    }

    fn error_on_if_block_attrs(
        &self,
        ctx_span: Span,
        is_ctx_else: bool,
        branch_span: Span,
        attrs: AttrWrapper,
    ) {
        if !attrs.is_empty()
            && let [x0 @ xn] | [x0, .., xn] = &*attrs.take_for_recovery(self.psess)
        {
            let attributes = x0.span.until(branch_span);
            let last = xn.span;
            let ctx = if is_ctx_else { "else" } else { "if" };
            self.dcx().emit_err(errors::OuterAttributeNotAllowedOnIfElse {
                last,
                branch_span,
                ctx_span,
                ctx: ctx.to_string(),
                attributes,
            });
        }
    }

    fn error_on_extra_if(&mut self, cond: &P<Expr>) -> PResult<'a, ()> {
        if let ExprKind::Binary(Spanned { span: binop_span, node: binop }, _, right) = &cond.kind
            && let BinOpKind::And = binop
            && let ExprKind::If(cond, ..) = &right.kind
        {
            Err(self.dcx().create_err(errors::UnexpectedIfWithIf(
                binop_span.shrink_to_hi().to(cond.span.shrink_to_lo()),
            )))
        } else {
            Ok(())
        }
    }

    fn parse_for_head(&mut self) -> PResult<'a, (P<Pat>, P<Expr>)> {
        let begin_paren = if self.token == token::OpenParen {
            // Record whether we are about to parse `for (`.
            // This is used below for recovery in case of `for ( $stuff ) $block`
            // in which case we will suggest `for $stuff $block`.
            let start_span = self.token.span;
            let left = self.prev_token.span.between(self.look_ahead(1, |t| t.span));
            Some((start_span, left))
        } else {
            None
        };
        // Try to parse the pattern `for ($PAT) in $EXPR`.
        let pat = match (
            self.parse_pat_allow_top_guard(
                None,
                RecoverComma::Yes,
                RecoverColon::Yes,
                CommaRecoveryMode::LikelyTuple,
            ),
            begin_paren,
        ) {
            (Ok(pat), _) => pat, // Happy path.
            (Err(err), Some((start_span, left))) if self.eat_keyword(exp!(In)) => {
                // We know for sure we have seen `for ($SOMETHING in`. In the happy path this would
                // happen right before the return of this method.
                let attrs = self.parse_outer_attributes()?;
                let (expr, _) = match self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, attrs) {
                    Ok(expr) => expr,
                    Err(expr_err) => {
                        // We don't know what followed the `in`, so cancel and bubble up the
                        // original error.
                        expr_err.cancel();
                        return Err(err);
                    }
                };
                return if self.token == token::CloseParen {
                    // We know for sure we have seen `for ($SOMETHING in $EXPR)`, so we recover the
                    // parser state and emit a targeted suggestion.
                    let span = vec![start_span, self.token.span];
                    let right = self.prev_token.span.between(self.look_ahead(1, |t| t.span));
                    self.bump(); // )
                    err.cancel();
                    self.dcx().emit_err(errors::ParenthesesInForHead {
                        span,
                        // With e.g. `for (x) in y)` this would replace `(x) in y)`
                        // with `x) in y)` which is syntactically invalid.
                        // However, this is prevented before we get here.
                        sugg: errors::ParenthesesInForHeadSugg { left, right },
                    });
                    Ok((self.mk_pat(start_span.to(right), ast::PatKind::Wild), expr))
                } else {
                    Err(err) // Some other error, bubble up.
                };
            }
            (Err(err), _) => return Err(err), // Some other error, bubble up.
        };
        if !self.eat_keyword(exp!(In)) {
            self.error_missing_in_for_loop();
        }
        self.check_for_for_in_in_typo(self.prev_token.span);
        let attrs = self.parse_outer_attributes()?;
        let (expr, _) = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, attrs)?;
        Ok((pat, expr))
    }

    /// Parses `for await? <src_pat> in <src_expr> <src_loop_block>` (`for` token already eaten).
    fn parse_expr_for(&mut self, opt_label: Option<Label>, lo: Span) -> PResult<'a, P<Expr>> {
        let is_await =
            self.token_uninterpolated_span().at_least_rust_2018() && self.eat_keyword(exp!(Await));

        if is_await {
            self.psess.gated_spans.gate(sym::async_for_loop, self.prev_token.span);
        }

        let kind = if is_await { ForLoopKind::ForAwait } else { ForLoopKind::For };

        let (pat, expr) = self.parse_for_head()?;
        // Recover from missing expression in `for` loop
        if matches!(expr.kind, ExprKind::Block(..))
            && self.token.kind != token::OpenBrace
            && self.may_recover()
        {
            let guar = self
                .dcx()
                .emit_err(errors::MissingExpressionInForLoop { span: expr.span.shrink_to_lo() });
            let err_expr = self.mk_expr(expr.span, ExprKind::Err(guar));
            let block = self.mk_block(thin_vec![], BlockCheckMode::Default, self.prev_token.span);
            return Ok(self.mk_expr(
                lo.to(self.prev_token.span),
                ExprKind::ForLoop { pat, iter: err_expr, body: block, label: opt_label, kind },
            ));
        }

        let (attrs, loop_block) = self.parse_inner_attrs_and_block(
            // Only suggest moving erroneous block label to the loop header
            // if there is not already a label there
            opt_label.is_none().then_some(lo),
        )?;

        let kind = ExprKind::ForLoop { pat, iter: expr, body: loop_block, label: opt_label, kind };

        self.recover_loop_else("for", lo)?;

        Ok(self.mk_expr_with_attrs(lo.to(self.prev_token.span), kind, attrs))
    }

    /// Recovers from an `else` clause after a loop (`for...else`, `while...else`)
    fn recover_loop_else(&mut self, loop_kind: &'static str, loop_kw: Span) -> PResult<'a, ()> {
        if self.token.is_keyword(kw::Else) && self.may_recover() {
            let else_span = self.token.span;
            self.bump();
            let else_clause = self.parse_expr_else()?;
            self.dcx().emit_err(errors::LoopElseNotSupported {
                span: else_span.to(else_clause.span),
                loop_kind,
                loop_kw,
            });
        }
        Ok(())
    }

    fn error_missing_in_for_loop(&mut self) {
        let (span, sub): (_, fn(_) -> _) = if self.token.is_ident_named(sym::of) {
            // Possibly using JS syntax (#75311).
            let span = self.token.span;
            self.bump();
            (span, errors::MissingInInForLoopSub::InNotOf)
        } else {
            (self.prev_token.span.between(self.token.span), errors::MissingInInForLoopSub::AddIn)
        };

        self.dcx().emit_err(errors::MissingInInForLoop { span, sub: sub(span) });
    }

    /// Parses a `while` or `while let` expression (`while` token already eaten).
    fn parse_expr_while(&mut self, opt_label: Option<Label>, lo: Span) -> PResult<'a, P<Expr>> {
        let policy = LetChainsPolicy::EditionDependent { current_edition: lo.edition() };
        let cond = self.parse_expr_cond(policy).map_err(|mut err| {
            err.span_label(lo, "while parsing the condition of this `while` expression");
            err
        })?;
        let (attrs, body) = self
            .parse_inner_attrs_and_block(
                // Only suggest moving erroneous block label to the loop header
                // if there is not already a label there
                opt_label.is_none().then_some(lo),
            )
            .map_err(|mut err| {
                err.span_label(lo, "while parsing the body of this `while` expression");
                err.span_label(cond.span, "this `while` condition successfully parsed");
                err
            })?;

        self.recover_loop_else("while", lo)?;

        Ok(self.mk_expr_with_attrs(
            lo.to(self.prev_token.span),
            ExprKind::While(cond, body, opt_label),
            attrs,
        ))
    }

    /// Parses `loop { ... }` (`loop` token already eaten).
    fn parse_expr_loop(&mut self, opt_label: Option<Label>, lo: Span) -> PResult<'a, P<Expr>> {
        let loop_span = self.prev_token.span;
        let (attrs, body) = self.parse_inner_attrs_and_block(
            // Only suggest moving erroneous block label to the loop header
            // if there is not already a label there
            opt_label.is_none().then_some(lo),
        )?;
        self.recover_loop_else("loop", lo)?;
        Ok(self.mk_expr_with_attrs(
            lo.to(self.prev_token.span),
            ExprKind::Loop(body, opt_label, loop_span),
            attrs,
        ))
    }

    pub(crate) fn eat_label(&mut self) -> Option<Label> {
        if let Some((ident, is_raw)) = self.token.lifetime() {
            // Disallow `'fn`, but with a better error message than `expect_lifetime`.
            if matches!(is_raw, IdentIsRaw::No) && ident.without_first_quote().is_reserved() {
                self.dcx().emit_err(errors::InvalidLabel { span: ident.span, name: ident.name });
            }

            self.bump();
            Some(Label { ident })
        } else {
            None
        }
    }

    /// Parses a `match ... { ... }` expression (`match` token already eaten).
    fn parse_expr_match(&mut self) -> PResult<'a, P<Expr>> {
        let match_span = self.prev_token.span;
        let attrs = self.parse_outer_attributes()?;
        let (scrutinee, _) = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, attrs)?;

        self.parse_match_block(match_span, match_span, scrutinee, MatchKind::Prefix)
    }

    /// Parses the block of a `match expr { ... }` or a `expr.match { ... }`
    /// expression. This is after the match token and scrutinee are eaten
    fn parse_match_block(
        &mut self,
        lo: Span,
        match_span: Span,
        scrutinee: P<Expr>,
        match_kind: MatchKind,
    ) -> PResult<'a, P<Expr>> {
        if let Err(mut e) = self.expect(exp!(OpenBrace)) {
            if self.token == token::Semi {
                e.span_suggestion_short(
                    match_span,
                    "try removing this `match`",
                    "",
                    Applicability::MaybeIncorrect, // speculative
                );
            }
            if self.maybe_recover_unexpected_block_label(None) {
                e.cancel();
                self.bump();
            } else {
                return Err(e);
            }
        }
        let attrs = self.parse_inner_attributes()?;

        let mut arms = ThinVec::new();
        while self.token != token::CloseBrace {
            match self.parse_arm() {
                Ok(arm) => arms.push(arm),
                Err(e) => {
                    // Recover by skipping to the end of the block.
                    let guar = e.emit();
                    self.recover_stmt();
                    let span = lo.to(self.token.span);
                    if self.token == token::CloseBrace {
                        self.bump();
                    }
                    // Always push at least one arm to make the match non-empty
                    arms.push(Arm {
                        attrs: Default::default(),
                        pat: self.mk_pat(span, ast::PatKind::Err(guar)),
                        guard: None,
                        body: Some(self.mk_expr_err(span, guar)),
                        span,
                        id: DUMMY_NODE_ID,
                        is_placeholder: false,
                    });
                    return Ok(self.mk_expr_with_attrs(
                        span,
                        ExprKind::Match(scrutinee, arms, match_kind),
                        attrs,
                    ));
                }
            }
        }
        let hi = self.token.span;
        self.bump();
        Ok(self.mk_expr_with_attrs(lo.to(hi), ExprKind::Match(scrutinee, arms, match_kind), attrs))
    }

    /// Attempt to recover from match arm body with statements and no surrounding braces.
    fn parse_arm_body_missing_braces(
        &mut self,
        first_expr: &P<Expr>,
        arrow_span: Span,
    ) -> Option<(Span, ErrorGuaranteed)> {
        if self.token != token::Semi {
            return None;
        }
        let start_snapshot = self.create_snapshot_for_diagnostic();
        let semi_sp = self.token.span;
        self.bump(); // `;`
        let mut stmts =
            vec![self.mk_stmt(first_expr.span, ast::StmtKind::Expr(first_expr.clone()))];
        let err = |this: &Parser<'_>, stmts: Vec<ast::Stmt>| {
            let span = stmts[0].span.to(stmts[stmts.len() - 1].span);

            let guar = this.dcx().emit_err(errors::MatchArmBodyWithoutBraces {
                statements: span,
                arrow: arrow_span,
                num_statements: stmts.len(),
                sub: if stmts.len() > 1 {
                    errors::MatchArmBodyWithoutBracesSugg::AddBraces {
                        left: span.shrink_to_lo(),
                        right: span.shrink_to_hi(),
                    }
                } else {
                    errors::MatchArmBodyWithoutBracesSugg::UseComma { semicolon: semi_sp }
                },
            });
            (span, guar)
        };
        // We might have either a `,` -> `;` typo, or a block without braces. We need
        // a more subtle parsing strategy.
        loop {
            if self.token == token::CloseBrace {
                // We have reached the closing brace of the `match` expression.
                return Some(err(self, stmts));
            }
            if self.token == token::Comma {
                self.restore_snapshot(start_snapshot);
                return None;
            }
            let pre_pat_snapshot = self.create_snapshot_for_diagnostic();
            match self.parse_pat_no_top_alt(None, None) {
                Ok(_pat) => {
                    if self.token == token::FatArrow {
                        // Reached arm end.
                        self.restore_snapshot(pre_pat_snapshot);
                        return Some(err(self, stmts));
                    }
                }
                Err(err) => {
                    err.cancel();
                }
            }

            self.restore_snapshot(pre_pat_snapshot);
            match self.parse_stmt_without_recovery(true, ForceCollect::No, false) {
                // Consume statements for as long as possible.
                Ok(Some(stmt)) => {
                    stmts.push(stmt);
                }
                Ok(None) => {
                    self.restore_snapshot(start_snapshot);
                    break;
                }
                // We couldn't parse either yet another statement missing it's
                // enclosing block nor the next arm's pattern or closing brace.
                Err(stmt_err) => {
                    stmt_err.cancel();
                    self.restore_snapshot(start_snapshot);
                    break;
                }
            }
        }
        None
    }

    pub(super) fn parse_arm(&mut self) -> PResult<'a, Arm> {
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let lo = this.token.span;
            let (pat, guard) = this.parse_match_arm_pat_and_guard()?;

            let span_before_body = this.prev_token.span;
            let arm_body;
            let is_fat_arrow = this.check(exp!(FatArrow));
            let is_almost_fat_arrow =
                TokenKind::FatArrow.similar_tokens().contains(&this.token.kind);

            // this avoids the compiler saying that a `,` or `}` was expected even though
            // the pattern isn't a never pattern (and thus an arm body is required)
            let armless = (!is_fat_arrow && !is_almost_fat_arrow && pat.could_be_never_pattern())
                || matches!(this.token.kind, token::Comma | token::CloseBrace);

            let mut result = if armless {
                // A pattern without a body, allowed for never patterns.
                arm_body = None;
                let span = lo.to(this.prev_token.span);
                this.expect_one_of(&[exp!(Comma)], &[exp!(CloseBrace)]).map(|x| {
                    // Don't gate twice
                    if !pat.contains_never_pattern() {
                        this.psess.gated_spans.gate(sym::never_patterns, span);
                    }
                    x
                })
            } else {
                if let Err(mut err) = this.expect(exp!(FatArrow)) {
                    // We might have a `=>` -> `=` or `->` typo (issue #89396).
                    if is_almost_fat_arrow {
                        err.span_suggestion(
                            this.token.span,
                            "use a fat arrow to start a match arm",
                            "=>",
                            Applicability::MachineApplicable,
                        );
                        if matches!(
                            (&this.prev_token.kind, &this.token.kind),
                            (token::DotDotEq, token::Gt)
                        ) {
                            // `error_inclusive_range_match_arrow` handles cases like `0..=> {}`,
                            // so we suppress the error here
                            err.delay_as_bug();
                        } else {
                            err.emit();
                        }
                        this.bump();
                    } else {
                        return Err(err);
                    }
                }
                let arrow_span = this.prev_token.span;
                let arm_start_span = this.token.span;

                let attrs = this.parse_outer_attributes()?;
                let (expr, _) =
                    this.parse_expr_res(Restrictions::STMT_EXPR, attrs).map_err(|mut err| {
                        err.span_label(arrow_span, "while parsing the `match` arm starting here");
                        err
                    })?;

                let require_comma =
                    !classify::expr_is_complete(&expr) && this.token != token::CloseBrace;

                if !require_comma {
                    arm_body = Some(expr);
                    // Eat a comma if it exists, though.
                    let _ = this.eat(exp!(Comma));
                    Ok(Recovered::No)
                } else if let Some((span, guar)) =
                    this.parse_arm_body_missing_braces(&expr, arrow_span)
                {
                    let body = this.mk_expr_err(span, guar);
                    arm_body = Some(body);
                    Ok(Recovered::Yes(guar))
                } else {
                    let expr_span = expr.span;
                    arm_body = Some(expr);
                    this.expect_one_of(&[exp!(Comma)], &[exp!(CloseBrace)]).map_err(|mut err| {
                        if this.token == token::FatArrow {
                            let sm = this.psess.source_map();
                            if let Ok(expr_lines) = sm.span_to_lines(expr_span)
                                && let Ok(arm_start_lines) = sm.span_to_lines(arm_start_span)
                                && expr_lines.lines.len() == 2
                            {
                                if arm_start_lines.lines[0].end_col == expr_lines.lines[0].end_col {
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
                                        ",",
                                        Applicability::MachineApplicable,
                                    );
                                } else if arm_start_lines.lines[0].end_col + rustc_span::CharPos(1)
                                    == expr_lines.lines[0].end_col
                                {
                                    // similar to the above, but we may typo a `.` or `/` at the end of the line
                                    let comma_span = arm_start_span
                                        .shrink_to_hi()
                                        .with_hi(arm_start_span.hi() + rustc_span::BytePos(1));
                                    if let Ok(res) = sm.span_to_snippet(comma_span)
                                        && (res == "." || res == "/")
                                    {
                                        err.span_suggestion_short(
                                            comma_span,
                                            "you might have meant to write a `,` to end this `match` arm",
                                            ",",
                                            Applicability::MachineApplicable,
                                        );
                                    }
                                }
                            }
                        } else {
                            err.span_label(
                                arrow_span,
                                "while parsing the `match` arm starting here",
                            );
                        }
                        err
                    })
                }
            };

            let hi_span = arm_body.as_ref().map_or(span_before_body, |body| body.span);
            let arm_span = lo.to(hi_span);

            // We want to recover:
            // X |     Some(_) => foo()
            //   |                     - missing comma
            // X |     None => "x"
            //   |     ^^^^ self.token.span
            // as well as:
            // X |     Some(!)
            //   |            - missing comma
            // X |     None => "x"
            //   |     ^^^^ self.token.span
            // But we musn't recover
            // X |     pat[0] => {}
            //   |        ^ self.token.span
            let recover_missing_comma = arm_body.is_some() || pat.could_be_never_pattern();
            if recover_missing_comma {
                result = result.or_else(|err| {
                    // FIXME(compiler-errors): We could also recover `; PAT =>` here

                    // Try to parse a following `PAT =>`, if successful
                    // then we should recover.
                    let mut snapshot = this.create_snapshot_for_diagnostic();
                    let pattern_follows = snapshot
                        .parse_pat_no_top_guard(
                            None,
                            RecoverComma::Yes,
                            RecoverColon::Yes,
                            CommaRecoveryMode::EitherTupleOrPipe,
                        )
                        .map_err(|err| err.cancel())
                        .is_ok();
                    if pattern_follows && snapshot.check(exp!(FatArrow)) {
                        err.cancel();
                        let guar = this.dcx().emit_err(errors::MissingCommaAfterMatchArm {
                            span: arm_span.shrink_to_hi(),
                        });
                        return Ok(Recovered::Yes(guar));
                    }
                    Err(err)
                });
            }
            result?;

            Ok((
                ast::Arm {
                    attrs,
                    pat,
                    guard,
                    body: arm_body,
                    span: arm_span,
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                Trailing::No,
                UsePreAttrPos::No,
            ))
        })
    }

    fn parse_match_arm_guard(&mut self) -> PResult<'a, Option<P<Expr>>> {
        // Used to check the `if_let_guard` feature mostly by scanning
        // `&&` tokens.
        fn has_let_expr(expr: &Expr) -> bool {
            match &expr.kind {
                ExprKind::Binary(BinOp { node: BinOpKind::And, .. }, lhs, rhs) => {
                    let lhs_rslt = has_let_expr(lhs);
                    let rhs_rslt = has_let_expr(rhs);
                    lhs_rslt || rhs_rslt
                }
                ExprKind::Let(..) => true,
                _ => false,
            }
        }
        if !self.eat_keyword(exp!(If)) {
            // No match arm guard present.
            return Ok(None);
        }

        let if_span = self.prev_token.span;
        let mut cond = self.parse_match_guard_condition()?;

        CondChecker::new(self, LetChainsPolicy::AlwaysAllowed).visit_expr(&mut cond);

        if has_let_expr(&cond) {
            let span = if_span.to(cond.span);
            self.psess.gated_spans.gate(sym::if_let_guard, span);
        }
        Ok(Some(cond))
    }

    fn parse_match_arm_pat_and_guard(&mut self) -> PResult<'a, (P<Pat>, Option<P<Expr>>)> {
        if self.token == token::OpenParen {
            let left = self.token.span;
            let pat = self.parse_pat_no_top_guard(
                None,
                RecoverComma::Yes,
                RecoverColon::Yes,
                CommaRecoveryMode::EitherTupleOrPipe,
            )?;
            if let ast::PatKind::Paren(subpat) = &pat.kind
                && let ast::PatKind::Guard(..) = &subpat.kind
            {
                // Detect and recover from `($pat if $cond) => $arm`.
                // FIXME(guard_patterns): convert this to a normal guard instead
                let span = pat.span;
                let ast::PatKind::Paren(subpat) = pat.kind else { unreachable!() };
                let ast::PatKind::Guard(_, mut cond) = subpat.kind else { unreachable!() };
                self.psess.gated_spans.ungate_last(sym::guard_patterns, cond.span);
                CondChecker::new(self, LetChainsPolicy::AlwaysAllowed).visit_expr(&mut cond);
                let right = self.prev_token.span;
                self.dcx().emit_err(errors::ParenthesesInMatchPat {
                    span: vec![left, right],
                    sugg: errors::ParenthesesInMatchPatSugg { left, right },
                });
                Ok((self.mk_pat(span, ast::PatKind::Wild), Some(cond)))
            } else {
                Ok((pat, self.parse_match_arm_guard()?))
            }
        } else {
            // Regular parser flow:
            let pat = self.parse_pat_no_top_guard(
                None,
                RecoverComma::Yes,
                RecoverColon::Yes,
                CommaRecoveryMode::EitherTupleOrPipe,
            )?;
            Ok((pat, self.parse_match_arm_guard()?))
        }
    }

    fn parse_match_guard_condition(&mut self) -> PResult<'a, P<Expr>> {
        let attrs = self.parse_outer_attributes()?;
        match self.parse_expr_res(Restrictions::ALLOW_LET | Restrictions::IN_IF_GUARD, attrs) {
            Ok((expr, _)) => Ok(expr),
            Err(mut err) => {
                if self.prev_token == token::OpenBrace {
                    let sugg_sp = self.prev_token.span.shrink_to_lo();
                    // Consume everything within the braces, let's avoid further parse
                    // errors.
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    let msg = "you might have meant to start a match arm after the match guard";
                    if self.eat(exp!(CloseBrace)) {
                        let applicability = if self.token != token::FatArrow {
                            // We have high confidence that we indeed didn't have a struct
                            // literal in the match guard, but rather we had some operation
                            // that ended in a path, immediately followed by a block that was
                            // meant to be the match arm.
                            Applicability::MachineApplicable
                        } else {
                            Applicability::MaybeIncorrect
                        };
                        err.span_suggestion_verbose(sugg_sp, msg, "=> ", applicability);
                    }
                }
                Err(err)
            }
        }
    }

    pub(crate) fn is_builtin(&self) -> bool {
        self.token.is_keyword(kw::Builtin) && self.look_ahead(1, |t| *t == token::Pound)
    }

    /// Parses a `try {...}` expression (`try` token already eaten).
    fn parse_try_block(&mut self, span_lo: Span) -> PResult<'a, P<Expr>> {
        let (attrs, body) = self.parse_inner_attrs_and_block(None)?;
        if self.eat_keyword(exp!(Catch)) {
            Err(self.dcx().create_err(errors::CatchAfterTry { span: self.prev_token.span }))
        } else {
            let span = span_lo.to(body.span);
            self.psess.gated_spans.gate(sym::try_blocks, span);
            Ok(self.mk_expr_with_attrs(span, ExprKind::TryBlock(body), attrs))
        }
    }

    fn is_do_catch_block(&self) -> bool {
        self.token.is_keyword(kw::Do)
            && self.is_keyword_ahead(1, &[kw::Catch])
            && self.look_ahead(2, |t| *t == token::OpenBrace || t.is_metavar_block())
            && !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_do_yeet(&self) -> bool {
        self.token.is_keyword(kw::Do) && self.is_keyword_ahead(1, &[kw::Yeet])
    }

    fn is_try_block(&self) -> bool {
        self.token.is_keyword(kw::Try)
            && self.look_ahead(1, |t| *t == token::OpenBrace || t.is_metavar_block())
            && self.token_uninterpolated_span().at_least_rust_2018()
    }

    /// Parses an `async move? {...}` or `gen move? {...}` expression.
    fn parse_gen_block(&mut self) -> PResult<'a, P<Expr>> {
        let lo = self.token.span;
        let kind = if self.eat_keyword(exp!(Async)) {
            if self.eat_keyword(exp!(Gen)) { GenBlockKind::AsyncGen } else { GenBlockKind::Async }
        } else {
            assert!(self.eat_keyword(exp!(Gen)));
            GenBlockKind::Gen
        };
        match kind {
            GenBlockKind::Async => {
                // `async` blocks are stable
            }
            GenBlockKind::Gen | GenBlockKind::AsyncGen => {
                self.psess.gated_spans.gate(sym::gen_blocks, lo.to(self.prev_token.span));
            }
        }
        let capture_clause = self.parse_capture_clause()?;
        let decl_span = lo.to(self.prev_token.span);
        let (attrs, body) = self.parse_inner_attrs_and_block(None)?;
        let kind = ExprKind::Gen(capture_clause, body, kind, decl_span);
        Ok(self.mk_expr_with_attrs(lo.to(self.prev_token.span), kind, attrs))
    }

    fn is_gen_block(&self, kw: Symbol, lookahead: usize) -> bool {
        self.is_keyword_ahead(lookahead, &[kw])
            && ((
                // `async move {`
                self.is_keyword_ahead(lookahead + 1, &[kw::Move, kw::Use])
                    && self.look_ahead(lookahead + 2, |t| {
                        *t == token::OpenBrace || t.is_metavar_block()
                    })
            ) || (
                // `async {`
                self.look_ahead(lookahead + 1, |t| *t == token::OpenBrace || t.is_metavar_block())
            ))
    }

    pub(super) fn is_async_gen_block(&self) -> bool {
        self.token.is_keyword(kw::Async) && self.is_gen_block(kw::Gen, 1)
    }

    fn is_certainly_not_a_block(&self) -> bool {
        // `{ ident, ` and `{ ident: ` cannot start a block.
        self.look_ahead(1, |t| t.is_ident())
            && self.look_ahead(2, |t| t == &token::Comma || t == &token::Colon)
    }

    fn maybe_parse_struct_expr(
        &mut self,
        qself: &Option<P<ast::QSelf>>,
        path: &ast::Path,
    ) -> Option<PResult<'a, P<Expr>>> {
        let struct_allowed = !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
        if struct_allowed || self.is_certainly_not_a_block() {
            if let Err(err) = self.expect(exp!(OpenBrace)) {
                return Some(Err(err));
            }
            let expr = self.parse_expr_struct(qself.clone(), path.clone(), true);
            if let (Ok(expr), false) = (&expr, struct_allowed) {
                // This is a struct literal, but we don't can't accept them here.
                self.dcx().emit_err(errors::StructLiteralNotAllowedHere {
                    span: expr.span,
                    sub: errors::StructLiteralNotAllowedHereSugg {
                        left: path.span.shrink_to_lo(),
                        right: expr.span.shrink_to_hi(),
                    },
                });
            }
            return Some(expr);
        }
        None
    }

    pub(super) fn parse_struct_fields(
        &mut self,
        pth: ast::Path,
        recover: bool,
        close: ExpTokenPair<'_>,
    ) -> PResult<
        'a,
        (
            ThinVec<ExprField>,
            ast::StructRest,
            Option<ErrorGuaranteed>, /* async blocks are forbidden in Rust 2015 */
        ),
    > {
        let mut fields = ThinVec::new();
        let mut base = ast::StructRest::None;
        let mut recovered_async = None;
        let in_if_guard = self.restrictions.contains(Restrictions::IN_IF_GUARD);

        let async_block_err = |e: &mut Diag<'_>, span: Span| {
            errors::AsyncBlockIn2015 { span }.add_to_diag(e);
            errors::HelpUseLatestEdition::new().add_to_diag(e);
        };

        while self.token != *close.tok {
            if self.eat(exp!(DotDot)) || self.recover_struct_field_dots(close.tok) {
                let exp_span = self.prev_token.span;
                // We permit `.. }` on the left-hand side of a destructuring assignment.
                if self.check(close) {
                    base = ast::StructRest::Rest(self.prev_token.span);
                    break;
                }
                match self.parse_expr() {
                    Ok(e) => base = ast::StructRest::Base(e),
                    Err(e) if recover => {
                        e.emit();
                        self.recover_stmt();
                    }
                    Err(e) => return Err(e),
                }
                self.recover_struct_comma_after_dotdot(exp_span);
                break;
            }

            // Peek the field's ident before parsing its expr in order to emit better diagnostics.
            let peek = self
                .token
                .ident()
                .filter(|(ident, is_raw)| {
                    (!ident.is_reserved() || matches!(is_raw, IdentIsRaw::Yes))
                        && self.look_ahead(1, |tok| *tok == token::Colon)
                })
                .map(|(ident, _)| ident);

            // We still want a field even if its expr didn't parse.
            let field_ident = |this: &Self, guar: ErrorGuaranteed| {
                peek.map(|ident| {
                    let span = ident.span;
                    ExprField {
                        ident,
                        span,
                        expr: this.mk_expr_err(span, guar),
                        is_shorthand: false,
                        attrs: AttrVec::new(),
                        id: DUMMY_NODE_ID,
                        is_placeholder: false,
                    }
                })
            };

            let parsed_field = match self.parse_expr_field() {
                Ok(f) => Ok(f),
                Err(mut e) => {
                    if pth == kw::Async {
                        async_block_err(&mut e, pth.span);
                    } else {
                        e.span_label(pth.span, "while parsing this struct");
                    }

                    if let Some((ident, _)) = self.token.ident()
                        && !self.token.is_reserved_ident()
                        && self.look_ahead(1, |t| {
                            AssocOp::from_token(t).is_some()
                                || matches!(
                                    t.kind,
                                    token::OpenParen | token::OpenBracket | token::OpenBrace
                                )
                                || *t == token::Dot
                        })
                    {
                        // Looks like they tried to write a shorthand, complex expression,
                        // E.g.: `n + m`, `f(a)`, `a[i]`, `S { x: 3 }`, or `x.y`.
                        e.span_suggestion_verbose(
                            self.token.span.shrink_to_lo(),
                            "try naming a field",
                            &format!("{ident}: ",),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    if in_if_guard && close.token_type == TokenType::CloseBrace {
                        return Err(e);
                    }

                    if !recover {
                        return Err(e);
                    }

                    let guar = e.emit();
                    if pth == kw::Async {
                        recovered_async = Some(guar);
                    }

                    // If the next token is a comma, then try to parse
                    // what comes next as additional fields, rather than
                    // bailing out until next `}`.
                    if self.token != token::Comma {
                        self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                        if self.token != token::Comma {
                            break;
                        }
                    }

                    Err(guar)
                }
            };

            let is_shorthand = parsed_field.as_ref().is_ok_and(|f| f.is_shorthand);
            // A shorthand field can be turned into a full field with `:`.
            // We should point this out.
            self.check_or_expected(!is_shorthand, TokenType::Colon);

            match self.expect_one_of(&[exp!(Comma)], &[close]) {
                Ok(_) => {
                    if let Ok(f) = parsed_field.or_else(|guar| field_ident(self, guar).ok_or(guar))
                    {
                        // Only include the field if there's no parse error for the field name.
                        fields.push(f);
                    }
                }
                Err(mut e) => {
                    if pth == kw::Async {
                        async_block_err(&mut e, pth.span);
                    } else {
                        e.span_label(pth.span, "while parsing this struct");
                        if peek.is_some() {
                            e.span_suggestion(
                                self.prev_token.span.shrink_to_hi(),
                                "try adding a comma",
                                ",",
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                    if !recover {
                        return Err(e);
                    }
                    let guar = e.emit();
                    if pth == kw::Async {
                        recovered_async = Some(guar);
                    } else if let Some(f) = field_ident(self, guar) {
                        fields.push(f);
                    }
                    self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                    let _ = self.eat(exp!(Comma));
                }
            }
        }
        Ok((fields, base, recovered_async))
    }

    /// Precondition: already parsed the '{'.
    pub(super) fn parse_expr_struct(
        &mut self,
        qself: Option<P<ast::QSelf>>,
        pth: ast::Path,
        recover: bool,
    ) -> PResult<'a, P<Expr>> {
        let lo = pth.span;
        let (fields, base, recovered_async) =
            self.parse_struct_fields(pth.clone(), recover, exp!(CloseBrace))?;
        let span = lo.to(self.token.span);
        self.expect(exp!(CloseBrace))?;
        let expr = if let Some(guar) = recovered_async {
            ExprKind::Err(guar)
        } else {
            ExprKind::Struct(P(ast::StructExpr { qself, path: pth, fields, rest: base }))
        };
        Ok(self.mk_expr(span, expr))
    }

    fn recover_struct_comma_after_dotdot(&mut self, span: Span) {
        if self.token != token::Comma {
            return;
        }
        self.dcx().emit_err(errors::CommaAfterBaseStruct {
            span: span.to(self.prev_token.span),
            comma: self.token.span,
        });
        self.recover_stmt();
    }

    fn recover_struct_field_dots(&mut self, close: &TokenKind) -> bool {
        if !self.look_ahead(1, |t| t == close) && self.eat(exp!(DotDotDot)) {
            // recover from typo of `...`, suggest `..`
            let span = self.prev_token.span;
            self.dcx().emit_err(errors::MissingDotDot { token_span: span, sugg_span: span });
            return true;
        }
        false
    }

    /// Converts an ident into 'label and emits an "expected a label, found an identifier" error.
    fn recover_ident_into_label(&mut self, ident: Ident) -> Label {
        // Convert `label` -> `'label`,
        // so that nameres doesn't complain about non-existing label
        let label = format!("'{}", ident.name);
        let ident = Ident::new(Symbol::intern(&label), ident.span);

        self.dcx().emit_err(errors::ExpectedLabelFoundIdent {
            span: ident.span,
            start: ident.span.shrink_to_lo(),
        });

        Label { ident }
    }

    /// Parses `ident (COLON expr)?`.
    fn parse_expr_field(&mut self) -> PResult<'a, ExprField> {
        let attrs = self.parse_outer_attributes()?;
        self.recover_vcs_conflict_marker();
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let lo = this.token.span;

            // Check if a colon exists one ahead. This means we're parsing a fieldname.
            let is_shorthand = !this.look_ahead(1, |t| t == &token::Colon || t == &token::Eq);
            // Proactively check whether parsing the field will be incorrect.
            let is_wrong = this.token.is_ident()
                && !this.token.is_reserved_ident()
                && !this.look_ahead(1, |t| {
                    t == &token::Colon
                        || t == &token::Eq
                        || t == &token::Comma
                        || t == &token::CloseBrace
                        || t == &token::CloseParen
                });
            if is_wrong {
                return Err(this.dcx().create_err(errors::ExpectedStructField {
                    span: this.look_ahead(1, |t| t.span),
                    ident_span: this.token.span,
                    token: this.look_ahead(1, |t| *t),
                }));
            }
            let (ident, expr) = if is_shorthand {
                // Mimic `x: x` for the `x` field shorthand.
                let ident = this.parse_ident_common(false)?;
                let path = ast::Path::from_ident(ident);
                (ident, this.mk_expr(ident.span, ExprKind::Path(None, path)))
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
                    attrs,
                    id: DUMMY_NODE_ID,
                    is_placeholder: false,
                },
                Trailing::from(this.token == token::Comma),
                UsePreAttrPos::No,
            ))
        })
    }

    /// Check for `=`. This means the source incorrectly attempts to
    /// initialize a field with an eq rather than a colon.
    fn error_on_eq_field_init(&self, field_name: Ident) {
        if self.token != token::Eq {
            return;
        }

        self.dcx().emit_err(errors::EqFieldInit {
            span: self.token.span,
            eq: field_name.span.shrink_to_hi().to(self.token.span),
        });
    }

    fn err_dotdotdot_syntax(&self, span: Span) {
        self.dcx().emit_err(errors::DotDotDot { span });
    }

    fn err_larrow_operator(&self, span: Span) {
        self.dcx().emit_err(errors::LeftArrowOperator { span });
    }

    fn mk_assign_op(&self, assign_op: AssignOp, lhs: P<Expr>, rhs: P<Expr>) -> ExprKind {
        ExprKind::AssignOp(assign_op, lhs, rhs)
    }

    fn mk_range(
        &mut self,
        start: Option<P<Expr>>,
        end: Option<P<Expr>>,
        limits: RangeLimits,
    ) -> ExprKind {
        if end.is_none() && limits == RangeLimits::Closed {
            let guar = self.inclusive_range_with_incorrect_end();
            ExprKind::Err(guar)
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

    fn mk_index(&self, expr: P<Expr>, idx: P<Expr>, brackets_span: Span) -> ExprKind {
        ExprKind::Index(expr, idx, brackets_span)
    }

    fn mk_call(&self, f: P<Expr>, args: ThinVec<P<Expr>>) -> ExprKind {
        ExprKind::Call(f, args)
    }

    fn mk_await_expr(&mut self, self_arg: P<Expr>, lo: Span) -> P<Expr> {
        let span = lo.to(self.prev_token.span);
        let await_expr = self.mk_expr(span, ExprKind::Await(self_arg, self.prev_token.span));
        self.recover_from_await_method_call();
        await_expr
    }

    fn mk_use_expr(&mut self, self_arg: P<Expr>, lo: Span) -> P<Expr> {
        let span = lo.to(self.prev_token.span);
        let use_expr = self.mk_expr(span, ExprKind::Use(self_arg, self.prev_token.span));
        self.recover_from_use();
        use_expr
    }

    pub(crate) fn mk_expr_with_attrs(&self, span: Span, kind: ExprKind, attrs: AttrVec) -> P<Expr> {
        P(Expr { kind, span, attrs, id: DUMMY_NODE_ID, tokens: None })
    }

    pub(crate) fn mk_expr(&self, span: Span, kind: ExprKind) -> P<Expr> {
        self.mk_expr_with_attrs(span, kind, AttrVec::new())
    }

    pub(super) fn mk_expr_err(&self, span: Span, guar: ErrorGuaranteed) -> P<Expr> {
        self.mk_expr(span, ExprKind::Err(guar))
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
        f: impl FnOnce(&mut Self, ast::AttrVec) -> PResult<'a, P<Expr>>,
    ) -> PResult<'a, P<Expr>> {
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let res = f(this, attrs)?;
            let trailing = Trailing::from(
                this.restrictions.contains(Restrictions::STMT_EXPR)
                     && this.token == token::Semi
                // FIXME: pass an additional condition through from the place
                // where we know we need a comma, rather than assuming that
                // `#[attr] expr,` always captures a trailing comma.
                || this.token == token::Comma,
            );
            Ok((res, trailing, UsePreAttrPos::No))
        })
    }
}

/// Could this lifetime/label be an unclosed char literal? For example, `'a`
/// could be, but `'abc` could not.
pub(crate) fn could_be_unclosed_char_literal(ident: Ident) -> bool {
    ident.name.as_str().starts_with('\'')
        && unescape_char(ident.without_first_quote().name.as_str()).is_ok()
}

/// Used to forbid `let` expressions in certain syntactic locations.
#[derive(Clone, Copy, Subdiagnostic)]
pub(crate) enum ForbiddenLetReason {
    /// `let` is not valid and the source environment is not important
    OtherForbidden,
    /// A let chain with the `||` operator
    #[note(parse_not_supported_or)]
    NotSupportedOr(#[primary_span] Span),
    /// A let chain with invalid parentheses
    ///
    /// For example, `let 1 = 1 && (expr && expr)` is allowed
    /// but `(let 1 = 1 && (let 1 = 1 && (let 1 = 1))) && let a = 1` is not
    #[note(parse_not_supported_parentheses)]
    NotSupportedParentheses(#[primary_span] Span),
}

/// Whether let chains are allowed on all editions, or it's edition dependent (allowed only on
/// 2024 and later). In case of edition dependence, specify the currently present edition.
pub enum LetChainsPolicy {
    AlwaysAllowed,
    EditionDependent { current_edition: Edition },
}

/// Visitor to check for invalid use of `ExprKind::Let` that can't
/// easily be caught in parsing. For example:
///
/// ```rust,ignore (example)
/// // Only know that the let isn't allowed once the `||` token is reached
/// if let Some(x) = y || true {}
/// // Only know that the let isn't allowed once the second `=` token is reached.
/// if let Some(x) = y && z = 1 {}
/// ```
struct CondChecker<'a> {
    parser: &'a Parser<'a>,
    let_chains_policy: LetChainsPolicy,
    depth: u32,
    forbid_let_reason: Option<ForbiddenLetReason>,
    missing_let: Option<errors::MaybeMissingLet>,
    comparison: Option<errors::MaybeComparison>,
}

impl<'a> CondChecker<'a> {
    fn new(parser: &'a Parser<'a>, let_chains_policy: LetChainsPolicy) -> Self {
        CondChecker {
            parser,
            forbid_let_reason: None,
            missing_let: None,
            comparison: None,
            let_chains_policy,
            depth: 0,
        }
    }
}

impl MutVisitor for CondChecker<'_> {
    fn visit_expr(&mut self, e: &mut P<Expr>) {
        self.depth += 1;
        use ForbiddenLetReason::*;

        let span = e.span;
        match e.kind {
            ExprKind::Let(_, _, _, ref mut recovered @ Recovered::No) => {
                if let Some(reason) = self.forbid_let_reason {
                    let error = match reason {
                        NotSupportedOr(or_span) => {
                            self.parser.dcx().emit_err(errors::OrInLetChain { span: or_span })
                        }
                        _ => self.parser.dcx().emit_err(errors::ExpectedExpressionFoundLet {
                            span,
                            reason,
                            missing_let: self.missing_let,
                            comparison: self.comparison,
                        }),
                    };
                    *recovered = Recovered::Yes(error);
                } else if self.depth > 1 {
                    // Top level `let` is always allowed; only gate chains
                    match self.let_chains_policy {
                        LetChainsPolicy::AlwaysAllowed => (),
                        LetChainsPolicy::EditionDependent { current_edition } => {
                            if !current_edition.at_least_rust_2024() || !span.at_least_rust_2024() {
                                self.parser.psess.gated_spans.gate(sym::let_chains, span);
                            }
                        }
                    }
                }
            }
            ExprKind::Binary(Spanned { node: BinOpKind::And, .. }, _, _) => {
                mut_visit::walk_expr(self, e);
            }
            ExprKind::Binary(Spanned { node: BinOpKind::Or, span: or_span }, _, _)
                if let None | Some(NotSupportedOr(_)) = self.forbid_let_reason =>
            {
                let forbid_let_reason = self.forbid_let_reason;
                self.forbid_let_reason = Some(NotSupportedOr(or_span));
                mut_visit::walk_expr(self, e);
                self.forbid_let_reason = forbid_let_reason;
            }
            ExprKind::Paren(ref inner)
                if let None | Some(NotSupportedParentheses(_)) = self.forbid_let_reason =>
            {
                let forbid_let_reason = self.forbid_let_reason;
                self.forbid_let_reason = Some(NotSupportedParentheses(inner.span));
                mut_visit::walk_expr(self, e);
                self.forbid_let_reason = forbid_let_reason;
            }
            ExprKind::Assign(ref lhs, _, span) => {
                let forbid_let_reason = self.forbid_let_reason;
                self.forbid_let_reason = Some(OtherForbidden);
                let missing_let = self.missing_let;
                if let ExprKind::Binary(_, _, rhs) = &lhs.kind
                    && let ExprKind::Path(_, _)
                    | ExprKind::Struct(_)
                    | ExprKind::Call(_, _)
                    | ExprKind::Array(_) = rhs.kind
                {
                    self.missing_let =
                        Some(errors::MaybeMissingLet { span: rhs.span.shrink_to_lo() });
                }
                let comparison = self.comparison;
                self.comparison = Some(errors::MaybeComparison { span: span.shrink_to_hi() });
                mut_visit::walk_expr(self, e);
                self.forbid_let_reason = forbid_let_reason;
                self.missing_let = missing_let;
                self.comparison = comparison;
            }
            ExprKind::Unary(_, _)
            | ExprKind::Await(_, _)
            | ExprKind::Use(_, _)
            | ExprKind::AssignOp(_, _, _)
            | ExprKind::Range(_, _, _)
            | ExprKind::Try(_)
            | ExprKind::AddrOf(_, _, _)
            | ExprKind::Binary(_, _, _)
            | ExprKind::Field(_, _)
            | ExprKind::Index(_, _, _)
            | ExprKind::Call(_, _)
            | ExprKind::MethodCall(_)
            | ExprKind::Tup(_)
            | ExprKind::Paren(_) => {
                let forbid_let_reason = self.forbid_let_reason;
                self.forbid_let_reason = Some(OtherForbidden);
                mut_visit::walk_expr(self, e);
                self.forbid_let_reason = forbid_let_reason;
            }
            ExprKind::Cast(ref mut op, _)
            | ExprKind::Type(ref mut op, _)
            | ExprKind::UnsafeBinderCast(_, ref mut op, _) => {
                let forbid_let_reason = self.forbid_let_reason;
                self.forbid_let_reason = Some(OtherForbidden);
                self.visit_expr(op);
                self.forbid_let_reason = forbid_let_reason;
            }
            ExprKind::Let(_, _, _, Recovered::Yes(_))
            | ExprKind::Array(_)
            | ExprKind::ConstBlock(_)
            | ExprKind::Lit(_)
            | ExprKind::If(_, _, _)
            | ExprKind::While(_, _, _)
            | ExprKind::ForLoop { .. }
            | ExprKind::Loop(_, _, _)
            | ExprKind::Match(_, _, _)
            | ExprKind::Closure(_)
            | ExprKind::Block(_, _)
            | ExprKind::Gen(_, _, _, _)
            | ExprKind::TryBlock(_)
            | ExprKind::Underscore
            | ExprKind::Path(_, _)
            | ExprKind::Break(_, _)
            | ExprKind::Continue(_)
            | ExprKind::Ret(_)
            | ExprKind::InlineAsm(_)
            | ExprKind::OffsetOf(_, _)
            | ExprKind::MacCall(_)
            | ExprKind::Struct(_)
            | ExprKind::Repeat(_, _)
            | ExprKind::Yield(_)
            | ExprKind::Yeet(_)
            | ExprKind::Become(_)
            | ExprKind::IncludedBytes(_)
            | ExprKind::FormatArgs(_)
            | ExprKind::Err(_)
            | ExprKind::Dummy => {
                // These would forbid any let expressions they contain already.
            }
        }
        self.depth -= 1;
    }
}
