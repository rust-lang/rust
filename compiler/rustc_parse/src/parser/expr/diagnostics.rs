use rustc_ast::util::parser::AssocOp;
use rustc_ast::{BinOpKind, Expr, token};
use rustc_errors::PResult;
use rustc_span::Spanned;

use crate::diagnostics;
use crate::parser::Parser;

impl<'a> Parser<'a> {
    /// Reject `...` being used as an expression operator.
    pub(super) fn reject_dotdotdot_expr_op(&self) {
        if self.token == token::DotDotDot {
            self.dcx().emit_err(diagnostics::DotDotDotExprOp { span: self.token.span });
        }
    }

    /// Reject `<-` being used as an expression operator.
    pub(super) fn reject_larrow_expr_op(&self) {
        if self.token == token::LArrow {
            self.dcx().emit_err(diagnostics::LArrowExprOp { span: self.token.span });
        }
    }

    /// Recover from strict equality operators `===` and `!==` as found in e.g., JS and PHP.
    pub(super) fn recover_from_strict_eq_op(&mut self, op: Spanned<AssocOp>) {
        if let AssocOp::Binary(bop @ BinOpKind::Eq | bop @ BinOpKind::Ne) = op.node
            && self.token == token::Eq
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let sp = op.span.to(self.token.span);
            let sugg = bop.as_str().into();
            let invalid = format!("{sugg}=");
            self.dcx().emit_err(diagnostics::InvalidComparisonOperator {
                span: sp,
                invalid: invalid.clone(),
                sub: diagnostics::InvalidComparisonOperatorSub::Correctable {
                    span: sp,
                    invalid,
                    correct: sugg,
                },
            });
            self.bump();
        }
    }

    /// Recover from inequality operator `<>` ("diamond") as found in e.g., PHP.
    pub(super) fn recover_from_diamond_ne_op(&mut self, op: Spanned<AssocOp>) {
        if op.node == AssocOp::Binary(BinOpKind::Lt)
            && self.token == token::Gt
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let sp = op.span.to(self.token.span);
            self.dcx().emit_err(diagnostics::InvalidComparisonOperator {
                span: sp,
                invalid: "<>".into(),
                sub: diagnostics::InvalidComparisonOperatorSub::Correctable {
                    span: sp,
                    invalid: "<>".into(),
                    correct: "!=".into(),
                },
            });
            self.bump();
        }
    }

    /// Recover from comparison operator `<=>` ("spaceship") as found in e.g., C++.
    pub(super) fn recover_from_spaceship_cmp_op(&mut self, op: Spanned<AssocOp>) {
        if op.node == AssocOp::Binary(BinOpKind::Le)
            && self.token == token::Gt
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let sp = op.span.to(self.token.span);
            self.dcx().emit_err(diagnostics::InvalidComparisonOperator {
                span: sp,
                invalid: "<=>".into(),
                sub: diagnostics::InvalidComparisonOperatorSub::Spaceship(sp),
            });
            self.bump();
        }
    }

    /// Recover from postfix increment operator `++` as found in many C-style languages.
    pub(super) fn recover_from_postfix_inc_op(
        &mut self,
        lhs: &Expr,
        starts_stmt: bool,
    ) -> PResult<'a, ()> {
        if let (token::Plus, token::Plus) = (self.prev_token.kind, self.token.kind)
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let op_span = self.prev_token.span.to(self.token.span);
            self.bump(); // eat the second `+`
            Err(self.recover_from_postfix_increment(lhs, op_span, starts_stmt))
        } else {
            Ok(())
        }
    }

    /// Recover from postfix decrement operator `--` as found in many C-style languages.
    pub(super) fn recover_from_postfix_dec_op(
        &mut self,
        lhs: &Expr,
        starts_stmt: bool,
    ) -> PResult<'a, ()> {
        if let (token::Minus, token::Minus) = (self.prev_token.kind, self.token.kind)
            && self.prev_token.span.hi() == self.token.span.lo()
            && !self.look_ahead(1, |tok| tok.can_begin_expr())
        {
            let op_span = self.prev_token.span.to(self.token.span);
            self.bump(); // eat the second `-`
            Err(self.recover_from_postfix_decrement(lhs, op_span, starts_stmt))
        } else {
            Ok(())
        }
    }
}
