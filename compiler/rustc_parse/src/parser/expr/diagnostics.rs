use rustc_ast::util::parser::AssocOp;
use rustc_ast::{BinOpKind, Expr, ExprKind, token};
use rustc_errors::{Applicability, Diag, PResult};
use rustc_span::{Span, Spanned};

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
            Err(self.report_inc_dec_op(lhs, starts_stmt, IncOrDec::Inc, UnaryFixity::Post, op_span))
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
            Err(self.report_inc_dec_op(lhs, starts_stmt, IncOrDec::Dec, UnaryFixity::Post, op_span))
        } else {
            Ok(())
        }
    }

    /// Report increment operator `++` & decrement operator `--` as found in many C-style languages.
    pub(super) fn report_inc_dec_op(
        &mut self,
        base: &Expr,
        starts_stmt: bool,
        op: IncOrDec,
        fixity: UnaryFixity,
        op_span: Span,
    ) -> Diag<'a> {
        // FIXME: Don't return an error diag, emit the diag here *and* return a new expr of the form
        //        `$base += 1` / `$base -= 1` (taking `base: Expr` by value) for *proper* recovery.
        //        (Just emitting the diag would be insufficient since callers would most likely just
        //        use `$base` as the recovered AST node which would lead to annoying follow-up diags
        //        like "variable doesn't need to be mutable" getting emitted in some cases.)

        let mut err = {
            let fixity = match fixity {
                UnaryFixity::Pre => "prefix",
                UnaryFixity::Post => "postfix",
            };
            let op = match op {
                IncOrDec::Inc => "increment",
                IncOrDec::Dec => "decrement",
            };
            self.dcx()
                .struct_span_err(op_span, format!("Rust has no {fixity} {op} operator"))
                .with_span_label(op_span, format!("not a valid {fixity} operator"))
        };

        let op = match op {
            IncOrDec::Inc => "+= 1",
            IncOrDec::Dec => "-= 1",
        };
        let (pre_span, post_span) = match fixity {
            UnaryFixity::Pre => (op_span, base.span.shrink_to_hi()),
            UnaryFixity::Post => (base.span.shrink_to_lo(), op_span),
        };

        if starts_stmt {
            let mut patches = Vec::new();
            if !pre_span.is_empty() {
                patches.push((pre_span, String::new()));
            }
            patches.push((post_span, format!(" {op}")));
            err.multipart_suggestion(
                format!("use `{op}` instead"),
                patches,
                Applicability::MachineApplicable,
            );
        } else {
            let Ok(base_src) = self.span_to_snippet(base.span) else {
                err.help(format!("use `{op}` instead"));
                return err;
            };
            match fixity {
                UnaryFixity::Pre => {
                    err.multipart_suggestion(
                        format!("use `{op}` instead"),
                        vec![(pre_span, "{ ".into()), (post_span, format!(" {op}; {base_src} }}"))],
                        Applicability::MachineApplicable,
                    );
                }
                UnaryFixity::Post => {
                    // won't suggest since we can not handle the precedences
                    // for example: `a + b++` has been parsed (a + b)++ and we can not suggest here
                    if !matches!(base.kind, ExprKind::Binary(..)) {
                        let tmp_var = if base_src.trim() == "tmp" { "tmp_" } else { "tmp" };
                        err.multipart_suggestion(
                            format!("use `{op}` instead"),
                            vec![
                                (pre_span, format!("{{ let {tmp_var} = ")),
                                (post_span, format!("; {base_src} {op}; {tmp_var} }}")),
                            ],
                            Applicability::HasPlaceholders,
                        );
                    }
                }
            }
        }
        err
    }
}

#[derive(Copy, Clone)]
pub(super) enum IncOrDec {
    Inc,
    Dec,
}

#[derive(Copy, Clone)]
pub(super) enum UnaryFixity {
    Pre,
    Post,
}
