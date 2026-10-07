use rustc_ast::util::parser::AssocOp;
use rustc_ast::{BinOpKind, Expr, ExprKind, token};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, Diag, PResult};
use rustc_span::{Span, Spanned, respan, sym};

use crate::parser::Parser;
use crate::{diagnostics, exp};

impl<'a> Parser<'a> {
    /// Recover from a binary operator after complete statement expression as in `{ 4 } / 2`.
    pub(super) fn recover_from_bin_op_after_complete_stmt_expr(&self, lhs: &Expr) -> bool {
        use BinOpKind::*;

        // Starting point: We've just parsed a *complete* stmt expr (e.g., a block like `{ 4 }`).

        let Some(op) = AssocOp::from_token(&self.token) else { return false };

        // We've now encountered a token that could(!) be interpreted as a binary operator. This
        // could mean that the user intended to write a binary operation where the left operand is
        // block-like. In that case they would need to parenthesize the LHS or the entire operation
        // (e.g., `{ 4 } / 2` -> `({ 4 }) / 2` or `({ 4 } / 2)`).

        if op_can_continue_stmt_expr_unambiguously(op) {
            // We know that the token (e.g., `/`) can't possibly begin a new statement or pattern.
            // Instead of letting the stmt/pat parser emit a generic & rather confusing diagnostic,
            // let's emit a more targeted one, and continue parsing the expression.

            self.dcx().emit_err(diagnostics::FoundExprWouldBeStmt {
                span: self.token.span,
                token: pprust::token_to_string(&self.token),
                suggestion: diagnostics::ExprParenthesesNeeded::surrounding(lhs.span),
            });
            return true;
        }

        // The token can begin a new statement or pattern; this means we're on the happy path!
        // Still, the user might've meant to write a bin op here but we can't tell at this stage of
        // compilation. So if it's a "common lookalike" let's proactively register it somewhere for
        // later stages of compilation to potentially retrieve (e.g., in typeck).
        //
        // We've left out `BitAnd` because guessing its intent is hard. We can make suggestions
        // based on the assumption that double-refs are rarely intentional, and closures are
        // distinct enough that they don't get mixed up with their return value.
        if let AssocOp::Binary(Add | And | BitOr | Mul | Or | Sub) = op {
            let sp = self.psess.source_map().start_point(self.token.span);
            self.psess
                .complete_stmt_exprs_before_bin_op_lookalike
                .borrow_mut()
                .insert(sp, lhs.span);
        }

        false
    }

    /// Recover from alphabetic logic operators `and` and `or` as found in e.g., Python and PHP.
    pub(super) fn recover_from_alpha_logic_op(&self) -> Option<Spanned<AssocOp>> {
        if self.may_recover()
            && let Some(ident) = self.token.non_raw_ident()
        {
            let (op, sub): (_, fn(_) -> _) = match ident.name {
                sym::and => (BinOpKind::And, diagnostics::InvalidLogicalOperatorSub::Conjunction),
                sym::or => (BinOpKind::Or, diagnostics::InvalidLogicalOperatorSub::Disjunction),
                _ => return None,
            };

            self.dcx().emit_err(diagnostics::InvalidLogicalOperator {
                span: self.token.span,
                incorrect: ident.name,
                sub: sub(self.token.span),
            });

            Some(respan(self.token.span, AssocOp::Binary(op)))
        } else {
            None
        }
    }

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
    pub(super) fn recover_from_diamond_ne_op(&mut self) {
        if let (token::Lt, token::Gt) = (self.prev_token.kind, self.token.kind)
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let sp = self.prev_token.span.to(self.token.span);
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
    pub(super) fn recover_from_spaceship_cmp_op(&mut self) {
        if let (token::Le, token::Gt) = (self.prev_token.kind, self.token.kind)
            && self.prev_token.span.hi() == self.token.span.lo()
        {
            let sp = self.prev_token.span.to(self.token.span);
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

    /// Recover from array expressions as found in C like `{0, 1, 2, 3}`.
    pub(super) fn recover_from_c_array(&mut self, lo: Span) -> Option<Box<Expr>> {
        if !self.may_recover()
            || self.token.kind != token::OpenBrace
            || self.look_ahead(1, |t| !matches!(t.kind, token::Literal(_)))
            || self.look_ahead(2, |t| t != &token::Comma)
            || self.look_ahead(3, |t| !t.can_begin_expr())
        {
            return None;
        }

        let mut snapshot = self.create_snapshot_for_diagnostic();
        match snapshot.parse_expr_array_or_repeat(exp!(CloseBrace)) {
            Ok(arr) => {
                let guar = self.dcx().emit_err(diagnostics::ArrayBracketsInsteadOfBraces {
                    span: arr.span,
                    sub: diagnostics::ArrayBracketsInsteadOfBracesSugg {
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
}

/// Whether this operator could be used to follow a complete statement expression unambiguously
/// during parse error recovery.
pub(in crate::parser) fn op_can_continue_stmt_expr_unambiguously(op: AssocOp) -> bool {
    // NOTE: At the time of writing, it's only safe to return `true` for tokens that don't share a
    //       prefix with statements or patterns. For context, statement lists and match arm bodies
    //       are (the only two) places that use `Restriction::STMT_EXPR` (in the happy path).

    use BinOpKind::*;
    match op {
        AssocOp::Assign | AssocOp::AssignOp(_) | AssocOp::Cast => true,
        AssocOp::Binary(bin_op) => match bin_op {
            | BitXor | Div | Ge | Gt | Le | Rem | Shr => true,
            | Add // unambiguous but would inhibit our recovery from unary plus
            | And // `&&` starts repeated borrow (e.g., `&&x`)
            | BitAnd // `&` starts borrow (e.g., `&x`)
            | BitOr // `|` starts closure (e.g., `|_| ()`) or leading vert (e.g., `| Some(_)`)
            | Eq | Ne // unambiguous but would inhibit our recovery from bad struct literals
            | Lt // `<` starts qualified path (e.g., `<T as X>::P`)
            | Mul // `*` starts unary deref (e.g., `*x`, `*x = 0`)
            | Or // `||` starts parameterless closure (e.g. `|| 0`)
            | Shl // `<<` starts repeated qualified path (e.g., `<<T>::Q as X>::P`)
            | Sub => false, // `-` starts unary negation (e.g, `-x`)
        },
        AssocOp::Range(_) => false, // `..`/`..=` start ranges w/o lower bound (e.g., `..`, `..=1`)
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
