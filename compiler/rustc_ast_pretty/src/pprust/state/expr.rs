use std::fmt::Write;

use ast::{ForLoopKind, MatchKind};
use itertools::{Itertools, Position};
use rustc_ast::ptr::P;
use rustc_ast::util::classify;
use rustc_ast::util::literal::escape_byte_str_symbol;
use rustc_ast::util::parser::{self, AssocOp, ExprPrecedence, Fixity};
use rustc_ast::{
    self as ast, BlockCheckMode, FormatAlignment, FormatArgPosition, FormatArgsPiece, FormatCount,
    FormatDebugHex, FormatSign, FormatTrait, token,
};

use crate::pp::Breaks::Inconsistent;
use crate::pprust::state::fixup::FixupContext;
use crate::pprust::state::{AnnNode, INDENT_UNIT, PrintState, State};

impl<'a> State<'a> {
    fn print_else(&mut self, els: Option<&ast::Expr>) {
        if let Some(_else) = els {
            match &_else.kind {
                // Another `else if` block.
                ast::ExprKind::If(i, then, e) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.word(" else if ");
                    self.print_expr_as_cond(i);
                    self.space();
                    self.print_block(then);
                    self.print_else(e.as_deref())
                }
                // Final `else` block.
                ast::ExprKind::Block(b, _) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.word(" else ");
                    self.print_block(b)
                }
                // Constraints would be great here!
                _ => {
                    panic!("print_if saw if with weird alternative");
                }
            }
        }
    }

    fn print_if(&mut self, test: &ast::Expr, blk: &ast::Block, elseopt: Option<&ast::Expr>) {
        self.head("if");
        self.print_expr_as_cond(test);
        self.space();
        self.print_block(blk);
        self.print_else(elseopt)
    }

    fn print_call_post(&mut self, args: &[P<ast::Expr>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    fn print_expr_as_cond(&mut self, expr: &ast::Expr) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr), FixupContext::new_cond())
    }

    /// Does `expr` need parentheses when printed in a condition position?
    ///
    /// These cases need parens due to the parse error observed in #26461: `if return {}`
    /// parses as the erroneous construct `if (return {})`, not `if (return) {}`.
    fn cond_needs_par(expr: &ast::Expr) -> bool {
        match expr.kind {
            ast::ExprKind::Break(..)
            | ast::ExprKind::Closure(..)
            | ast::ExprKind::Ret(..)
            | ast::ExprKind::Yeet(..) => true,
            _ => parser::contains_exterior_struct_lit(expr),
        }
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    pub(super) fn print_expr_cond_paren(
        &mut self,
        expr: &ast::Expr,
        needs_par: bool,
        mut fixup: FixupContext,
    ) {
        if needs_par {
            self.popen();

            // If we are surrounding the whole cond in parentheses, such as:
            //
            //     if (return Struct {}) {}
            //
            // then there is no need for parenthesizing the individual struct
            // expressions within. On the other hand if the whole cond is not
            // parenthesized, then print_expr must parenthesize exterior struct
            // literals.
            //
            //     if x == (Struct {}) {}
            //
            fixup = FixupContext::default();
        }

        self.print_expr(expr, fixup);

        if needs_par {
            self.pclose();
        }
    }

    fn print_expr_vec(&mut self, exprs: &[P<ast::Expr>]) {
        self.ibox(INDENT_UNIT);
        self.word("[");
        self.commasep_exprs(Inconsistent, exprs);
        self.word("]");
        self.end();
    }

    pub(super) fn print_expr_anon_const(
        &mut self,
        expr: &ast::AnonConst,
        attrs: &[ast::Attribute],
    ) {
        self.ibox(INDENT_UNIT);
        self.word("const");
        self.nbsp();
        if let ast::ExprKind::Block(block, None) = &expr.value.kind {
            self.cbox(0);
            self.ibox(0);
            self.print_block_with_attrs(block, attrs);
        } else {
            self.print_expr(&expr.value, FixupContext::default());
        }
        self.end();
    }

    fn print_expr_repeat(&mut self, element: &ast::Expr, count: &ast::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.word("[");
        self.print_expr(element, FixupContext::default());
        self.word_space(";");
        self.print_expr(&count.value, FixupContext::default());
        self.word("]");
        self.end();
    }

    fn print_expr_struct(
        &mut self,
        qself: &Option<P<ast::QSelf>>,
        path: &ast::Path,
        fields: &[ast::ExprField],
        rest: &ast::StructRest,
    ) {
        if let Some(qself) = qself {
            self.print_qpath(path, qself, true);
        } else {
            self.print_path(path, true, 0);
        }
        self.nbsp();
        self.word("{");
        let has_rest = match rest {
            ast::StructRest::Base(_) | ast::StructRest::Rest(_) => true,
            ast::StructRest::None => false,
        };
        if fields.is_empty() && !has_rest {
            self.word("}");
            return;
        }
        self.cbox(0);
        for (pos, field) in fields.iter().with_position() {
            let is_first = matches!(pos, Position::First | Position::Only);
            let is_last = matches!(pos, Position::Last | Position::Only);
            self.maybe_print_comment(field.span.hi());
            self.print_outer_attributes(&field.attrs);
            if is_first {
                self.space_if_not_bol();
            }
            if !field.is_shorthand {
                self.print_ident(field.ident);
                self.word_nbsp(":");
            }
            self.print_expr(&field.expr, FixupContext::default());
            if !is_last || has_rest {
                self.word_space(",");
            } else {
                self.trailing_comma_or_space();
            }
        }
        if has_rest {
            if fields.is_empty() {
                self.space();
            }
            self.word("..");
            if let ast::StructRest::Base(expr) = rest {
                self.print_expr(expr, FixupContext::default());
            }
            self.space();
        }
        self.offset(-INDENT_UNIT);
        self.end();
        self.word("}");
    }

    fn print_expr_tup(&mut self, exprs: &[P<ast::Expr>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, exprs);
        if exprs.len() == 1 {
            self.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &ast::Expr, args: &[P<ast::Expr>], fixup: FixupContext) {
        let needs_paren = match func.kind {
            ast::ExprKind::Field(..) => true,
            _ => func.precedence() < ExprPrecedence::Unambiguous,
        };

        // Independent of parenthesization related to precedence, we must
        // parenthesize `func` if this is a statement context in which without
        // parentheses, a statement boundary would occur inside `func` or
        // immediately after `func`.
        //
        // Suppose `func` represents `match () { _ => f }`. We must produce:
        //
        //     (match () { _ => f })();
        //
        // instead of:
        //
        //     match () { _ => f } ();
        //
        // because the latter is valid syntax but with the incorrect meaning.
        // It's a match-expression followed by tuple-expression, not a function
        // call.
        self.print_expr_cond_paren(func, needs_paren, fixup.leftmost_subexpression());

        self.print_call_post(args)
    }

    fn print_expr_method_call(
        &mut self,
        segment: &ast::PathSegment,
        receiver: &ast::Expr,
        base_args: &[P<ast::Expr>],
        fixup: FixupContext,
    ) {
        // Unlike in `print_expr_call`, no change to fixup here because
        // statement boundaries never occur in front of a `.` (or `?`) token.
        //
        //     match () { _ => f }.method();
        //
        // Parenthesizing only for precedence and not with regard to statement
        // boundaries, `$receiver.method()` can be parsed back as a statement
        // containing an expression if and only if `$receiver` can be parsed as
        // a statement containing an expression.
        self.print_expr_cond_paren(
            receiver,
            receiver.precedence() < ExprPrecedence::Unambiguous,
            fixup,
        );

        self.word(".");
        self.print_ident(segment.ident);
        if let Some(args) = &segment.args {
            self.print_generic_args(args, true);
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        fixup: FixupContext,
    ) {
        let assoc_op = AssocOp::from_ast_binop(op.node);
        let binop_prec = assoc_op.precedence();
        let left_prec = lhs.precedence();
        let right_prec = rhs.precedence();

        let (mut left_needs_paren, right_needs_paren) = match assoc_op.fixity() {
            Fixity::Left => (left_prec < binop_prec, right_prec <= binop_prec),
            Fixity::Right => (left_prec <= binop_prec, right_prec < binop_prec),
            Fixity::None => (left_prec <= binop_prec, right_prec <= binop_prec),
        };

        match (&lhs.kind, op.node) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&ast::ExprKind::Cast { .. }, ast::BinOpKind::Lt | ast::BinOpKind::Shl) => {
                left_needs_paren = true;
            }
            // We are given `(let _ = a) OP b`.
            //
            // - When `OP <= LAnd` we should print `let _ = a OP b` to avoid redundant parens
            //   as the parser will interpret this as `(let _ = a) OP b`.
            //
            // - Otherwise, e.g. when we have `(let a = b) < c` in AST,
            //   parens are required since the parser would interpret `let a = b < c` as
            //   `let a = (b < c)`. To achieve this, we force parens.
            (&ast::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(binop_prec) => {
                left_needs_paren = true;
            }
            _ => {}
        }

        self.print_expr_cond_paren(lhs, left_needs_paren, fixup.leftmost_subexpression());
        self.space();
        self.word_space(op.node.as_str());
        self.print_expr_cond_paren(rhs, right_needs_paren, fixup.subsequent_subexpression());
    }

    fn print_expr_unary(&mut self, op: ast::UnOp, expr: &ast::Expr, fixup: FixupContext) {
        self.word(op.as_str());
        self.print_expr_cond_paren(
            expr,
            expr.precedence() < ExprPrecedence::Prefix,
            fixup.subsequent_subexpression(),
        );
    }

    fn print_expr_addr_of(
        &mut self,
        kind: ast::BorrowKind,
        mutability: ast::Mutability,
        expr: &ast::Expr,
        fixup: FixupContext,
    ) {
        self.word("&");
        match kind {
            ast::BorrowKind::Ref => self.print_mutability(mutability, false),
            ast::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_cond_paren(
            expr,
            expr.precedence() < ExprPrecedence::Prefix,
            fixup.subsequent_subexpression(),
        );
    }

    pub(super) fn print_expr(&mut self, expr: &ast::Expr, fixup: FixupContext) {
        self.print_expr_outer_attr_style(expr, true, fixup)
    }

    pub(super) fn print_expr_outer_attr_style(
        &mut self,
        expr: &ast::Expr,
        is_inline: bool,
        mut fixup: FixupContext,
    ) {
        self.maybe_print_comment(expr.span.lo());

        let attrs = &expr.attrs;
        if is_inline {
            self.print_outer_attributes_inline(attrs);
        } else {
            self.print_outer_attributes(attrs);
        }

        self.ibox(INDENT_UNIT);

        // The Match subexpression in `match x {} - 1` must be parenthesized if
        // it is the leftmost subexpression in a statement:
        //
        //     (match x {}) - 1;
        //
        // But not otherwise:
        //
        //     let _ = match x {} - 1;
        //
        // Same applies to a small set of other expression kinds which eagerly
        // terminate a statement which opens with them.
        let needs_par = fixup.would_cause_statement_boundary(expr);
        if needs_par {
            self.popen();
            fixup = FixupContext::default();
        }

        self.ann.pre(self, AnnNode::Expr(expr));

        match &expr.kind {
            ast::ExprKind::Array(exprs) => {
                self.print_expr_vec(exprs);
            }
            ast::ExprKind::ConstBlock(anon_const) => {
                self.print_expr_anon_const(anon_const, attrs);
            }
            ast::ExprKind::Repeat(element, count) => {
                self.print_expr_repeat(element, count);
            }
            ast::ExprKind::Struct(se) => {
                self.print_expr_struct(&se.qself, &se.path, &se.fields, &se.rest);
            }
            ast::ExprKind::Tup(exprs) => {
                self.print_expr_tup(exprs);
            }
            ast::ExprKind::Call(func, args) => {
                self.print_expr_call(func, args, fixup);
            }
            ast::ExprKind::MethodCall(box ast::MethodCall { seg, receiver, args, .. }) => {
                self.print_expr_method_call(seg, receiver, args, fixup);
            }
            ast::ExprKind::Binary(op, lhs, rhs) => {
                self.print_expr_binary(*op, lhs, rhs, fixup);
            }
            ast::ExprKind::Unary(op, expr) => {
                self.print_expr_unary(*op, expr, fixup);
            }
            ast::ExprKind::AddrOf(k, m, expr) => {
                self.print_expr_addr_of(*k, *m, expr, fixup);
            }
            ast::ExprKind::Lit(token_lit) => {
                self.print_token_literal(*token_lit, expr.span);
            }
            ast::ExprKind::IncludedBytes(bytes) => {
                let lit = token::Lit::new(token::ByteStr, escape_byte_str_symbol(bytes), None);
                self.print_token_literal(lit, expr.span)
            }
            ast::ExprKind::Cast(expr, ty) => {
                self.print_expr_cond_paren(
                    expr,
                    expr.precedence() < ExprPrecedence::Cast,
                    fixup.leftmost_subexpression(),
                );
                self.space();
                self.word_space("as");
                self.print_type(ty);
            }
            ast::ExprKind::Type(expr, ty) => {
                self.word("builtin # type_ascribe");
                self.popen();
                self.ibox(0);
                self.print_expr(expr, FixupContext::default());

                self.word(",");
                self.space_if_not_bol();
                self.print_type(ty);

                self.end();
                self.pclose();
            }
            ast::ExprKind::Let(pat, scrutinee, _, _) => {
                self.print_let(pat, scrutinee, fixup);
            }
            ast::ExprKind::If(test, blk, elseopt) => self.print_if(test, blk, elseopt.as_deref()),
            ast::ExprKind::While(test, blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("while");
                self.print_expr_as_cond(test);
                self.space();
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::ForLoop { pat, iter, body, label, kind } => {
                if let Some(label) = label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("for");
                if kind == &ForLoopKind::ForAwait {
                    self.word_nbsp("await");
                }
                self.print_pat(pat);
                self.space();
                self.word_space("in");
                self.print_expr_as_cond(iter);
                self.space();
                self.print_block_with_attrs(body, attrs);
            }
            ast::ExprKind::Loop(blk, opt_label, _) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("loop");
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Match(expr, arms, match_kind) => {
                self.cbox(0);
                self.ibox(0);

                match match_kind {
                    MatchKind::Prefix => {
                        self.word_nbsp("match");
                        self.print_expr_as_cond(expr);
                        self.space();
                    }
                    MatchKind::Postfix => {
                        self.print_expr_cond_paren(
                            expr,
                            expr.precedence() < ExprPrecedence::Unambiguous,
                            fixup,
                        );
                        self.word_nbsp(".match");
                    }
                }

                self.bopen();
                self.print_inner_attributes_no_trailing_hardbreak(attrs);
                for arm in arms {
                    self.print_arm(arm);
                }
                let empty = attrs.is_empty() && arms.is_empty();
                self.bclose(expr.span, empty);
            }
            ast::ExprKind::Closure(box ast::Closure {
                binder,
                capture_clause,
                constness,
                coroutine_kind,
                movability,
                fn_decl,
                body,
                fn_decl_span: _,
                fn_arg_span: _,
            }) => {
                self.print_closure_binder(binder);
                self.print_constness(*constness);
                self.print_movability(*movability);
                coroutine_kind.map(|coroutine_kind| self.print_coroutine_kind(coroutine_kind));
                self.print_capture_clause(*capture_clause);

                self.print_fn_params_and_ret(fn_decl, true);
                self.space();
                self.print_expr(body, FixupContext::default());
                self.end(); // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            ast::ExprKind::Block(blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at }
                self.cbox(0);
                // head-box, will be closed by print-block after {
                self.ibox(0);
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Gen(capture_clause, blk, kind, _decl_span) => {
                self.word_nbsp(kind.modifier());
                self.print_capture_clause(*capture_clause);
                // cbox/ibox in analogy to the `ExprKind::Block` arm above
                self.cbox(0);
                self.ibox(0);
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Await(expr, _) => {
                self.print_expr_cond_paren(
                    expr,
                    expr.precedence() < ExprPrecedence::Unambiguous,
                    fixup,
                );
                self.word(".await");
            }
            ast::ExprKind::Assign(lhs, rhs, _) => {
                self.print_expr_cond_paren(
                    lhs,
                    // Ranges are allowed on the right-hand side of assignment,
                    // but not the left. `(a..b) = c` needs parentheses.
                    lhs.precedence() <= ExprPrecedence::Range,
                    fixup.leftmost_subexpression(),
                );
                self.space();
                self.word_space("=");
                self.print_expr_cond_paren(
                    rhs,
                    rhs.precedence() < ExprPrecedence::Assign,
                    fixup.subsequent_subexpression(),
                );
            }
            ast::ExprKind::AssignOp(op, lhs, rhs) => {
                self.print_expr_cond_paren(
                    lhs,
                    lhs.precedence() <= ExprPrecedence::Range,
                    fixup.leftmost_subexpression(),
                );
                self.space();
                self.word(op.node.as_str());
                self.word_space("=");
                self.print_expr_cond_paren(
                    rhs,
                    rhs.precedence() < ExprPrecedence::Assign,
                    fixup.subsequent_subexpression(),
                );
            }
            ast::ExprKind::Field(expr, ident) => {
                self.print_expr_cond_paren(
                    expr,
                    expr.precedence() < ExprPrecedence::Unambiguous,
                    fixup,
                );
                self.word(".");
                self.print_ident(*ident);
            }
            ast::ExprKind::Index(expr, index, _) => {
                self.print_expr_cond_paren(
                    expr,
                    expr.precedence() < ExprPrecedence::Unambiguous,
                    fixup.leftmost_subexpression(),
                );
                self.word("[");
                self.print_expr(index, FixupContext::default());
                self.word("]");
            }
            ast::ExprKind::Range(start, end, limits) => {
                // Special case for `Range`. `AssocOp` claims that `Range` has higher precedence
                // than `Assign`, but `x .. x = x` gives a parse error instead of `x .. (x = x)`.
                // Here we use a fake precedence value so that any child with lower precedence than
                // a "normal" binop gets parenthesized. (`LOr` is the lowest-precedence binop.)
                let fake_prec = ExprPrecedence::LOr;
                if let Some(e) = start {
                    self.print_expr_cond_paren(
                        e,
                        e.precedence() < fake_prec,
                        fixup.leftmost_subexpression(),
                    );
                }
                match limits {
                    ast::RangeLimits::HalfOpen => self.word(".."),
                    ast::RangeLimits::Closed => self.word("..="),
                }
                if let Some(e) = end {
                    self.print_expr_cond_paren(
                        e,
                        e.precedence() < fake_prec,
                        fixup.subsequent_subexpression(),
                    );
                }
            }
            ast::ExprKind::Underscore => self.word("_"),
            ast::ExprKind::Path(None, path) => self.print_path(path, true, 0),
            ast::ExprKind::Path(Some(qself), path) => self.print_qpath(path, qself, true),
            ast::ExprKind::Break(opt_label, opt_expr) => {
                self.word("break");
                if let Some(label) = opt_label {
                    self.space();
                    self.print_ident(label.ident);
                }
                if let Some(expr) = opt_expr {
                    self.space();
                    self.print_expr_cond_paren(
                        expr,
                        // Parenthesize if required by precedence, or in the
                        // case of `break 'inner: loop { break 'inner 1 } + 1`
                        expr.precedence() < ExprPrecedence::Jump
                            || (opt_label.is_none() && classify::leading_labeled_expr(expr)),
                        fixup.subsequent_subexpression(),
                    );
                }
            }
            ast::ExprKind::Continue(opt_label) => {
                self.word("continue");
                if let Some(label) = opt_label {
                    self.space();
                    self.print_ident(label.ident);
                }
            }
            ast::ExprKind::Ret(result) => {
                self.word("return");
                if let Some(expr) = result {
                    self.word(" ");
                    self.print_expr_cond_paren(
                        expr,
                        expr.precedence() < ExprPrecedence::Jump,
                        fixup.subsequent_subexpression(),
                    );
                }
            }
            ast::ExprKind::Yeet(result) => {
                self.word("do");
                self.word(" ");
                self.word("yeet");
                if let Some(expr) = result {
                    self.word(" ");
                    self.print_expr_cond_paren(
                        expr,
                        expr.precedence() < ExprPrecedence::Jump,
                        fixup.subsequent_subexpression(),
                    );
                }
            }
            ast::ExprKind::Become(result) => {
                self.word("become");
                self.word(" ");
                self.print_expr_cond_paren(
                    result,
                    result.precedence() < ExprPrecedence::Jump,
                    fixup.subsequent_subexpression(),
                );
            }
            ast::ExprKind::InlineAsm(a) => {
                // FIXME: Print `builtin # asm` once macro `asm` uses `builtin_syntax`.
                self.word("asm!");
                self.print_inline_asm(a);
            }
            ast::ExprKind::FormatArgs(fmt) => {
                // FIXME: Print `builtin # format_args` once macro `format_args` uses `builtin_syntax`.
                self.word("format_args!");
                self.popen();
                self.ibox(0);
                self.word(reconstruct_format_args_template_string(&fmt.template));
                for arg in fmt.arguments.all_args() {
                    self.word_space(",");
                    self.print_expr(&arg.expr, FixupContext::default());
                }
                self.end();
                self.pclose();
            }
            ast::ExprKind::OffsetOf(container, fields) => {
                self.word("builtin # offset_of");
                self.popen();
                self.ibox(0);
                self.print_type(container);
                self.word(",");
                self.space();

                if let Some((&first, rest)) = fields.split_first() {
                    self.print_ident(first);

                    for &field in rest {
                        self.word(".");
                        self.print_ident(field);
                    }
                }
                self.end();
                self.pclose();
            }
            ast::ExprKind::MacCall(m) => self.print_mac(m),
            ast::ExprKind::Paren(e) => {
                self.popen();
                self.print_expr(e, FixupContext::default());
                self.pclose();
            }
            ast::ExprKind::Yield(e) => {
                self.word("yield");

                if let Some(expr) = e {
                    self.space();
                    self.print_expr_cond_paren(
                        expr,
                        expr.precedence() < ExprPrecedence::Jump,
                        fixup.subsequent_subexpression(),
                    );
                }
            }
            ast::ExprKind::Try(e) => {
                self.print_expr_cond_paren(e, e.precedence() < ExprPrecedence::Unambiguous, fixup);
                self.word("?")
            }
            ast::ExprKind::TryBlock(blk) => {
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("try");
                self.print_block_with_attrs(blk, attrs)
            }
            ast::ExprKind::UnsafeBinderCast(kind, expr, ty) => {
                self.word("builtin # ");
                match kind {
                    ast::UnsafeBinderCastKind::Wrap => self.word("wrap_binder"),
                    ast::UnsafeBinderCastKind::Unwrap => self.word("unwrap_binder"),
                }
                self.popen();
                self.ibox(0);
                self.print_expr(expr, FixupContext::default());

                if let Some(ty) = ty {
                    self.word(",");
                    self.space();
                    self.print_type(ty);
                }

                self.end();
                self.pclose();
            }
            ast::ExprKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose()
            }
            ast::ExprKind::Dummy => {
                self.popen();
                self.word("/*DUMMY*/");
                self.pclose();
            }
        }

        self.ann.post(self, AnnNode::Expr(expr));

        if needs_par {
            self.pclose();
        }

        self.end();
    }

    fn print_arm(&mut self, arm: &ast::Arm) {
        // Note, I have no idea why this check is necessary, but here it is.
        if arm.attrs.is_empty() {
            self.space();
        }
        self.cbox(INDENT_UNIT);
        self.ibox(0);
        self.maybe_print_comment(arm.pat.span.lo());
        self.print_outer_attributes(&arm.attrs);
        self.print_pat(&arm.pat);
        self.space();
        if let Some(e) = &arm.guard {
            self.word_space("if");
            self.print_expr(e, FixupContext::default());
            self.space();
        }

        if let Some(body) = &arm.body {
            self.word_space("=>");

            match &body.kind {
                ast::ExprKind::Block(blk, opt_label) => {
                    if let Some(label) = opt_label {
                        self.print_ident(label.ident);
                        self.word_space(":");
                    }

                    // The block will close the pattern's ibox.
                    self.print_block_unclosed_indent(blk);

                    // If it is a user-provided unsafe block, print a comma after it.
                    if let BlockCheckMode::Unsafe(ast::UserProvided) = blk.rules {
                        self.word(",");
                    }
                }
                _ => {
                    self.end(); // Close the ibox for the pattern.
                    self.print_expr(body, FixupContext::new_match_arm());
                    self.word(",");
                }
            }
        } else {
            self.word(",");
        }
        self.end(); // Close enclosing cbox.
    }

    fn print_closure_binder(&mut self, binder: &ast::ClosureBinder) {
        match binder {
            ast::ClosureBinder::NotPresent => {}
            ast::ClosureBinder::For { generic_params, .. } => {
                self.print_formal_generic_params(generic_params)
            }
        }
    }

    fn print_movability(&mut self, movability: ast::Movability) {
        match movability {
            ast::Movability::Static => self.word_space("static"),
            ast::Movability::Movable => {}
        }
    }

    fn print_capture_clause(&mut self, capture_clause: ast::CaptureBy) {
        match capture_clause {
            ast::CaptureBy::Value { .. } => self.word_space("move"),
            ast::CaptureBy::Ref => {}
        }
    }
}

fn reconstruct_format_args_template_string(pieces: &[FormatArgsPiece]) -> String {
    let mut template = "\"".to_string();
    for piece in pieces {
        match piece {
            FormatArgsPiece::Literal(s) => {
                for c in s.as_str().chars() {
                    template.extend(c.escape_debug());
                    if let '{' | '}' = c {
                        template.push(c);
                    }
                }
            }
            FormatArgsPiece::Placeholder(p) => {
                template.push('{');
                let (Ok(n) | Err(n)) = p.argument.index;
                write!(template, "{n}").unwrap();
                if p.format_options != Default::default() || p.format_trait != FormatTrait::Display
                {
                    template.push(':');
                }
                if let Some(fill) = p.format_options.fill {
                    template.push(fill);
                }
                match p.format_options.alignment {
                    Some(FormatAlignment::Left) => template.push('<'),
                    Some(FormatAlignment::Right) => template.push('>'),
                    Some(FormatAlignment::Center) => template.push('^'),
                    None => {}
                }
                match p.format_options.sign {
                    Some(FormatSign::Plus) => template.push('+'),
                    Some(FormatSign::Minus) => template.push('-'),
                    None => {}
                }
                if p.format_options.alternate {
                    template.push('#');
                }
                if p.format_options.zero_pad {
                    template.push('0');
                }
                if let Some(width) = &p.format_options.width {
                    match width {
                        FormatCount::Literal(n) => write!(template, "{n}").unwrap(),
                        FormatCount::Argument(FormatArgPosition {
                            index: Ok(n) | Err(n), ..
                        }) => {
                            write!(template, "{n}$").unwrap();
                        }
                    }
                }
                if let Some(precision) = &p.format_options.precision {
                    template.push('.');
                    match precision {
                        FormatCount::Literal(n) => write!(template, "{n}").unwrap(),
                        FormatCount::Argument(FormatArgPosition {
                            index: Ok(n) | Err(n), ..
                        }) => {
                            write!(template, "{n}$").unwrap();
                        }
                    }
                }
                match p.format_options.debug_hex {
                    Some(FormatDebugHex::Lower) => template.push('x'),
                    Some(FormatDebugHex::Upper) => template.push('X'),
                    None => {}
                }
                template.push_str(match p.format_trait {
                    FormatTrait::Display => "",
                    FormatTrait::Debug => "?",
                    FormatTrait::LowerExp => "e",
                    FormatTrait::UpperExp => "E",
                    FormatTrait::Octal => "o",
                    FormatTrait::Pointer => "p",
                    FormatTrait::Binary => "b",
                    FormatTrait::LowerHex => "x",
                    FormatTrait::UpperHex => "X",
                });
                template.push('}');
            }
        }
    }
    template.push('"');
    template
}
