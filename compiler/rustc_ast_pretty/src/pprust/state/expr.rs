use crate::pp::Breaks::Inconsistent;
use crate::pprust::state::{AnnNode, IterDelimited, PrintState, State, INDENT_UNIT};

use rustc_ast::ptr::P;
use rustc_ast::util::parser::{self, AssocOp, Fixity};
use rustc_ast::{self as ast, BlockCheckMode};

impl<'a> State<'a> {
    fn print_else(&mut self, els: Option<&ast::Expr>) {
        if let Some(_else) = els {
            match _else.kind {
                // Another `else if` block.
                ast::ExprKind::If(ref i, ref then, ref e) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.word(" else if ");
                    self.print_expr_as_cond(i);
                    self.space();
                    self.print_block(then);
                    self.print_else(e.as_deref())
                }
                // Final `else` block.
                ast::ExprKind::Block(ref b, _) => {
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

    fn print_expr_maybe_paren(&mut self, expr: &ast::Expr, prec: i8) {
        self.print_expr_cond_paren(expr, expr.precedence().order() < prec)
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    fn print_expr_as_cond(&mut self, expr: &ast::Expr) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    // Does `expr` need parentheses when printed in a condition position?
    //
    // These cases need parens due to the parse error observed in #26461: `if return {}`
    // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
    pub(super) fn cond_needs_par(expr: &ast::Expr) -> bool {
        match expr.kind {
            ast::ExprKind::Break(..) | ast::ExprKind::Closure(..) | ast::ExprKind::Ret(..) => true,
            _ => parser::contains_exterior_struct_lit(expr),
        }
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    pub(super) fn print_expr_cond_paren(&mut self, expr: &ast::Expr, needs_par: bool) {
        if needs_par {
            self.popen();
        }
        self.print_expr(expr);
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

    pub(super) fn print_expr_anon_const(&mut self, expr: &ast::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.word("const");
        self.print_expr(&expr.value);
        self.end();
    }

    fn print_expr_repeat(&mut self, element: &ast::Expr, count: &ast::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.word("[");
        self.print_expr(element);
        self.word_space(";");
        self.print_expr(&count.value);
        self.word("]");
        self.end();
    }

    fn print_expr_struct(
        &mut self,
        qself: &Option<ast::QSelf>,
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
        for field in fields.iter().delimited() {
            self.maybe_print_comment(field.span.hi());
            self.print_outer_attributes(&field.attrs);
            if field.is_first {
                self.space_if_not_bol();
            }
            if !field.is_shorthand {
                self.print_ident(field.ident);
                self.word_nbsp(":");
            }
            self.print_expr(&field.expr);
            if !field.is_last || has_rest {
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
                self.print_expr(expr);
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

    fn print_expr_call(&mut self, func: &ast::Expr, args: &[P<ast::Expr>]) {
        let prec = match func.kind {
            ast::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
            _ => parser::PREC_POSTFIX,
        };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self, segment: &ast::PathSegment, args: &[P<ast::Expr>]) {
        let base_args = &args[1..];
        self.print_expr_maybe_paren(&args[0], parser::PREC_POSTFIX);
        self.word(".");
        self.print_ident(segment.ident);
        if let Some(ref args) = segment.args {
            self.print_generic_args(args, true);
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self, op: ast::BinOp, lhs: &ast::Expr, rhs: &ast::Expr) {
        let assoc_op = AssocOp::from_ast_binop(op.node);
        let prec = assoc_op.precedence() as i8;
        let fixity = assoc_op.fixity();

        let (left_prec, right_prec) = match fixity {
            Fixity::Left => (prec, prec + 1),
            Fixity::Right => (prec + 1, prec),
            Fixity::None => (prec + 1, prec + 1),
        };

        let left_prec = match (&lhs.kind, op.node) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&ast::ExprKind::Cast { .. }, ast::BinOpKind::Lt | ast::BinOpKind::Shl) => {
                parser::PREC_FORCE_PAREN
            }
            // We are given `(let _ = a) OP b`.
            //
            // - When `OP <= LAnd` we should print `let _ = a OP b` to avoid redundant parens
            //   as the parser will interpret this as `(let _ = a) OP b`.
            //
            // - Otherwise, e.g. when we have `(let a = b) < c` in AST,
            //   parens are required since the parser would interpret `let a = b < c` as
            //   `let a = (b < c)`. To achieve this, we force parens.
            (&ast::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(prec) => {
                parser::PREC_FORCE_PAREN
            }
            _ => left_prec,
        };

        self.print_expr_maybe_paren(lhs, left_prec);
        self.space();
        self.word_space(op.node.to_string());
        self.print_expr_maybe_paren(rhs, right_prec)
    }

    fn print_expr_unary(&mut self, op: ast::UnOp, expr: &ast::Expr) {
        self.word(ast::UnOp::to_string(op));
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(
        &mut self,
        kind: ast::BorrowKind,
        mutability: ast::Mutability,
        expr: &ast::Expr,
    ) {
        self.word("&");
        match kind {
            ast::BorrowKind::Ref => self.print_mutability(mutability, false),
            ast::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    pub fn print_expr(&mut self, expr: &ast::Expr) {
        self.print_expr_outer_attr_style(expr, true)
    }

    pub(super) fn print_expr_outer_attr_style(&mut self, expr: &ast::Expr, is_inline: bool) {
        self.maybe_print_comment(expr.span.lo());

        let attrs = &expr.attrs;
        if is_inline {
            self.print_outer_attributes_inline(attrs);
        } else {
            self.print_outer_attributes(attrs);
        }

        self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.kind {
            ast::ExprKind::Box(ref expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            ast::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(exprs);
            }
            ast::ExprKind::ConstBlock(ref anon_const) => {
                self.print_expr_anon_const(anon_const);
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(element, count);
            }
            ast::ExprKind::Struct(ref se) => {
                self.print_expr_struct(&se.qself, &se.path, &se.fields, &se.rest);
            }
            ast::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(exprs);
            }
            ast::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(func, &args);
            }
            ast::ExprKind::MethodCall(ref segment, ref args, _) => {
                self.print_expr_method_call(segment, &args);
            }
            ast::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, lhs, rhs);
            }
            ast::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, expr);
            }
            ast::ExprKind::AddrOf(k, m, ref expr) => {
                self.print_expr_addr_of(k, m, expr);
            }
            ast::ExprKind::Lit(ref lit) => {
                self.print_literal(lit);
            }
            ast::ExprKind::Cast(ref expr, ref ty) => {
                let prec = AssocOp::As.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.space();
                self.word_space("as");
                self.print_type(ty);
            }
            ast::ExprKind::Type(ref expr, ref ty) => {
                let prec = AssocOp::Colon.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.word_space(":");
                self.print_type(ty);
            }
            ast::ExprKind::Let(ref pat, ref scrutinee, _) => {
                self.print_let(pat, scrutinee);
            }
            ast::ExprKind::If(ref test, ref blk, ref elseopt) => {
                self.print_if(test, blk, elseopt.as_deref())
            }
            ast::ExprKind::While(ref test, ref blk, opt_label) => {
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
            ast::ExprKind::ForLoop(ref pat, ref iter, ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("for");
                self.print_pat(pat);
                self.space();
                self.word_space("in");
                self.print_expr_as_cond(iter);
                self.space();
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Loop(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("loop");
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Match(ref expr, ref arms) => {
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("match");
                self.print_expr_as_cond(expr);
                self.space();
                self.bopen();
                self.print_inner_attributes_no_trailing_hardbreak(attrs);
                for arm in arms {
                    self.print_arm(arm);
                }
                let empty = attrs.is_empty() && arms.is_empty();
                self.bclose(expr.span, empty);
            }
            ast::ExprKind::Closure(
                capture_clause,
                asyncness,
                movability,
                ref decl,
                ref body,
                _,
            ) => {
                self.print_movability(movability);
                self.print_asyncness(asyncness);
                self.print_capture_clause(capture_clause);

                self.print_fn_params_and_ret(decl, true);
                self.space();
                self.print_expr(body);
                self.end(); // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            ast::ExprKind::Block(ref blk, opt_label) => {
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
            ast::ExprKind::Async(capture_clause, _, ref blk) => {
                self.word_nbsp("async");
                self.print_capture_clause(capture_clause);
                // cbox/ibox in analogy to the `ExprKind::Block` arm above
                self.cbox(0);
                self.ibox(0);
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Await(ref expr) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.word(".await");
            }
            ast::ExprKind::Assign(ref lhs, ref rhs, _) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.space();
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            ast::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.space();
                self.word(op.node.to_string());
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            ast::ExprKind::Field(ref expr, ident) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.word(".");
                self.print_ident(ident);
            }
            ast::ExprKind::Index(ref expr, ref index) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.word("[");
                self.print_expr(index);
                self.word("]");
            }
            ast::ExprKind::Range(ref start, ref end, limits) => {
                // Special case for `Range`.  `AssocOp` claims that `Range` has higher precedence
                // than `Assign`, but `x .. x = x` gives a parse error instead of `x .. (x = x)`.
                // Here we use a fake precedence value so that any child with lower precedence than
                // a "normal" binop gets parenthesized.  (`LOr` is the lowest-precedence binop.)
                let fake_prec = AssocOp::LOr.precedence() as i8;
                if let Some(ref e) = *start {
                    self.print_expr_maybe_paren(e, fake_prec);
                }
                if limits == ast::RangeLimits::HalfOpen {
                    self.word("..");
                } else {
                    self.word("..=");
                }
                if let Some(ref e) = *end {
                    self.print_expr_maybe_paren(e, fake_prec);
                }
            }
            ast::ExprKind::Underscore => self.word("_"),
            ast::ExprKind::Path(None, ref path) => self.print_path(path, true, 0),
            ast::ExprKind::Path(Some(ref qself), ref path) => self.print_qpath(path, qself, true),
            ast::ExprKind::Break(opt_label, ref opt_expr) => {
                self.word("break");
                if let Some(label) = opt_label {
                    self.space();
                    self.print_ident(label.ident);
                }
                if let Some(ref expr) = *opt_expr {
                    self.space();
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            ast::ExprKind::Continue(opt_label) => {
                self.word("continue");
                if let Some(label) = opt_label {
                    self.space();
                    self.print_ident(label.ident);
                }
            }
            ast::ExprKind::Ret(ref result) => {
                self.word("return");
                if let Some(ref expr) = *result {
                    self.word(" ");
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            ast::ExprKind::InlineAsm(ref a) => {
                self.word("asm!");
                self.print_inline_asm(a);
            }
            ast::ExprKind::MacCall(ref m) => self.print_mac(m),
            ast::ExprKind::Paren(ref e) => {
                self.popen();
                self.print_expr(e);
                self.pclose();
            }
            ast::ExprKind::Yield(ref e) => {
                self.word("yield");

                if let Some(ref expr) = *e {
                    self.space();
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            ast::ExprKind::Try(ref e) => {
                self.print_expr_maybe_paren(e, parser::PREC_POSTFIX);
                self.word("?")
            }
            ast::ExprKind::TryBlock(ref blk) => {
                self.cbox(0);
                self.ibox(0);
                self.word_nbsp("try");
                self.print_block_with_attrs(blk, attrs)
            }
            ast::ExprKind::Err => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose()
            }
        }
        self.ann.post(self, AnnNode::Expr(expr));
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
        if let Some(ref e) = arm.guard {
            self.word_space("if");
            self.print_expr(e);
            self.space();
        }
        self.word_space("=>");

        match arm.body.kind {
            ast::ExprKind::Block(ref blk, opt_label) => {
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
                self.print_expr(&arm.body);
                self.word(",");
            }
        }
        self.end(); // Close enclosing cbox.
    }

    fn print_movability(&mut self, movability: ast::Movability) {
        match movability {
            ast::Movability::Static => self.word_space("static"),
            ast::Movability::Movable => {}
        }
    }

    fn print_capture_clause(&mut self, capture_clause: ast::CaptureBy) {
        match capture_clause {
            ast::CaptureBy::Value => self.word_space("move"),
            ast::CaptureBy::Ref => {}
        }
    }
}
