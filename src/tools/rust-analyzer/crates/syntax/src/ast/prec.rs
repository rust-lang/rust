//! Precedence representation.

use stdx::always;

use crate::{
    AstNode, SyntaxNode,
    ast::{self, BinaryOp, Expr, HasArgList, RangeItem},
    match_ast,
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum ExprPrecedence {
    // return val, break val, yield val, closures
    Jump,
    // = += -= *= /= %= &= |= ^= <<= >>=
    Assign,
    // .. ..=
    Range,
    // ||
    LOr,
    // &&
    LAnd,
    // == != < > <= >=
    Compare,
    // |
    BitOr,
    // ^
    BitXor,
    // &
    BitAnd,
    // << >>
    Shift,
    // + -
    Sum,
    // * / %
    Product,
    // as
    Cast,
    // unary - * ! & &mut
    Prefix,
    // function calls, array indexing, field expressions, method calls
    Postfix,
    // paths, loops,
    Unambiguous,
}

impl ExprPrecedence {
    pub fn needs_parentheses_in(self, other: ExprPrecedence) -> bool {
        match other {
            ExprPrecedence::Unambiguous => false,
            // postfix ops have higher precedence than any other operator, so we need to wrap
            // any inner expression that is below
            ExprPrecedence::Postfix => self < ExprPrecedence::Postfix,
            // We need to wrap all binary like things, thats everything below prefix except for
            // jumps (as those are prefix operations as well)
            ExprPrecedence::Prefix => ExprPrecedence::Jump < self && self < ExprPrecedence::Prefix,
            parent => self <= parent,
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum Fixity {
    /// The operator is left-associative
    Left,
    /// The operator is right-associative
    Right,
    /// The operator is not associative
    None,
}

pub fn precedence(expr: &ast::Expr) -> ExprPrecedence {
    match expr {
        Expr::ClosureExpr(closure) => match closure.ret_type() {
            None => ExprPrecedence::Jump,
            Some(_) => ExprPrecedence::Unambiguous,
        },

        Expr::BreakExpr(e) if e.expr().is_some() => ExprPrecedence::Jump,
        Expr::BecomeExpr(e) if e.expr().is_some() => ExprPrecedence::Jump,
        Expr::ReturnExpr(e) if e.expr().is_some() => ExprPrecedence::Jump,
        Expr::YeetExpr(e) if e.expr().is_some() => ExprPrecedence::Jump,
        Expr::YieldExpr(e) if e.expr().is_some() => ExprPrecedence::Jump,

        Expr::BreakExpr(_)
        | Expr::BecomeExpr(_)
        | Expr::ReturnExpr(_)
        | Expr::YeetExpr(_)
        | Expr::YieldExpr(_)
        | Expr::ContinueExpr(_) => ExprPrecedence::Unambiguous,

        Expr::RangeExpr(..) => ExprPrecedence::Range,

        Expr::BinExpr(bin_expr) => match bin_expr.op_kind() {
            Some(it) => match it {
                BinaryOp::LogicOp(logic_op) => match logic_op {
                    ast::LogicOp::And => ExprPrecedence::LAnd,
                    ast::LogicOp::Or => ExprPrecedence::LOr,
                },
                BinaryOp::ArithOp(arith_op) => match arith_op {
                    ast::ArithOp::Add | ast::ArithOp::Sub => ExprPrecedence::Sum,
                    ast::ArithOp::Div | ast::ArithOp::Rem | ast::ArithOp::Mul => {
                        ExprPrecedence::Product
                    }
                    ast::ArithOp::Shl | ast::ArithOp::Shr => ExprPrecedence::Shift,
                    ast::ArithOp::BitXor => ExprPrecedence::BitXor,
                    ast::ArithOp::BitOr => ExprPrecedence::BitOr,
                    ast::ArithOp::BitAnd => ExprPrecedence::BitAnd,
                },
                BinaryOp::CmpOp(_) => ExprPrecedence::Compare,
                BinaryOp::Assignment { .. } => ExprPrecedence::Assign,
            },
            None => ExprPrecedence::Unambiguous,
        },
        Expr::CastExpr(_) => ExprPrecedence::Cast,

        Expr::LetExpr(_) | Expr::PrefixExpr(_) | Expr::RefExpr(_) => ExprPrecedence::Prefix,

        Expr::AwaitExpr(_)
        | Expr::CallExpr(_)
        | Expr::FieldExpr(_)
        | Expr::IndexExpr(_)
        | Expr::MethodCallExpr(_)
        | Expr::TryExpr(_) => ExprPrecedence::Postfix,

        Expr::ArrayExpr(_)
        | Expr::AsmExpr(_)
        | Expr::BlockExpr(_)
        | Expr::ForExpr(_)
        | Expr::FormatArgsExpr(_)
        | Expr::IfExpr(_)
        | Expr::Literal(_)
        | Expr::LoopExpr(_)
        | Expr::MacroExpr(_)
        | Expr::MatchExpr(_)
        | Expr::OffsetOfExpr(_)
        | Expr::ParenExpr(_)
        | Expr::PathExpr(_)
        | Expr::RecordExpr(_)
        | Expr::TupleExpr(_)
        | Expr::UnderscoreExpr(_)
        | Expr::WhileExpr(_) => ExprPrecedence::Unambiguous,
    }
}

fn check_ancestry(ancestor: &SyntaxNode, descendent: &SyntaxNode) -> bool {
    let bail = || always!(false, "{} is not an ancestor of {}", ancestor, descendent);

    if !ancestor.text_range().contains_range(descendent.text_range()) {
        return bail();
    }

    for anc in descendent.ancestors() {
        if anc == *ancestor {
            return true;
        }
    }

    bail()
}

impl Expr {
    pub fn precedence(&self) -> ExprPrecedence {
        precedence(self)
    }

    // Implementation is based on
    // - https://doc.rust-lang.org/reference/expressions.html#expression-precedence
    // - https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
    // - rustc source, including, but not limited to
    //   - https://github.com/rust-lang/rust/blob/b6852428a8ea9728369b64b9964cad8e258403d3/compiler/rustc_ast/src/util/parser.rs#L296

    /// Returns `true` if `self` would need to be wrapped in parentheses given that its parent is `parent`.
    pub fn needs_parens_in(&self, parent: &SyntaxNode) -> bool {
        self.needs_parens_in_place_of(parent, self.syntax())
    }

    /// Returns `true` if `self` would need to be wrapped in parentheses if it replaces `place_of`
    /// given that `place_of`'s parent is `parent`.
    pub fn needs_parens_in_place_of(&self, parent: &SyntaxNode, place_of: &SyntaxNode) -> bool {
        if !check_ancestry(parent, place_of) {
            return false;
        }

        match_ast! {
            match parent {
                ast::Expr(e) => self.needs_parens_in_expr(&e, place_of),
                ast::Stmt(e) => self.needs_parens_in_stmt(Some(&e)),
                ast::StmtList(_) => self.needs_parens_in_stmt(None),
                ast::ArgList(_) => false,
                ast::MatchArm(_) => false,
                _ => false,
            }
        }
    }

    fn needs_parens_in_expr(&self, parent: &Expr, place_of: &SyntaxNode) -> bool {
        // Parentheses are necessary when calling a function-like pointer that is a member of a struct or union
        // (e.g. `(a.f)()`).
        let is_parent_call_expr = matches!(parent, ast::Expr::CallExpr(_));
        let is_field_expr = matches!(self, ast::Expr::FieldExpr(_));
        if is_parent_call_expr && is_field_expr {
            return true;
        }

        // Special-case block weirdness
        if parent.child_is_followed_by_a_block() {
            use Expr::*;
            match self {
                // Cases like `if return {}` (need parens or else `{}` is returned, instead of being `if`'s body)
                ReturnExpr(e) if e.expr().is_none() => return true,
                BreakExpr(e) if e.expr().is_none() => return true,
                YieldExpr(e) if e.expr().is_none() => return true,

                // Same but with `..{}`
                RangeExpr(e) if matches!(e.end(), Some(BlockExpr(..))) => return true,

                // Similarly with struct literals, e.g. `if S{} == 1 {}`
                _ if self.contains_exterior_struct_lit() => return true,
                _ => {}
            }
        }

        // Special-case `return.f()`
        if self.is_ret_like_with_no_value() && parent.is_postfix() {
            return false;
        }

        if self.is_paren_like()
            || parent.is_paren_like()
            || self.is_prefix()
                && (parent.is_prefix()
                    || !self.is_ordered_before_parent_in_place_of(parent, place_of))
            || self.is_postfix()
                && (parent.is_postfix()
                    || self.is_ordered_before_parent_in_place_of(parent, place_of))
        {
            return false;
        }

        let (left, right, inv) = match self.is_ordered_before_parent_in_place_of(parent, place_of) {
            true => (self, parent, false),
            false => (parent, self, true),
        };

        let (_, left_right_bp) = left.binding_power();
        let (right_left_bp, _) = right.binding_power();

        (left_right_bp < right_left_bp) ^ inv
    }

    fn needs_parens_in_stmt(&self, stmt: Option<&ast::Stmt>) -> bool {
        use Expr::*;

        // Prevent false-positives in cases like `fn x() -> u8 { ({ 0 } + 1) }`,
        // `{ { 0 } + 1 }` won't parse -- `{ 0 }` would be parsed as a self-contained stmt,
        // leaving `+ 1` as a parse error.
        let mut innermost = self.clone();
        loop {
            let next = match &innermost {
                BinExpr(e) => e.lhs(),
                CallExpr(e) => e.expr(),
                CastExpr(e) => e.expr(),
                IndexExpr(e) => e.base(),
                _ => break,
            };

            if let Some(next) = next {
                innermost = next;
                if !innermost.requires_semi_to_be_stmt() {
                    return true;
                }
            } else {
                break;
            }
        }

        // Not every expression can be followed by `else` in the `let-else`
        if let Some(ast::Stmt::LetStmt(e)) = stmt
            && e.let_else().is_some()
        {
            match self {
                BinExpr(e)
                    if e.op_kind()
                        .map(|op| matches!(op, BinaryOp::LogicOp(_)))
                        .unwrap_or(false) =>
                {
                    return true;
                }
                _ if self.clone().trailing_brace().is_some() => return true,
                _ => {}
            }
        }

        false
    }

    /// Returns left and right so-called "binding powers" of this expression.
    fn binding_power(&self) -> (u8, u8) {
        use ast::{ArithOp::*, BinaryOp::*, Expr::*, LogicOp::*};

        match self {
            // (0, 0)   -- paren-like/nullary
            // (0, N)   -- prefix
            // (N, 0)   -- postfix
            // (N, N)   -- infix, requires parens
            // (N, N+1) -- infix, left to right associative
            // (N+1, N) -- infix, right to left associative
            // N is odd
            //
            ContinueExpr(_) => (0, 0),

            ClosureExpr(_) | ReturnExpr(_) | BecomeExpr(_) | YieldExpr(_) | YeetExpr(_)
            | BreakExpr(_) | OffsetOfExpr(_) | FormatArgsExpr(_) | AsmExpr(_) => (0, 1),

            RangeExpr(_) => (5, 5),

            BinExpr(e) => {
                // Return a dummy value if we don't know the op
                let Some(op) = e.op_kind() else { return (0, 0) };
                match op {
                    Assignment { .. } => (4, 3),
                    //
                    // Ranges are here in order :)
                    //
                    LogicOp(op) => match op {
                        Or => (7, 8),
                        And => (9, 10),
                    },
                    CmpOp(_) => (11, 11),
                    ArithOp(op) => match op {
                        BitOr => (13, 14),
                        BitXor => (15, 16),
                        BitAnd => (17, 18),
                        Shl | Shr => (19, 20),
                        Add | Sub => (21, 22),
                        Mul | Div | Rem => (23, 24),
                    },
                }
            }

            CastExpr(_) => (25, 26),

            RefExpr(_) | LetExpr(_) | PrefixExpr(_) => (0, 27),

            AwaitExpr(_) | CallExpr(_) | MethodCallExpr(_) | IndexExpr(_) | TryExpr(_)
            | MacroExpr(_) => (29, 0),

            FieldExpr(_) => (31, 32),

            ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_) | IfExpr(_)
            | WhileExpr(_) | ForExpr(_) | LoopExpr(_) | MatchExpr(_) | BlockExpr(_)
            | RecordExpr(_) | UnderscoreExpr(_) => (0, 0),
        }
    }

    fn is_paren_like(&self) -> bool {
        matches!(self.binding_power(), (0, 0))
    }

    fn is_prefix(&self) -> bool {
        matches!(self.binding_power(), (0, 1..))
    }

    fn is_postfix(&self) -> bool {
        matches!(self.binding_power(), (1.., 0))
    }

    /// Returns `true` if this expression can't be a standalone statement.
    fn requires_semi_to_be_stmt(&self) -> bool {
        use Expr::*;
        !matches!(
            self,
            IfExpr(..) | MatchExpr(..) | BlockExpr(..) | WhileExpr(..) | LoopExpr(..) | ForExpr(..)
        )
    }

    /// If an expression ends with `}`, returns the innermost expression ending in this `}`.
    fn trailing_brace(mut self) -> Option<Expr> {
        use Expr::*;

        loop {
            let rhs = match self {
                RefExpr(e) => e.expr(),
                BinExpr(e) => e.rhs(),
                BreakExpr(e) => e.expr(),
                LetExpr(e) => e.expr(),
                RangeExpr(e) => e.end(),
                ReturnExpr(e) => e.expr(),
                PrefixExpr(e) => e.expr(),
                YieldExpr(e) => e.expr(),
                ClosureExpr(e) => e.body(),

                BlockExpr(..) | ForExpr(..) | IfExpr(..) | LoopExpr(..) | MatchExpr(..)
                | RecordExpr(..) | WhileExpr(..) => break Some(self),
                _ => break None,
            };

            self = rhs?;
        }
    }

    /// Expressions that syntactically contain an "exterior" struct literal i.e., not surrounded by any
    /// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
    /// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
    fn contains_exterior_struct_lit(&self) -> bool {
        return contains_exterior_struct_lit_inner(self).is_some();

        fn contains_exterior_struct_lit_inner(expr: &Expr) -> Option<()> {
            use Expr::*;

            match expr {
                RecordExpr(..) => Some(()),

                // X { y: 1 } + X { y: 2 }
                BinExpr(e) => e
                    .lhs()
                    .as_ref()
                    .and_then(contains_exterior_struct_lit_inner)
                    .or_else(|| e.rhs().as_ref().and_then(contains_exterior_struct_lit_inner)),

                // `&X { y: 1 }`, `X { y: 1 }.y`, `X { y: 1 }.bar(...)`, etc
                IndexExpr(e) => contains_exterior_struct_lit_inner(&e.base()?),
                AwaitExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                PrefixExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                CastExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                FieldExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                MethodCallExpr(e) => contains_exterior_struct_lit_inner(&e.receiver()?),

                _ => None,
            }
        }
    }

    /// Returns true if self is one of `return`, `break`, `continue` or `yield` with **no associated value**.
    pub fn is_ret_like_with_no_value(&self) -> bool {
        use Expr::*;

        match self {
            ReturnExpr(e) => e.expr().is_none(),
            BreakExpr(e) => e.expr().is_none(),
            ContinueExpr(_) => true,
            YieldExpr(e) => e.expr().is_none(),
            BecomeExpr(e) => e.expr().is_none(),
            _ => false,
        }
    }

    fn is_ordered_before_parent_in_place_of(&self, parent: &Expr, place_of: &SyntaxNode) -> bool {
        use Expr::*;
        use rowan::TextSize;

        let self_range = self.syntax().text_range();
        let place_of_range = place_of.text_range();

        let self_order_adjusted = order(self) - self_range.start() + place_of_range.start();

        let parent_order = order(parent);
        let parent_order_adjusted = if parent_order <= place_of_range.start() {
            parent_order
        } else if parent_order >= place_of_range.end() {
            parent_order - place_of_range.len() + self_range.len()
        } else {
            return false;
        };

        return self_order_adjusted < parent_order_adjusted;

        /// Returns text range that can be used to compare two expression for order (which goes first).
        fn order(this: &Expr) -> TextSize {
            // For non-paren-like operators: get the operator itself
            let token = match this {
                RangeExpr(e) => e.op_token(),
                BinExpr(e) => e.op_token(),
                CastExpr(e) => e.as_token(),
                FieldExpr(e) => e.dot_token(),
                AwaitExpr(e) => e.dot_token(),
                BreakExpr(e) => e.break_token(),
                CallExpr(e) => e.arg_list().and_then(|args| args.l_paren_token()),
                ClosureExpr(e) => e.param_list().and_then(|params| params.l_paren_token()),
                ContinueExpr(e) => e.continue_token(),
                IndexExpr(e) => e.l_brack_token(),
                MethodCallExpr(e) => e.dot_token(),
                PrefixExpr(e) => e.op_token(),
                RefExpr(e) => e.amp_token(),
                ReturnExpr(e) => e.return_token(),
                BecomeExpr(e) => e.become_token(),
                TryExpr(e) => e.question_mark_token(),
                YieldExpr(e) => e.yield_token(),
                YeetExpr(e) => e.do_token(),
                LetExpr(e) => e.let_token(),
                OffsetOfExpr(e) => e.builtin_token(),
                FormatArgsExpr(e) => e.builtin_token(),
                AsmExpr(e) => e.builtin_token(),
                ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_)
                | IfExpr(_) | WhileExpr(_) | ForExpr(_) | LoopExpr(_) | MatchExpr(_)
                | BlockExpr(_) | RecordExpr(_) | UnderscoreExpr(_) | MacroExpr(_) => None,
            };

            token.map(|t| t.text_range()).unwrap_or_else(|| this.syntax().text_range()).start()
        }
    }

    fn child_is_followed_by_a_block(&self) -> bool {
        use Expr::*;

        match self {
            ArrayExpr(_) | AwaitExpr(_) | BlockExpr(_) | CallExpr(_) | CastExpr(_)
            | ClosureExpr(_) | FieldExpr(_) | IndexExpr(_) | Literal(_) | LoopExpr(_)
            | MacroExpr(_) | MethodCallExpr(_) | ParenExpr(_) | PathExpr(_) | RecordExpr(_)
            | TryExpr(_) | TupleExpr(_) | UnderscoreExpr(_) | OffsetOfExpr(_)
            | FormatArgsExpr(_) | AsmExpr(_) => false,

            // For BinExpr and RangeExpr this is technically wrong -- the child can be on the left...
            BinExpr(_) | RangeExpr(_) | BreakExpr(_) | ContinueExpr(_) | PrefixExpr(_)
            | RefExpr(_) | ReturnExpr(_) | BecomeExpr(_) | YieldExpr(_) | YeetExpr(_)
            | LetExpr(_) => self
                .syntax()
                .parent()
                .and_then(Expr::cast)
                .map(|e| e.child_is_followed_by_a_block())
                .unwrap_or(false),

            ForExpr(_) | IfExpr(_) | MatchExpr(_) | WhileExpr(_) => true,
        }
    }
}
