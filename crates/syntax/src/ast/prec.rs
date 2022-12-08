//! Precedence representation.

use crate::ast::{self, BinExpr, Expr};

/// Precedence of an expression.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ExprPrecedence {
    // N.B.: Order is important
    Closure,
    Jump,
    Range,
    Bin(BinOpPresedence),
    Prefix,
    Postfix,
    Paren,
}

/// Precedence of a binary operator.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum BinOpPresedence {
    // N.B.: Order is important
    /// `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `|=`, `&=`
    Assign,
    /// `||`
    LOr,
    /// `&&`
    LAnd,
    /// `<`, `<=`, `>`, `>=`, `==` and `!=`
    Cmp,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
    /// `<<` and `>>`
    Shift,
    /// `+` and `-`
    Add,
    /// `*`, `/` and `%`
    Mul,
    /// `as`
    As,
}

impl Expr {
    /// Returns precedence of this expression.
    /// Usefull to preserve semantics in assists.
    ///
    /// Returns `None` if this is a [`BinExpr`] and its [`op_kind`] returns `None`.
    ///
    /// [`op_kind`]: BinExpr::op_kind
    /// [`BinExpr`]: Expr::BinExpr
    pub fn precedence(&self) -> Option<ExprPrecedence> {
        // Copied from <https://github.com/rust-lang/rust/blob/b6852428a8ea9728369b64b9964cad8e258403d3/compiler/rustc_ast/src/util/parser.rs#L296>
        use Expr::*;

        let prec = match self {
            ClosureExpr(_) => ExprPrecedence::Closure,

            ContinueExpr(_) | ReturnExpr(_) | YieldExpr(_) | BreakExpr(_) => ExprPrecedence::Jump,

            RangeExpr(_) => ExprPrecedence::Range,

            BinExpr(bin_expr) => return bin_expr.precedence().map(ExprPrecedence::Bin),
            CastExpr(_) => ExprPrecedence::Bin(BinOpPresedence::As),

            BoxExpr(_) | RefExpr(_) | LetExpr(_) | PrefixExpr(_) => ExprPrecedence::Prefix,

            AwaitExpr(_) | CallExpr(_) | MethodCallExpr(_) | FieldExpr(_) | IndexExpr(_)
            | TryExpr(_) | MacroExpr(_) => ExprPrecedence::Postfix,

            ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_) | IfExpr(_)
            | WhileExpr(_) | ForExpr(_) | LoopExpr(_) | MatchExpr(_) | BlockExpr(_)
            | RecordExpr(_) | UnderscoreExpr(_) => ExprPrecedence::Paren,
        };

        Some(prec)
    }
}

impl BinExpr {
    /// Returns precedence of this binary expression.
    /// Usefull to preserve semantics in assists.
    ///
    /// Returns `None` if [`op_kind`] returns `None`.
    ///
    /// [`op_kind`]: BinExpr::op_kind
    pub fn precedence(&self) -> Option<BinOpPresedence> {
        use ast::{ArithOp::*, BinaryOp::*, LogicOp::*};

        let prec = match self.op_kind()? {
            LogicOp(op) => match op {
                And => BinOpPresedence::LAnd,
                Or => BinOpPresedence::LOr,
            },
            ArithOp(op) => match op {
                Add => BinOpPresedence::Add,
                Mul => BinOpPresedence::Mul,
                Sub => BinOpPresedence::Add,
                Div => BinOpPresedence::Mul,
                Rem => BinOpPresedence::Mul,
                Shl => BinOpPresedence::Shift,
                Shr => BinOpPresedence::Shift,
                BitXor => BinOpPresedence::BitXor,
                BitOr => BinOpPresedence::BitOr,
                BitAnd => BinOpPresedence::BitAnd,
            },
            CmpOp(_) => BinOpPresedence::Cmp,
            Assignment { .. } => BinOpPresedence::Assign,
        };

        Some(prec)
    }
}
