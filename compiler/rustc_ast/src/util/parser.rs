use rustc_span::kw;

use crate::ast::{self, BinOpKind};
use crate::token::{self, BinOpToken, Token};

/// Associative operator with precedence.
///
/// This is the enum which specifies operator precedence and fixity to the parser.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AssocOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `/`
    Divide,
    /// `%`
    Modulus,
    /// `&&`
    LAnd,
    /// `||`
    LOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `<<`
    ShiftLeft,
    /// `>>`
    ShiftRight,
    /// `==`
    Equal,
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `!=`
    NotEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
    /// `=`
    Assign,
    /// `?=` where ? is one of the BinOpToken
    AssignOp(BinOpToken),
    /// `as`
    As,
    /// `..` range
    DotDot,
    /// `..=` range
    DotDotEq,
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

impl AssocOp {
    /// Creates a new AssocOP from a token
    pub fn from_token(t: &Token) -> Option<AssocOp> {
        use AssocOp::*;
        match t.kind {
            token::BinOpEq(k) => Some(AssignOp(k)),
            token::Eq => Some(Assign),
            token::BinOp(BinOpToken::Star) => Some(Multiply),
            token::BinOp(BinOpToken::Slash) => Some(Divide),
            token::BinOp(BinOpToken::Percent) => Some(Modulus),
            token::BinOp(BinOpToken::Plus) => Some(Add),
            token::BinOp(BinOpToken::Minus) => Some(Subtract),
            token::BinOp(BinOpToken::Shl) => Some(ShiftLeft),
            token::BinOp(BinOpToken::Shr) => Some(ShiftRight),
            token::BinOp(BinOpToken::And) => Some(BitAnd),
            token::BinOp(BinOpToken::Caret) => Some(BitXor),
            token::BinOp(BinOpToken::Or) => Some(BitOr),
            token::Lt => Some(Less),
            token::Le => Some(LessEqual),
            token::Ge => Some(GreaterEqual),
            token::Gt => Some(Greater),
            token::EqEq => Some(Equal),
            token::Ne => Some(NotEqual),
            token::AndAnd => Some(LAnd),
            token::OrOr => Some(LOr),
            token::DotDot => Some(DotDot),
            token::DotDotEq => Some(DotDotEq),
            // DotDotDot is no longer supported, but we need some way to display the error
            token::DotDotDot => Some(DotDotEq),
            // `<-` should probably be `< -`
            token::LArrow => Some(Less),
            _ if t.is_keyword(kw::As) => Some(As),
            _ => None,
        }
    }

    /// Creates a new AssocOp from ast::BinOpKind.
    pub fn from_ast_binop(op: BinOpKind) -> Self {
        use AssocOp::*;
        match op {
            BinOpKind::Lt => Less,
            BinOpKind::Gt => Greater,
            BinOpKind::Le => LessEqual,
            BinOpKind::Ge => GreaterEqual,
            BinOpKind::Eq => Equal,
            BinOpKind::Ne => NotEqual,
            BinOpKind::Mul => Multiply,
            BinOpKind::Div => Divide,
            BinOpKind::Rem => Modulus,
            BinOpKind::Add => Add,
            BinOpKind::Sub => Subtract,
            BinOpKind::Shl => ShiftLeft,
            BinOpKind::Shr => ShiftRight,
            BinOpKind::BitAnd => BitAnd,
            BinOpKind::BitXor => BitXor,
            BinOpKind::BitOr => BitOr,
            BinOpKind::And => LAnd,
            BinOpKind::Or => LOr,
        }
    }

    /// Gets the precedence of this operator
    pub fn precedence(&self) -> ExprPrecedence {
        use AssocOp::*;
        match *self {
            As => ExprPrecedence::Cast,
            Multiply | Divide | Modulus => ExprPrecedence::Product,
            Add | Subtract => ExprPrecedence::Sum,
            ShiftLeft | ShiftRight => ExprPrecedence::Shift,
            BitAnd => ExprPrecedence::BitAnd,
            BitXor => ExprPrecedence::BitXor,
            BitOr => ExprPrecedence::BitOr,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => ExprPrecedence::Compare,
            LAnd => ExprPrecedence::LAnd,
            LOr => ExprPrecedence::LOr,
            DotDot | DotDotEq => ExprPrecedence::Range,
            Assign | AssignOp(_) => ExprPrecedence::Assign,
        }
    }

    /// Gets the fixity of this operator
    pub fn fixity(&self) -> Fixity {
        use AssocOp::*;
        // NOTE: it is a bug to have an operators that has same precedence but different fixities!
        match *self {
            Assign | AssignOp(_) => Fixity::Right,
            As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd
            | BitXor | BitOr | LAnd | LOr => Fixity::Left,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | DotDot | DotDotEq => {
                Fixity::None
            }
        }
    }

    pub fn is_comparison(&self) -> bool {
        use AssocOp::*;
        match *self {
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
            Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add | Subtract
            | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot | DotDotEq => {
                false
            }
        }
    }

    pub fn is_assign_like(&self) -> bool {
        use AssocOp::*;
        match *self {
            Assign | AssignOp(_) => true,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply
            | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor
            | BitOr | LAnd | LOr | DotDot | DotDotEq => false,
        }
    }

    pub fn to_ast_binop(&self) -> Option<BinOpKind> {
        use AssocOp::*;
        match *self {
            Less => Some(BinOpKind::Lt),
            Greater => Some(BinOpKind::Gt),
            LessEqual => Some(BinOpKind::Le),
            GreaterEqual => Some(BinOpKind::Ge),
            Equal => Some(BinOpKind::Eq),
            NotEqual => Some(BinOpKind::Ne),
            Multiply => Some(BinOpKind::Mul),
            Divide => Some(BinOpKind::Div),
            Modulus => Some(BinOpKind::Rem),
            Add => Some(BinOpKind::Add),
            Subtract => Some(BinOpKind::Sub),
            ShiftLeft => Some(BinOpKind::Shl),
            ShiftRight => Some(BinOpKind::Shr),
            BitAnd => Some(BinOpKind::BitAnd),
            BitXor => Some(BinOpKind::BitXor),
            BitOr => Some(BinOpKind::BitOr),
            LAnd => Some(BinOpKind::And),
            LOr => Some(BinOpKind::Or),
            Assign | AssignOp(_) | As | DotDot | DotDotEq => None,
        }
    }

    /// This operator could be used to follow a block unambiguously.
    ///
    /// This is used for error recovery at the moment, providing a suggestion to wrap blocks with
    /// parentheses while having a high degree of confidence on the correctness of the suggestion.
    pub fn can_continue_expr_unambiguously(&self) -> bool {
        use AssocOp::*;
        matches!(
            self,
            BitXor | // `{ 42 } ^ 3`
            Assign | // `{ 42 } = { 42 }`
            Divide | // `{ 42 } / 42`
            Modulus | // `{ 42 } % 2`
            ShiftRight | // `{ 42 } >> 2`
            LessEqual | // `{ 42 } <= 3`
            Greater | // `{ 42 } > 3`
            GreaterEqual | // `{ 42 } >= 3`
            AssignOp(_) | // `{ 42 } +=`
            // Equal | // `{ 42 } == { 42 }`    Accepting these here would regress incorrect
            // NotEqual | // `{ 42 } != { 42 }  struct literals parser recovery.
            As // `{ 42 } as usize`
        )
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum ExprPrecedence {
    // return, break, yield, closures
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
    // paths, loops, function calls, array indexing, field expressions, method calls
    Unambiguous,
}

/// In `let p = e`, operators with precedence `<=` this one requires parentheses in `e`.
pub fn prec_let_scrutinee_needs_par() -> ExprPrecedence {
    ExprPrecedence::LAnd
}

/// Suppose we have `let _ = e` and the `order` of `e`.
/// Is the `order` such that `e` in `let _ = e` needs parentheses when it is on the RHS?
///
/// Conversely, suppose that we have `(let _ = a) OP b` and `order` is that of `OP`.
/// Can we print this as `let _ = a OP b`?
pub fn needs_par_as_let_scrutinee(order: ExprPrecedence) -> bool {
    order <= prec_let_scrutinee_needs_par()
}

/// Expressions that syntactically contain an "exterior" struct literal i.e., not surrounded by any
/// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
pub fn contains_exterior_struct_lit(value: &ast::Expr) -> bool {
    match &value.kind {
        ast::ExprKind::Struct(..) => true,

        ast::ExprKind::Assign(lhs, rhs, _)
        | ast::ExprKind::AssignOp(_, lhs, rhs)
        | ast::ExprKind::Binary(_, lhs, rhs) => {
            // X { y: 1 } + X { y: 2 }
            contains_exterior_struct_lit(lhs) || contains_exterior_struct_lit(rhs)
        }
        ast::ExprKind::Await(x, _)
        | ast::ExprKind::Unary(_, x)
        | ast::ExprKind::Cast(x, _)
        | ast::ExprKind::Type(x, _)
        | ast::ExprKind::Field(x, _)
        | ast::ExprKind::Index(x, _, _)
        | ast::ExprKind::Match(x, _, ast::MatchKind::Postfix) => {
            // &X { y: 1 }, X { y: 1 }.y
            contains_exterior_struct_lit(x)
        }

        ast::ExprKind::MethodCall(box ast::MethodCall { receiver, .. }) => {
            // X { y: 1 }.bar(...)
            contains_exterior_struct_lit(receiver)
        }

        _ => false,
    }
}
