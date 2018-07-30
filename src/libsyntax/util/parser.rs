// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use parse::token::{Token, BinOpToken};
use symbol::keywords;
use ast::{self, BinOpKind};

/// Associative operator with precedence.
///
/// This is the enum which specifies operator precedence and fixity to the parser.
#[derive(PartialEq, Debug)]
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
    /// `<-`
    ObsoleteInPlace,
    /// `?=` where ? is one of the BinOpToken
    AssignOp(BinOpToken),
    /// `as`
    As,
    /// `..` range
    DotDot,
    /// `..=` range
    DotDotEq,
    /// `:`
    Colon,
}

#[derive(PartialEq, Debug)]
pub enum Fixity {
    /// The operator is left-associative
    Left,
    /// The operator is right-associative
    Right,
    /// The operator is not associative
    None
}

impl AssocOp {
    /// Create a new AssocOP from a token
    pub fn from_token(t: &Token) -> Option<AssocOp> {
        use self::AssocOp::*;
        match *t {
            Token::BinOpEq(k) => Some(AssignOp(k)),
            Token::LArrow => Some(ObsoleteInPlace),
            Token::Eq => Some(Assign),
            Token::BinOp(BinOpToken::Star) => Some(Multiply),
            Token::BinOp(BinOpToken::Slash) => Some(Divide),
            Token::BinOp(BinOpToken::Percent) => Some(Modulus),
            Token::BinOp(BinOpToken::Plus) => Some(Add),
            Token::BinOp(BinOpToken::Minus) => Some(Subtract),
            Token::BinOp(BinOpToken::Shl) => Some(ShiftLeft),
            Token::BinOp(BinOpToken::Shr) => Some(ShiftRight),
            Token::BinOp(BinOpToken::And) => Some(BitAnd),
            Token::BinOp(BinOpToken::Caret) => Some(BitXor),
            Token::BinOp(BinOpToken::Or) => Some(BitOr),
            Token::Lt => Some(Less),
            Token::Le => Some(LessEqual),
            Token::Ge => Some(GreaterEqual),
            Token::Gt => Some(Greater),
            Token::EqEq => Some(Equal),
            Token::Ne => Some(NotEqual),
            Token::AndAnd => Some(LAnd),
            Token::OrOr => Some(LOr),
            Token::DotDot => Some(DotDot),
            Token::DotDotEq => Some(DotDotEq),
            // DotDotDot is no longer supported, but we need some way to display the error
            Token::DotDotDot => Some(DotDotEq),
            Token::Colon => Some(Colon),
            _ if t.is_keyword(keywords::As) => Some(As),
            _ => None
        }
    }

    /// Create a new AssocOp from ast::BinOpKind.
    pub fn from_ast_binop(op: BinOpKind) -> Self {
        use self::AssocOp::*;
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
            BinOpKind::Or => LOr
        }
    }

    /// Gets the precedence of this operator
    pub fn precedence(&self) -> usize {
        use self::AssocOp::*;
        match *self {
            As | Colon => 14,
            Multiply | Divide | Modulus => 13,
            Add | Subtract => 12,
            ShiftLeft | ShiftRight => 11,
            BitAnd => 10,
            BitXor => 9,
            BitOr => 8,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => 7,
            LAnd => 6,
            LOr => 5,
            DotDot | DotDotEq => 4,
            ObsoleteInPlace => 3,
            Assign | AssignOp(_) => 2,
        }
    }

    /// Gets the fixity of this operator
    pub fn fixity(&self) -> Fixity {
        use self::AssocOp::*;
        // NOTE: it is a bug to have an operators that has same precedence but different fixities!
        match *self {
            ObsoleteInPlace | Assign | AssignOp(_) => Fixity::Right,
            As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd |
            BitXor | BitOr | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual |
            LAnd | LOr | Colon => Fixity::Left,
            DotDot | DotDotEq => Fixity::None
        }
    }

    pub fn is_comparison(&self) -> bool {
        use self::AssocOp::*;
        match *self {
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
            ObsoleteInPlace | Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add |
            Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr |
            DotDot | DotDotEq | Colon => false
        }
    }

    pub fn is_assign_like(&self) -> bool {
        use self::AssocOp::*;
        match *self {
            Assign | AssignOp(_) | ObsoleteInPlace => true,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply | Divide |
            Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd |
            LOr | DotDot | DotDotEq | Colon => false
        }
    }

    pub fn to_ast_binop(&self) -> Option<BinOpKind> {
        use self::AssocOp::*;
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
            ObsoleteInPlace | Assign | AssignOp(_) | As | DotDot | DotDotEq | Colon => None
        }
    }
}

pub const PREC_RESET: i8 = -100;
pub const PREC_CLOSURE: i8 = -40;
pub const PREC_JUMP: i8 = -30;
pub const PREC_RANGE: i8 = -10;
// The range 2 ... 14 is reserved for AssocOp binary operator precedences.
pub const PREC_PREFIX: i8 = 50;
pub const PREC_POSTFIX: i8 = 60;
pub const PREC_PAREN: i8 = 99;
pub const PREC_FORCE_PAREN: i8 = 100;

#[derive(Debug, Clone, Copy)]
pub enum ExprPrecedence {
    Closure,
    Break,
    Continue,
    Ret,
    Yield,

    Range,

    Binary(BinOpKind),

    ObsoleteInPlace,
    Cast,
    Type,

    Assign,
    AssignOp,

    Box,
    AddrOf,
    Unary,

    Call,
    MethodCall,
    Field,
    Index,
    Try,
    InlineAsm,
    Mac,

    Array,
    Repeat,
    Tup,
    Lit,
    Path,
    Paren,
    If,
    IfLet,
    While,
    WhileLet,
    ForLoop,
    Loop,
    Match,
    Block,
    Catch,
    Struct,
    Async,
}

impl ExprPrecedence {
    pub fn order(self) -> i8 {
        match self {
            ExprPrecedence::Closure => PREC_CLOSURE,

            ExprPrecedence::Break |
            ExprPrecedence::Continue |
            ExprPrecedence::Ret |
            ExprPrecedence::Yield => PREC_JUMP,

            // `Range` claims to have higher precedence than `Assign`, but `x .. x = x` fails to
            // parse, instead of parsing as `(x .. x) = x`.  Giving `Range` a lower precedence
            // ensures that `pprust` will add parentheses in the right places to get the desired
            // parse.
            ExprPrecedence::Range => PREC_RANGE,

            // Binop-like expr kinds, handled by `AssocOp`.
            ExprPrecedence::Binary(op) => AssocOp::from_ast_binop(op).precedence() as i8,
            ExprPrecedence::ObsoleteInPlace => AssocOp::ObsoleteInPlace.precedence() as i8,
            ExprPrecedence::Cast => AssocOp::As.precedence() as i8,
            ExprPrecedence::Type => AssocOp::Colon.precedence() as i8,

            ExprPrecedence::Assign |
            ExprPrecedence::AssignOp => AssocOp::Assign.precedence() as i8,

            // Unary, prefix
            ExprPrecedence::Box |
            ExprPrecedence::AddrOf |
            ExprPrecedence::Unary => PREC_PREFIX,

            // Unary, postfix
            ExprPrecedence::Call |
            ExprPrecedence::MethodCall |
            ExprPrecedence::Field |
            ExprPrecedence::Index |
            ExprPrecedence::Try |
            ExprPrecedence::InlineAsm |
            ExprPrecedence::Mac => PREC_POSTFIX,

            // Never need parens
            ExprPrecedence::Array |
            ExprPrecedence::Repeat |
            ExprPrecedence::Tup |
            ExprPrecedence::Lit |
            ExprPrecedence::Path |
            ExprPrecedence::Paren |
            ExprPrecedence::If |
            ExprPrecedence::IfLet |
            ExprPrecedence::While |
            ExprPrecedence::WhileLet |
            ExprPrecedence::ForLoop |
            ExprPrecedence::Loop |
            ExprPrecedence::Match |
            ExprPrecedence::Block |
            ExprPrecedence::Catch |
            ExprPrecedence::Async |
            ExprPrecedence::Struct => PREC_PAREN,
        }
    }
}


/// Expressions that syntactically contain an "exterior" struct literal i.e. not surrounded by any
/// parens or other delimiters, e.g. `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
pub fn contains_exterior_struct_lit(value: &ast::Expr) -> bool {
    match value.node {
        ast::ExprKind::Struct(..) => true,

        ast::ExprKind::Assign(ref lhs, ref rhs) |
        ast::ExprKind::AssignOp(_, ref lhs, ref rhs) |
        ast::ExprKind::Binary(_, ref lhs, ref rhs) => {
            // X { y: 1 } + X { y: 2 }
            contains_exterior_struct_lit(&lhs) || contains_exterior_struct_lit(&rhs)
        }
        ast::ExprKind::Unary(_, ref x) |
        ast::ExprKind::Cast(ref x, _) |
        ast::ExprKind::Type(ref x, _) |
        ast::ExprKind::Field(ref x, _) |
        ast::ExprKind::Index(ref x, _) => {
            // &X { y: 1 }, X { y: 1 }.y
            contains_exterior_struct_lit(&x)
        }

        ast::ExprKind::MethodCall(.., ref exprs) => {
            // X { y: 1 }.bar(...)
            contains_exterior_struct_lit(&exprs[0])
        }

        _ => false,
    }
}
