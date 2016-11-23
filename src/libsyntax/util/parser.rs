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
use ast::BinOpKind;

/// Associative operator with precedence.
///
/// This is the enum which specifies operator precedence and fixity to the parser.
#[derive(Debug, PartialEq, Eq)]
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
    Inplace,
    /// `?=` where ? is one of the BinOpToken
    AssignOp(BinOpToken),
    /// `as`
    As,
    /// `..` range
    DotDot,
    /// `...` range
    DotDotDot,
    /// `:`
    Colon,
}

#[derive(Debug, PartialEq, Eq)]
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
            Token::LArrow => Some(Inplace),
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
            Token::DotDotDot => Some(DotDotDot),
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
            DotDot | DotDotDot => 4,
            Inplace => 3,
            Assign | AssignOp(_) => 2,
        }
    }

    /// Gets the fixity of this operator
    pub fn fixity(&self) -> Fixity {
        use self::AssocOp::*;
        // NOTE: it is a bug to have an operators that has same precedence but different fixities!
        match *self {
            Inplace | Assign | AssignOp(_) => Fixity::Right,
            As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd |
            BitXor | BitOr | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual |
            LAnd | LOr | Colon => Fixity::Left,
            DotDot | DotDotDot => Fixity::None
        }
    }

    pub fn is_comparison(&self) -> bool {
        use self::AssocOp::*;
        match *self {
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
            Inplace | Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add | Subtract |
            ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr |
            DotDot | DotDotDot | Colon => false
        }
    }

    pub fn is_assign_like(&self) -> bool {
        use self::AssocOp::*;
        match *self {
            Assign | AssignOp(_) | Inplace => true,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply | Divide |
            Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd |
            LOr | DotDot | DotDotDot | Colon => false
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
            Inplace | Assign | AssignOp(_) | As | DotDot | DotDotDot | Colon => None
        }
    }
}
