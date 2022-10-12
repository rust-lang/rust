//! Defines a bunch of data-less enums for unary and binary operators.
//!
//! Types here don't know about AST, this allows re-using them for both AST and
//! HIR.
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RangeOp {
    /// `..`
    Exclusive,
    /// `..=`
    Inclusive,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `*`
    Deref,
    /// `!`
    Not,
    /// `-`
    Neg,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    LogicOp(LogicOp),
    ArithOp(ArithOp),
    CmpOp(CmpOp),
    Assignment { op: Option<ArithOp> },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq { negated: bool },
    Ord { ordering: Ordering, strict: bool },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    Less,
    Greater,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArithOp {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
    Shl,
    Shr,
    BitXor,
    BitOr,
    BitAnd,
}

impl fmt::Display for LogicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let res = match self {
            LogicOp::And => "&&",
            LogicOp::Or => "||",
        };
        f.write_str(res)
    }
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let res = match self {
            ArithOp::Add => "+",
            ArithOp::Mul => "*",
            ArithOp::Sub => "-",
            ArithOp::Div => "/",
            ArithOp::Rem => "%",
            ArithOp::Shl => "<<",
            ArithOp::Shr => ">>",
            ArithOp::BitXor => "^",
            ArithOp::BitOr => "|",
            ArithOp::BitAnd => "&",
        };
        f.write_str(res)
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let res = match self {
            CmpOp::Eq { negated: false } => "==",
            CmpOp::Eq { negated: true } => "!=",
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => "<=",
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => "<",
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => ">=",
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => ">",
        };
        f.write_str(res)
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::LogicOp(op) => fmt::Display::fmt(op, f),
            BinaryOp::ArithOp(op) => fmt::Display::fmt(op, f),
            BinaryOp::CmpOp(op) => fmt::Display::fmt(op, f),
            BinaryOp::Assignment { op } => {
                if let Some(op) = op {
                    fmt::Display::fmt(op, f)?;
                }
                f.write_str("=")?;
                Ok(())
            }
        }
    }
}
