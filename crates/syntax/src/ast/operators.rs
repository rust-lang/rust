#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RangeOp {
    /// `..`
    Exclusive,
    /// `..=`
    Inclusive,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// The `*` operator for dereferencing
    Deref,
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
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

use crate::ast;
impl From<ast::BinOp> for BinaryOp {
    fn from(ast_op: ast::BinOp) -> Self {
        match ast_op {
            ast::BinOp::BooleanOr => BinaryOp::LogicOp(LogicOp::Or),
            ast::BinOp::BooleanAnd => BinaryOp::LogicOp(LogicOp::And),
            ast::BinOp::EqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
            ast::BinOp::NegatedEqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: true }),
            ast::BinOp::LesserEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: false })
            }
            ast::BinOp::GreaterEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: false })
            }
            ast::BinOp::LesserTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: true })
            }
            ast::BinOp::GreaterTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: true })
            }
            ast::BinOp::Addition => BinaryOp::ArithOp(ArithOp::Add),
            ast::BinOp::Multiplication => BinaryOp::ArithOp(ArithOp::Mul),
            ast::BinOp::Subtraction => BinaryOp::ArithOp(ArithOp::Sub),
            ast::BinOp::Division => BinaryOp::ArithOp(ArithOp::Div),
            ast::BinOp::Remainder => BinaryOp::ArithOp(ArithOp::Rem),
            ast::BinOp::LeftShift => BinaryOp::ArithOp(ArithOp::Shl),
            ast::BinOp::RightShift => BinaryOp::ArithOp(ArithOp::Shr),
            ast::BinOp::BitwiseXor => BinaryOp::ArithOp(ArithOp::BitXor),
            ast::BinOp::BitwiseOr => BinaryOp::ArithOp(ArithOp::BitOr),
            ast::BinOp::BitwiseAnd => BinaryOp::ArithOp(ArithOp::BitAnd),
            ast::BinOp::Assignment => BinaryOp::Assignment { op: None },
            ast::BinOp::AddAssign => BinaryOp::Assignment { op: Some(ArithOp::Add) },
            ast::BinOp::DivAssign => BinaryOp::Assignment { op: Some(ArithOp::Div) },
            ast::BinOp::MulAssign => BinaryOp::Assignment { op: Some(ArithOp::Mul) },
            ast::BinOp::RemAssign => BinaryOp::Assignment { op: Some(ArithOp::Rem) },
            ast::BinOp::ShlAssign => BinaryOp::Assignment { op: Some(ArithOp::Shl) },
            ast::BinOp::ShrAssign => BinaryOp::Assignment { op: Some(ArithOp::Shr) },
            ast::BinOp::SubAssign => BinaryOp::Assignment { op: Some(ArithOp::Sub) },
            ast::BinOp::BitOrAssign => BinaryOp::Assignment { op: Some(ArithOp::BitOr) },
            ast::BinOp::BitAndAssign => BinaryOp::Assignment { op: Some(ArithOp::BitAnd) },
            ast::BinOp::BitXorAssign => BinaryOp::Assignment { op: Some(ArithOp::BitXor) },
        }
    }
}
