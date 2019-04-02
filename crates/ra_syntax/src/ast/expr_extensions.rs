use crate::{
    SyntaxToken, SyntaxElement, SmolStr,
    ast::{self, AstNode, AstChildren, children, child_opt},
    SyntaxKind::*
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElseBranch<'a> {
    Block(&'a ast::Block),
    IfExpr(&'a ast::IfExpr),
}

impl ast::IfExpr {
    pub fn then_branch(&self) -> Option<&ast::Block> {
        self.blocks().nth(0)
    }
    pub fn else_branch(&self) -> Option<ElseBranch> {
        let res = match self.blocks().nth(1) {
            Some(block) => ElseBranch::Block(block),
            None => {
                let elif: &ast::IfExpr = child_opt(self)?;
                ElseBranch::IfExpr(elif)
            }
        };
        Some(res)
    }

    fn blocks(&self) -> AstChildren<ast::Block> {
        children(self)
    }
}

impl ast::RefExpr {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrefixOp {
    /// The `*` operator for dereferencing
    Deref,
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl ast::PrefixExpr {
    pub fn op_kind(&self) -> Option<PrefixOp> {
        match self.op_token()?.kind() {
            STAR => Some(PrefixOp::Deref),
            EXCL => Some(PrefixOp::Not),
            MINUS => Some(PrefixOp::Neg),
            _ => None,
        }
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.syntax().first_child_or_token()?.as_token()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// The `||` operator for boolean OR
    BooleanOr,
    /// The `&&` operator for boolean AND
    BooleanAnd,
    /// The `==` operator for equality testing
    EqualityTest,
    /// The `!=` operator for equality testing
    NegatedEqualityTest,
    /// The `<=` operator for lesser-equal testing
    LesserEqualTest,
    /// The `>=` operator for greater-equal testing
    GreaterEqualTest,
    /// The `<` operator for comparison
    LesserTest,
    /// The `>` operator for comparison
    GreaterTest,
    /// The `+` operator for addition
    Addition,
    /// The `*` operator for multiplication
    Multiplication,
    /// The `-` operator for subtraction
    Subtraction,
    /// The `/` operator for division
    Division,
    /// The `%` operator for remainder after division
    Remainder,
    /// The `<<` operator for left shift
    LeftShift,
    /// The `>>` operator for right shift
    RightShift,
    /// The `^` operator for bitwise XOR
    BitwiseXor,
    /// The `|` operator for bitwise OR
    BitwiseOr,
    /// The `&` operator for bitwise AND
    BitwiseAnd,
    /// The `..` operator for right-open ranges
    RangeRightOpen,
    /// The `..=` operator for right-closed ranges
    RangeRightClosed,
    /// The `=` operator for assignment
    Assignment,
    /// The `+=` operator for assignment after addition
    AddAssign,
    /// The `/=` operator for assignment after division
    DivAssign,
    /// The `*=` operator for assignment after multiplication
    MulAssign,
    /// The `%=` operator for assignment after remainders
    RemAssign,
    /// The `>>=` operator for assignment after shifting right
    ShrAssign,
    /// The `<<=` operator for assignment after shifting left
    ShlAssign,
    /// The `-=` operator for assignment after subtraction
    SubAssign,
    /// The `|=` operator for assignment after bitwise OR
    BitOrAssign,
    /// The `&=` operator for assignment after bitwise AND
    BitAndAssign,
    /// The `^=` operator for assignment after bitwise XOR
    BitXorAssign,
}

impl ast::BinExpr {
    fn op_details(&self) -> Option<(SyntaxToken, BinOp)> {
        self.syntax().children_with_tokens().filter_map(|it| it.as_token()).find_map(|c| {
            match c.kind() {
                PIPEPIPE => Some((c, BinOp::BooleanOr)),
                AMPAMP => Some((c, BinOp::BooleanAnd)),
                EQEQ => Some((c, BinOp::EqualityTest)),
                NEQ => Some((c, BinOp::NegatedEqualityTest)),
                LTEQ => Some((c, BinOp::LesserEqualTest)),
                GTEQ => Some((c, BinOp::GreaterEqualTest)),
                L_ANGLE => Some((c, BinOp::LesserTest)),
                R_ANGLE => Some((c, BinOp::GreaterTest)),
                PLUS => Some((c, BinOp::Addition)),
                STAR => Some((c, BinOp::Multiplication)),
                MINUS => Some((c, BinOp::Subtraction)),
                SLASH => Some((c, BinOp::Division)),
                PERCENT => Some((c, BinOp::Remainder)),
                SHL => Some((c, BinOp::LeftShift)),
                SHR => Some((c, BinOp::RightShift)),
                CARET => Some((c, BinOp::BitwiseXor)),
                PIPE => Some((c, BinOp::BitwiseOr)),
                AMP => Some((c, BinOp::BitwiseAnd)),
                DOTDOT => Some((c, BinOp::RangeRightOpen)),
                DOTDOTEQ => Some((c, BinOp::RangeRightClosed)),
                EQ => Some((c, BinOp::Assignment)),
                PLUSEQ => Some((c, BinOp::AddAssign)),
                SLASHEQ => Some((c, BinOp::DivAssign)),
                STAREQ => Some((c, BinOp::MulAssign)),
                PERCENTEQ => Some((c, BinOp::RemAssign)),
                SHREQ => Some((c, BinOp::ShrAssign)),
                SHLEQ => Some((c, BinOp::ShlAssign)),
                MINUSEQ => Some((c, BinOp::SubAssign)),
                PIPEEQ => Some((c, BinOp::BitOrAssign)),
                AMPEQ => Some((c, BinOp::BitAndAssign)),
                CARETEQ => Some((c, BinOp::BitXorAssign)),
                _ => None,
            }
        })
    }

    pub fn op_kind(&self) -> Option<BinOp> {
        self.op_details().map(|t| t.1)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.op_details().map(|t| t.0)
    }

    pub fn lhs(&self) -> Option<&ast::Expr> {
        children(self).nth(0)
    }

    pub fn rhs(&self) -> Option<&ast::Expr> {
        children(self).nth(1)
    }

    pub fn sub_exprs(&self) -> (Option<&ast::Expr>, Option<&ast::Expr>) {
        let mut children = children(self);
        let first = children.next();
        let second = children.next();
        (first, second)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralKind {
    String,
    ByteString,
    Char,
    Byte,
    IntNumber { suffix: Option<SmolStr> },
    FloatNumber { suffix: Option<SmolStr> },
    Bool,
}

impl ast::Literal {
    pub fn token(&self) -> SyntaxToken {
        match self.syntax().first_child_or_token().unwrap() {
            SyntaxElement::Token(token) => token,
            _ => unreachable!(),
        }
    }

    pub fn kind(&self) -> LiteralKind {
        match self.token().kind() {
            INT_NUMBER => {
                let allowed_suffix_list = [
                    "isize", "i128", "i64", "i32", "i16", "i8", "usize", "u128", "u64", "u32",
                    "u16", "u8",
                ];
                let text = self.token().text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralKind::IntNumber { suffix }
            }
            FLOAT_NUMBER => {
                let allowed_suffix_list = ["f64", "f32"];
                let text = self.token().text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralKind::FloatNumber { suffix: suffix }
            }
            STRING | RAW_STRING => LiteralKind::String,
            TRUE_KW | FALSE_KW => LiteralKind::Bool,
            BYTE_STRING | RAW_BYTE_STRING => LiteralKind::ByteString,
            CHAR => LiteralKind::Char,
            BYTE => LiteralKind::Byte,
            _ => unreachable!(),
        }
    }
}

impl ast::NamedField {
    pub fn parent_struct_lit(&self) -> &ast::StructLit {
        self.syntax().ancestors().find_map(ast::StructLit::cast).unwrap()
    }
}
