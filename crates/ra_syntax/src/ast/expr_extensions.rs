//! Various extension methods to ast Expr Nodes, which are hard to code-generate.

use crate::{
    ast::{self, child_opt, children, AstChildren, AstNode},
    SmolStr,
    SyntaxKind::*,
    SyntaxToken, T,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElseBranch {
    Block(ast::Block),
    IfExpr(ast::IfExpr),
}

impl ast::IfExpr {
    pub fn then_branch(&self) -> Option<ast::Block> {
        self.blocks().nth(0)
    }
    pub fn else_branch(&self) -> Option<ElseBranch> {
        let res = match self.blocks().nth(1) {
            Some(block) => ElseBranch::Block(block),
            None => {
                let elif: ast::IfExpr = child_opt(self)?;
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
        self.syntax().children_with_tokens().any(|n| n.kind() == T![mut])
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
            T![*] => Some(PrefixOp::Deref),
            T![!] => Some(PrefixOp::Not),
            T![-] => Some(PrefixOp::Neg),
            _ => None,
        }
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.syntax().first_child_or_token()?.into_token()
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
        self.syntax().children_with_tokens().filter_map(|it| it.into_token()).find_map(|c| match c
            .kind()
        {
            T![||] => Some((c, BinOp::BooleanOr)),
            T![&&] => Some((c, BinOp::BooleanAnd)),
            T![==] => Some((c, BinOp::EqualityTest)),
            T![!=] => Some((c, BinOp::NegatedEqualityTest)),
            T![<=] => Some((c, BinOp::LesserEqualTest)),
            T![>=] => Some((c, BinOp::GreaterEqualTest)),
            T![<] => Some((c, BinOp::LesserTest)),
            T![>] => Some((c, BinOp::GreaterTest)),
            T![+] => Some((c, BinOp::Addition)),
            T![*] => Some((c, BinOp::Multiplication)),
            T![-] => Some((c, BinOp::Subtraction)),
            T![/] => Some((c, BinOp::Division)),
            T![%] => Some((c, BinOp::Remainder)),
            T![<<] => Some((c, BinOp::LeftShift)),
            T![>>] => Some((c, BinOp::RightShift)),
            T![^] => Some((c, BinOp::BitwiseXor)),
            T![|] => Some((c, BinOp::BitwiseOr)),
            T![&] => Some((c, BinOp::BitwiseAnd)),
            T![..] => Some((c, BinOp::RangeRightOpen)),
            T![..=] => Some((c, BinOp::RangeRightClosed)),
            T![=] => Some((c, BinOp::Assignment)),
            T![+=] => Some((c, BinOp::AddAssign)),
            T![/=] => Some((c, BinOp::DivAssign)),
            T![*=] => Some((c, BinOp::MulAssign)),
            T![%=] => Some((c, BinOp::RemAssign)),
            T![>>=] => Some((c, BinOp::ShrAssign)),
            T![<<=] => Some((c, BinOp::ShlAssign)),
            T![-=] => Some((c, BinOp::SubAssign)),
            T![|=] => Some((c, BinOp::BitOrAssign)),
            T![&=] => Some((c, BinOp::BitAndAssign)),
            T![^=] => Some((c, BinOp::BitXorAssign)),
            _ => None,
        })
    }

    pub fn op_kind(&self) -> Option<BinOp> {
        self.op_details().map(|t| t.1)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.op_details().map(|t| t.0)
    }

    pub fn lhs(&self) -> Option<ast::Expr> {
        children(self).nth(0)
    }

    pub fn rhs(&self) -> Option<ast::Expr> {
        children(self).nth(1)
    }

    pub fn sub_exprs(&self) -> (Option<ast::Expr>, Option<ast::Expr>) {
        let mut children = children(self);
        let first = children.next();
        let second = children.next();
        (first, second)
    }
}

pub enum ArrayExprKind {
    Repeat { initializer: Option<ast::Expr>, repeat: Option<ast::Expr> },
    ElementList(AstChildren<ast::Expr>),
}

impl ast::ArrayExpr {
    pub fn kind(&self) -> ArrayExprKind {
        if self.is_repeat() {
            ArrayExprKind::Repeat {
                initializer: children(self).nth(0),
                repeat: children(self).nth(1),
            }
        } else {
            ArrayExprKind::ElementList(children(self))
        }
    }

    fn is_repeat(&self) -> bool {
        self.syntax().children_with_tokens().any(|it| it.kind() == T![;])
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
        self.syntax()
            .children_with_tokens()
            .find(|e| e.kind() != ATTR && !e.kind().is_trivia())
            .and_then(|e| e.into_token())
            .unwrap()
    }

    pub fn kind(&self) -> LiteralKind {
        match self.token().kind() {
            INT_NUMBER => {
                let int_suffix_list = [
                    "isize", "i128", "i64", "i32", "i16", "i8", "usize", "u128", "u64", "u32",
                    "u16", "u8",
                ];

                // The lexer treats e.g. `1f64` as an integer literal. See
                // https://github.com/rust-analyzer/rust-analyzer/issues/1592
                // and the comments on the linked PR.
                let float_suffix_list = ["f32", "f64"];

                let text = self.token().text().to_string();

                let float_suffix = float_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));

                if float_suffix.is_some() {
                    LiteralKind::FloatNumber { suffix: float_suffix }
                } else {
                    let suffix = int_suffix_list
                        .iter()
                        .find(|&s| text.ends_with(s))
                        .map(|&suf| SmolStr::new(suf));
                    LiteralKind::IntNumber { suffix }
                }
            }
            FLOAT_NUMBER => {
                let allowed_suffix_list = ["f64", "f32"];
                let text = self.token().text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralKind::FloatNumber { suffix }
            }
            STRING | RAW_STRING => LiteralKind::String,
            T![true] | T![false] => LiteralKind::Bool,
            BYTE_STRING | RAW_BYTE_STRING => LiteralKind::ByteString,
            CHAR => LiteralKind::Char,
            BYTE => LiteralKind::Byte,
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_literal_with_attr() {
    let parse = ast::SourceFile::parse(r#"const _: &str = { #[attr] "Hello" };"#);
    let lit = parse.tree().syntax().descendants().find_map(ast::Literal::cast).unwrap();
    assert_eq!(lit.token().text(), r#""Hello""#);
}

impl ast::NamedField {
    pub fn parent_struct_lit(&self) -> ast::StructLit {
        self.syntax().ancestors().find_map(ast::StructLit::cast).unwrap()
    }
}
