//! Various extension methods to ast Expr Nodes, which are hard to code-generate.
//!
//! These methods should only do simple, shallow tasks related to the syntax of the node itself.

use crate::{
    ast::{
        self,
        operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
        support, AstChildren, AstNode,
    },
    AstToken,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T,
};

impl ast::HasAttrs for ast::Expr {}

impl ast::Expr {
    pub fn is_block_like(&self) -> bool {
        matches!(
            self,
            ast::Expr::IfExpr(_)
                | ast::Expr::LoopExpr(_)
                | ast::Expr::ForExpr(_)
                | ast::Expr::WhileExpr(_)
                | ast::Expr::BlockExpr(_)
                | ast::Expr::MatchExpr(_)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElseBranch {
    Block(ast::BlockExpr),
    IfExpr(ast::IfExpr),
}

impl From<ast::BlockExpr> for ElseBranch {
    fn from(block_expr: ast::BlockExpr) -> Self {
        Self::Block(block_expr)
    }
}

impl From<ast::IfExpr> for ElseBranch {
    fn from(if_expr: ast::IfExpr) -> Self {
        Self::IfExpr(if_expr)
    }
}

impl ast::IfExpr {
    pub fn condition(&self) -> Option<ast::Expr> {
        // If the condition is a BlockExpr, check if the then body is missing.
        // If it is assume the condition is the expression that is missing instead.
        let mut exprs = support::children(self.syntax());
        let first = exprs.next();
        match first {
            Some(ast::Expr::BlockExpr(_)) => exprs.next().and(first),
            first => first,
        }
    }

    pub fn then_branch(&self) -> Option<ast::BlockExpr> {
        match support::children(self.syntax()).nth(1)? {
            ast::Expr::BlockExpr(block) => Some(block),
            _ => None,
        }
    }

    pub fn else_branch(&self) -> Option<ElseBranch> {
        match support::children(self.syntax()).nth(2)? {
            ast::Expr::BlockExpr(block) => Some(ElseBranch::Block(block)),
            ast::Expr::IfExpr(elif) => Some(ElseBranch::IfExpr(elif)),
            _ => None,
        }
    }
}

#[test]
fn if_block_condition() {
    let parse = ast::SourceFile::parse(
        r#"
        fn test() {
            if { true } { "if" }
            else if { false } { "first elif" }
            else if true { "second elif" }
            else if (true) { "third elif" }
            else { "else" }
        }
        "#,
    );
    let if_ = parse.tree().syntax().descendants().find_map(ast::IfExpr::cast).unwrap();
    assert_eq!(if_.then_branch().unwrap().syntax().text(), r#"{ "if" }"#);
    let elif = match if_.else_branch().unwrap() {
        ElseBranch::IfExpr(elif) => elif,
        ElseBranch::Block(_) => panic!("should be `else if`"),
    };
    assert_eq!(elif.then_branch().unwrap().syntax().text(), r#"{ "first elif" }"#);
    let elif = match elif.else_branch().unwrap() {
        ElseBranch::IfExpr(elif) => elif,
        ElseBranch::Block(_) => panic!("should be `else if`"),
    };
    assert_eq!(elif.then_branch().unwrap().syntax().text(), r#"{ "second elif" }"#);
    let elif = match elif.else_branch().unwrap() {
        ElseBranch::IfExpr(elif) => elif,
        ElseBranch::Block(_) => panic!("should be `else if`"),
    };
    assert_eq!(elif.then_branch().unwrap().syntax().text(), r#"{ "third elif" }"#);
    let else_ = match elif.else_branch().unwrap() {
        ElseBranch::Block(else_) => else_,
        ElseBranch::IfExpr(_) => panic!("should be `else`"),
    };
    assert_eq!(else_.syntax().text(), r#"{ "else" }"#);
}

#[test]
fn if_condition_with_if_inside() {
    let parse = ast::SourceFile::parse(
        r#"
        fn test() {
            if if true { true } else { false } { "if" }
            else { "else" }
        }
        "#,
    );
    let if_ = parse.tree().syntax().descendants().find_map(ast::IfExpr::cast).unwrap();
    assert_eq!(if_.then_branch().unwrap().syntax().text(), r#"{ "if" }"#);
    let else_ = match if_.else_branch().unwrap() {
        ElseBranch::Block(else_) => else_,
        ElseBranch::IfExpr(_) => panic!("should be `else`"),
    };
    assert_eq!(else_.syntax().text(), r#"{ "else" }"#);
}

impl ast::PrefixExpr {
    pub fn op_kind(&self) -> Option<UnaryOp> {
        let res = match self.op_token()?.kind() {
            T![*] => UnaryOp::Deref,
            T![!] => UnaryOp::Not,
            T![-] => UnaryOp::Neg,
            _ => return None,
        };
        Some(res)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.syntax().first_child_or_token()?.into_token()
    }
}

impl ast::BinExpr {
    pub fn op_details(&self) -> Option<(SyntaxToken, BinaryOp)> {
        self.syntax().children_with_tokens().filter_map(|it| it.into_token()).find_map(|c| {
            #[rustfmt::skip]
            let bin_op = match c.kind() {
                T![||] => BinaryOp::LogicOp(LogicOp::Or),
                T![&&] => BinaryOp::LogicOp(LogicOp::And),

                T![==] => BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
                T![!=] => BinaryOp::CmpOp(CmpOp::Eq { negated: true }),
                T![<=] => BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less,    strict: false }),
                T![>=] => BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: false }),
                T![<]  => BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less,    strict: true }),
                T![>]  => BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: true }),

                T![+]  => BinaryOp::ArithOp(ArithOp::Add),
                T![*]  => BinaryOp::ArithOp(ArithOp::Mul),
                T![-]  => BinaryOp::ArithOp(ArithOp::Sub),
                T![/]  => BinaryOp::ArithOp(ArithOp::Div),
                T![%]  => BinaryOp::ArithOp(ArithOp::Rem),
                T![<<] => BinaryOp::ArithOp(ArithOp::Shl),
                T![>>] => BinaryOp::ArithOp(ArithOp::Shr),
                T![^]  => BinaryOp::ArithOp(ArithOp::BitXor),
                T![|]  => BinaryOp::ArithOp(ArithOp::BitOr),
                T![&]  => BinaryOp::ArithOp(ArithOp::BitAnd),

                T![=]   => BinaryOp::Assignment { op: None },
                T![+=]  => BinaryOp::Assignment { op: Some(ArithOp::Add) },
                T![*=]  => BinaryOp::Assignment { op: Some(ArithOp::Mul) },
                T![-=]  => BinaryOp::Assignment { op: Some(ArithOp::Sub) },
                T![/=]  => BinaryOp::Assignment { op: Some(ArithOp::Div) },
                T![%=]  => BinaryOp::Assignment { op: Some(ArithOp::Rem) },
                T![<<=] => BinaryOp::Assignment { op: Some(ArithOp::Shl) },
                T![>>=] => BinaryOp::Assignment { op: Some(ArithOp::Shr) },
                T![^=]  => BinaryOp::Assignment { op: Some(ArithOp::BitXor) },
                T![|=]  => BinaryOp::Assignment { op: Some(ArithOp::BitOr) },
                T![&=]  => BinaryOp::Assignment { op: Some(ArithOp::BitAnd) },

                _ => return None,
            };
            Some((c, bin_op))
        })
    }

    pub fn op_kind(&self) -> Option<BinaryOp> {
        self.op_details().map(|t| t.1)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.op_details().map(|t| t.0)
    }

    pub fn lhs(&self) -> Option<ast::Expr> {
        support::children(self.syntax()).next()
    }

    pub fn rhs(&self) -> Option<ast::Expr> {
        support::children(self.syntax()).nth(1)
    }

    pub fn sub_exprs(&self) -> (Option<ast::Expr>, Option<ast::Expr>) {
        let mut children = support::children(self.syntax());
        let first = children.next();
        let second = children.next();
        (first, second)
    }
}

impl ast::RangeExpr {
    fn op_details(&self) -> Option<(usize, SyntaxToken, RangeOp)> {
        self.syntax().children_with_tokens().enumerate().find_map(|(ix, child)| {
            let token = child.into_token()?;
            let bin_op = match token.kind() {
                T![..] => RangeOp::Exclusive,
                T![..=] => RangeOp::Inclusive,
                _ => return None,
            };
            Some((ix, token, bin_op))
        })
    }

    pub fn op_kind(&self) -> Option<RangeOp> {
        self.op_details().map(|t| t.2)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.op_details().map(|t| t.1)
    }

    pub fn start(&self) -> Option<ast::Expr> {
        let op_ix = self.op_details()?.0;
        self.syntax()
            .children_with_tokens()
            .take(op_ix)
            .find_map(|it| ast::Expr::cast(it.into_node()?))
    }

    pub fn end(&self) -> Option<ast::Expr> {
        let op_ix = self.op_details()?.0;
        self.syntax()
            .children_with_tokens()
            .skip(op_ix + 1)
            .find_map(|it| ast::Expr::cast(it.into_node()?))
    }
}

impl ast::IndexExpr {
    pub fn base(&self) -> Option<ast::Expr> {
        support::children(self.syntax()).next()
    }
    pub fn index(&self) -> Option<ast::Expr> {
        support::children(self.syntax()).nth(1)
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
                initializer: support::children(self.syntax()).next(),
                repeat: support::children(self.syntax()).nth(1),
            }
        } else {
            ArrayExprKind::ElementList(support::children(self.syntax()))
        }
    }

    fn is_repeat(&self) -> bool {
        self.semicolon_token().is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralKind {
    String(ast::String),
    ByteString(ast::ByteString),
    IntNumber(ast::IntNumber),
    FloatNumber(ast::FloatNumber),
    Char(ast::Char),
    Byte(ast::Byte),
    Bool(bool),
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
        let token = self.token();

        if let Some(t) = ast::IntNumber::cast(token.clone()) {
            return LiteralKind::IntNumber(t);
        }
        if let Some(t) = ast::FloatNumber::cast(token.clone()) {
            return LiteralKind::FloatNumber(t);
        }
        if let Some(t) = ast::String::cast(token.clone()) {
            return LiteralKind::String(t);
        }
        if let Some(t) = ast::ByteString::cast(token.clone()) {
            return LiteralKind::ByteString(t);
        }
        if let Some(t) = ast::Char::cast(token.clone()) {
            return LiteralKind::Char(t);
        }
        if let Some(t) = ast::Byte::cast(token.clone()) {
            return LiteralKind::Byte(t);
        }

        match token.kind() {
            T![true] => LiteralKind::Bool(true),
            T![false] => LiteralKind::Bool(false),
            _ => unreachable!(),
        }
    }
}

pub enum BlockModifier {
    Async(SyntaxToken),
    Unsafe(SyntaxToken),
    Try(SyntaxToken),
    Const(SyntaxToken),
    Label(ast::Label),
}

impl ast::BlockExpr {
    pub fn modifier(&self) -> Option<BlockModifier> {
        self.async_token()
            .map(BlockModifier::Async)
            .or_else(|| self.unsafe_token().map(BlockModifier::Unsafe))
            .or_else(|| self.try_token().map(BlockModifier::Try))
            .or_else(|| self.const_token().map(BlockModifier::Const))
            .or_else(|| self.label().map(BlockModifier::Label))
    }
    /// false if the block is an intrinsic part of the syntax and can't be
    /// replaced with arbitrary expression.
    ///
    /// ```not_rust
    /// fn foo() { not_stand_alone }
    /// const FOO: () = { stand_alone };
    /// ```
    pub fn is_standalone(&self) -> bool {
        let parent = match self.syntax().parent() {
            Some(it) => it,
            None => return true,
        };
        match parent.kind() {
            FOR_EXPR | IF_EXPR => parent
                .children()
                .filter(|it| ast::Expr::can_cast(it.kind()))
                .next()
                .map_or(true, |it| it == *self.syntax()),
            LET_ELSE | FN | WHILE_EXPR | LOOP_EXPR | CONST_BLOCK_PAT => false,
            _ => true,
        }
    }
}

#[test]
fn test_literal_with_attr() {
    let parse = ast::SourceFile::parse(r#"const _: &str = { #[attr] "Hello" };"#);
    let lit = parse.tree().syntax().descendants().find_map(ast::Literal::cast).unwrap();
    assert_eq!(lit.token().text(), r#""Hello""#);
}

impl ast::RecordExprField {
    pub fn parent_record_lit(&self) -> ast::RecordExpr {
        self.syntax().ancestors().find_map(ast::RecordExpr::cast).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CallableExpr {
    Call(ast::CallExpr),
    MethodCall(ast::MethodCallExpr),
}

impl ast::HasAttrs for CallableExpr {}
impl ast::HasArgList for CallableExpr {}

impl AstNode for CallableExpr {
    fn can_cast(kind: parser::SyntaxKind) -> bool
    where
        Self: Sized,
    {
        ast::CallExpr::can_cast(kind) || ast::MethodCallExpr::can_cast(kind)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        if let Some(it) = ast::CallExpr::cast(syntax.clone()) {
            Some(Self::Call(it))
        } else {
            ast::MethodCallExpr::cast(syntax).map(Self::MethodCall)
        }
    }

    fn syntax(&self) -> &SyntaxNode {
        match self {
            Self::Call(it) => it.syntax(),
            Self::MethodCall(it) => it.syntax(),
        }
    }
}
