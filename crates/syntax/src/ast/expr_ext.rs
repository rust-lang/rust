//! Various extension methods to ast Expr Nodes, which are hard to code-generate.

use rowan::WalkEvent;

use crate::{
    ast::{
        self,
        operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
        support, AstChildren, AstNode,
    },
    AstToken,
    SyntaxKind::*,
    SyntaxToken, T,
};

impl ast::AttrsOwner for ast::Expr {}

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
                | ast::Expr::EffectExpr(_)
        )
    }

    pub fn name_ref(&self) -> Option<ast::NameRef> {
        if let ast::Expr::PathExpr(expr) = self {
            let path = expr.path()?;
            let segment = path.segment()?;
            let name_ref = segment.name_ref()?;
            if path.qualifier().is_none() {
                return Some(name_ref);
            }
        }
        None
    }

    /// Preorder walk all the expression's child expressions.
    pub fn walk(&self, cb: &mut dyn FnMut(ast::Expr)) {
        self.preorder(&mut |ev| {
            if let WalkEvent::Enter(expr) = ev {
                cb(expr);
            }
            false
        })
    }

    /// Preorder walk all the expression's child expressions preserving events.
    /// If the callback returns true on an [`WalkEvent::Enter`], the subtree of the expression will be skipped.
    /// Note that the subtree may already be skipped due to the context analysis this function does.
    pub fn preorder(&self, cb: &mut dyn FnMut(WalkEvent<ast::Expr>) -> bool) {
        let mut preorder = self.syntax().preorder();
        while let Some(event) = preorder.next() {
            let node = match event {
                WalkEvent::Enter(node) => node,
                WalkEvent::Leave(node) => {
                    if let Some(expr) = ast::Expr::cast(node) {
                        cb(WalkEvent::Leave(expr));
                    }
                    continue;
                }
            };
            match ast::Stmt::cast(node.clone()) {
                // recursively walk the initializer, skipping potential const pat expressions
                // let statements aren't usually nested too deeply so this is fine to recurse on
                Some(ast::Stmt::LetStmt(l)) => {
                    if let Some(expr) = l.initializer() {
                        expr.preorder(cb);
                    }
                    preorder.skip_subtree();
                }
                // Don't skip subtree since we want to process the expression child next
                Some(ast::Stmt::ExprStmt(_)) => (),
                // skip inner items which might have their own expressions
                Some(ast::Stmt::Item(_)) => preorder.skip_subtree(),
                None => {
                    // skip const args, those expressions are a different context
                    if ast::GenericArg::can_cast(node.kind()) {
                        preorder.skip_subtree();
                    } else if let Some(expr) = ast::Expr::cast(node) {
                        let is_different_context = match &expr {
                            ast::Expr::EffectExpr(effect) => {
                                matches!(
                                    effect.effect(),
                                    ast::Effect::Async(_)
                                        | ast::Effect::Try(_)
                                        | ast::Effect::Const(_)
                                )
                            }
                            ast::Expr::ClosureExpr(_) => true,
                            _ => false,
                        };
                        let skip = cb(WalkEvent::Enter(expr));
                        if skip || is_different_context {
                            preorder.skip_subtree();
                        }
                    }
                }
            }
        }
    }

    /// Preorder walk all the expression's child patterns.
    pub fn walk_patterns(&self, cb: &mut dyn FnMut(ast::Pat)) {
        let mut preorder = self.syntax().preorder();
        while let Some(event) = preorder.next() {
            let node = match event {
                WalkEvent::Enter(node) => node,
                WalkEvent::Leave(_) => continue,
            };
            match ast::Stmt::cast(node.clone()) {
                Some(ast::Stmt::LetStmt(l)) => {
                    if let Some(pat) = l.pat() {
                        pat.walk(cb);
                    }
                    if let Some(expr) = l.initializer() {
                        expr.walk_patterns(cb);
                    }
                    preorder.skip_subtree();
                }
                // Don't skip subtree since we want to process the expression child next
                Some(ast::Stmt::ExprStmt(_)) => (),
                // skip inner items which might have their own patterns
                Some(ast::Stmt::Item(_)) => preorder.skip_subtree(),
                None => {
                    // skip const args, those are a different context
                    if ast::GenericArg::can_cast(node.kind()) {
                        preorder.skip_subtree();
                    } else if let Some(expr) = ast::Expr::cast(node.clone()) {
                        let is_different_context = match &expr {
                            ast::Expr::EffectExpr(effect) => {
                                matches!(
                                    effect.effect(),
                                    ast::Effect::Async(_)
                                        | ast::Effect::Try(_)
                                        | ast::Effect::Const(_)
                                )
                            }
                            ast::Expr::ClosureExpr(_) => true,
                            _ => false,
                        };
                        if is_different_context {
                            preorder.skip_subtree();
                        }
                    } else if let Some(pat) = ast::Pat::cast(node) {
                        preorder.skip_subtree();
                        pat.walk(cb);
                    }
                }
            }
        }
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
    pub fn then_branch(&self) -> Option<ast::BlockExpr> {
        self.blocks().next()
    }

    pub fn else_branch(&self) -> Option<ElseBranch> {
        let res = match self.blocks().nth(1) {
            Some(block) => ElseBranch::Block(block),
            None => {
                let elif: ast::IfExpr = support::child(self.syntax())?;
                ElseBranch::IfExpr(elif)
            }
        };
        Some(res)
    }

    pub fn blocks(&self) -> AstChildren<ast::BlockExpr> {
        support::children(self.syntax())
    }
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
        self.syntax().children_with_tokens().any(|it| it.kind() == T![;])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralKind {
    String(ast::String),
    ByteString(ast::ByteString),
    IntNumber(ast::IntNumber),
    FloatNumber(ast::FloatNumber),
    Char,
    Byte,
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

        match token.kind() {
            T![true] => LiteralKind::Bool(true),
            T![false] => LiteralKind::Bool(false),
            CHAR => LiteralKind::Char,
            BYTE => LiteralKind::Byte,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    Async(SyntaxToken),
    Unsafe(SyntaxToken),
    Try(SyntaxToken),
    Const(SyntaxToken),
    // Very much not an effect, but we stuff it into this node anyway
    Label(ast::Label),
}

impl ast::EffectExpr {
    pub fn effect(&self) -> Effect {
        if let Some(token) = self.async_token() {
            return Effect::Async(token);
        }
        if let Some(token) = self.unsafe_token() {
            return Effect::Unsafe(token);
        }
        if let Some(token) = self.try_token() {
            return Effect::Try(token);
        }
        if let Some(token) = self.const_token() {
            return Effect::Const(token);
        }
        if let Some(label) = self.label() {
            return Effect::Label(label);
        }
        unreachable!("ast::EffectExpr without Effect")
    }
}

impl ast::BlockExpr {
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
        !matches!(parent.kind(), FN | IF_EXPR | WHILE_EXPR | LOOP_EXPR | EFFECT_EXPR)
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
