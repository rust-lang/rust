use {
    ast,
    SyntaxNodeRef, AstNode,
    SyntaxKind::*,
};

// ArrayExpr
#[derive(Debug, Clone, Copy)]
pub struct ArrayExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ArrayExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            ARRAY_EXPR => Some(ArrayExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ArrayExpr<'a> {}

// ArrayType
#[derive(Debug, Clone, Copy)]
pub struct ArrayType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ArrayType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            ARRAY_TYPE => Some(ArrayType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ArrayType<'a> {}

// Attr
#[derive(Debug, Clone, Copy)]
pub struct Attr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Attr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            ATTR => Some(Attr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Attr<'a> {
    pub fn value(self) -> Option<TokenTree<'a>> {
        super::child_opt(self)
    }
}

// BinExpr
#[derive(Debug, Clone, Copy)]
pub struct BinExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for BinExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            BIN_EXPR => Some(BinExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> BinExpr<'a> {}

// BindPat
#[derive(Debug, Clone, Copy)]
pub struct BindPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for BindPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            BIND_PAT => Some(BindPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for BindPat<'a> {}
impl<'a> BindPat<'a> {}

// Block
#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Block<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            BLOCK => Some(Block { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Block<'a> {
    pub fn statements(self) -> impl Iterator<Item = Stmt<'a>> + 'a {
        super::children(self)
    }

    pub fn expr(self) -> Option<Expr<'a>> {
        super::child_opt(self)
    }
}

// BlockExpr
#[derive(Debug, Clone, Copy)]
pub struct BlockExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for BlockExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            BLOCK_EXPR => Some(BlockExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> BlockExpr<'a> {
    pub fn block(self) -> Option<Block<'a>> {
        super::child_opt(self)
    }
}

// BreakExpr
#[derive(Debug, Clone, Copy)]
pub struct BreakExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for BreakExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            BREAK_EXPR => Some(BreakExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> BreakExpr<'a> {}

// CallExpr
#[derive(Debug, Clone, Copy)]
pub struct CallExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for CallExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            CALL_EXPR => Some(CallExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> CallExpr<'a> {}

// CastExpr
#[derive(Debug, Clone, Copy)]
pub struct CastExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for CastExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            CAST_EXPR => Some(CastExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> CastExpr<'a> {}

// Condition
#[derive(Debug, Clone, Copy)]
pub struct Condition<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Condition<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            CONDITION => Some(Condition { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Condition<'a> {
    pub fn pat(self) -> Option<Pat<'a>> {
        super::child_opt(self)
    }

    pub fn expr(self) -> Option<Expr<'a>> {
        super::child_opt(self)
    }
}

// ConstDef
#[derive(Debug, Clone, Copy)]
pub struct ConstDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ConstDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            CONST_DEF => Some(ConstDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for ConstDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for ConstDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for ConstDef<'a> {}
impl<'a> ConstDef<'a> {}

// ContinueExpr
#[derive(Debug, Clone, Copy)]
pub struct ContinueExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ContinueExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            CONTINUE_EXPR => Some(ContinueExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ContinueExpr<'a> {}

// DynTraitType
#[derive(Debug, Clone, Copy)]
pub struct DynTraitType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for DynTraitType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            DYN_TRAIT_TYPE => Some(DynTraitType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> DynTraitType<'a> {}

// EnumDef
#[derive(Debug, Clone, Copy)]
pub struct EnumDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for EnumDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            ENUM_DEF => Some(EnumDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for EnumDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for EnumDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for EnumDef<'a> {}
impl<'a> EnumDef<'a> {}

// Expr
#[derive(Debug, Clone, Copy)]
pub enum Expr<'a> {
    TupleExpr(TupleExpr<'a>),
    ArrayExpr(ArrayExpr<'a>),
    ParenExpr(ParenExpr<'a>),
    PathExpr(PathExpr<'a>),
    LambdaExpr(LambdaExpr<'a>),
    IfExpr(IfExpr<'a>),
    LoopExpr(LoopExpr<'a>),
    ForExpr(ForExpr<'a>),
    WhileExpr(WhileExpr<'a>),
    ContinueExpr(ContinueExpr<'a>),
    BreakExpr(BreakExpr<'a>),
    Label(Label<'a>),
    BlockExpr(BlockExpr<'a>),
    ReturnExpr(ReturnExpr<'a>),
    MatchExpr(MatchExpr<'a>),
    MatchArmList(MatchArmList<'a>),
    MatchArm(MatchArm<'a>),
    MatchGuard(MatchGuard<'a>),
    StructLit(StructLit<'a>),
    NamedFieldList(NamedFieldList<'a>),
    NamedField(NamedField<'a>),
    CallExpr(CallExpr<'a>),
    IndexExpr(IndexExpr<'a>),
    MethodCallExpr(MethodCallExpr<'a>),
    FieldExpr(FieldExpr<'a>),
    TryExpr(TryExpr<'a>),
    CastExpr(CastExpr<'a>),
    RefExpr(RefExpr<'a>),
    PrefixExpr(PrefixExpr<'a>),
    RangeExpr(RangeExpr<'a>),
    BinExpr(BinExpr<'a>),
    Literal(Literal<'a>),
}

impl<'a> AstNode<'a> for Expr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_EXPR => Some(Expr::TupleExpr(TupleExpr { syntax })),
            ARRAY_EXPR => Some(Expr::ArrayExpr(ArrayExpr { syntax })),
            PAREN_EXPR => Some(Expr::ParenExpr(ParenExpr { syntax })),
            PATH_EXPR => Some(Expr::PathExpr(PathExpr { syntax })),
            LAMBDA_EXPR => Some(Expr::LambdaExpr(LambdaExpr { syntax })),
            IF_EXPR => Some(Expr::IfExpr(IfExpr { syntax })),
            LOOP_EXPR => Some(Expr::LoopExpr(LoopExpr { syntax })),
            FOR_EXPR => Some(Expr::ForExpr(ForExpr { syntax })),
            WHILE_EXPR => Some(Expr::WhileExpr(WhileExpr { syntax })),
            CONTINUE_EXPR => Some(Expr::ContinueExpr(ContinueExpr { syntax })),
            BREAK_EXPR => Some(Expr::BreakExpr(BreakExpr { syntax })),
            LABEL => Some(Expr::Label(Label { syntax })),
            BLOCK_EXPR => Some(Expr::BlockExpr(BlockExpr { syntax })),
            RETURN_EXPR => Some(Expr::ReturnExpr(ReturnExpr { syntax })),
            MATCH_EXPR => Some(Expr::MatchExpr(MatchExpr { syntax })),
            MATCH_ARM_LIST => Some(Expr::MatchArmList(MatchArmList { syntax })),
            MATCH_ARM => Some(Expr::MatchArm(MatchArm { syntax })),
            MATCH_GUARD => Some(Expr::MatchGuard(MatchGuard { syntax })),
            STRUCT_LIT => Some(Expr::StructLit(StructLit { syntax })),
            NAMED_FIELD_LIST => Some(Expr::NamedFieldList(NamedFieldList { syntax })),
            NAMED_FIELD => Some(Expr::NamedField(NamedField { syntax })),
            CALL_EXPR => Some(Expr::CallExpr(CallExpr { syntax })),
            INDEX_EXPR => Some(Expr::IndexExpr(IndexExpr { syntax })),
            METHOD_CALL_EXPR => Some(Expr::MethodCallExpr(MethodCallExpr { syntax })),
            FIELD_EXPR => Some(Expr::FieldExpr(FieldExpr { syntax })),
            TRY_EXPR => Some(Expr::TryExpr(TryExpr { syntax })),
            CAST_EXPR => Some(Expr::CastExpr(CastExpr { syntax })),
            REF_EXPR => Some(Expr::RefExpr(RefExpr { syntax })),
            PREFIX_EXPR => Some(Expr::PrefixExpr(PrefixExpr { syntax })),
            RANGE_EXPR => Some(Expr::RangeExpr(RangeExpr { syntax })),
            BIN_EXPR => Some(Expr::BinExpr(BinExpr { syntax })),
            LITERAL => Some(Expr::Literal(Literal { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            Expr::TupleExpr(inner) => inner.syntax(),
            Expr::ArrayExpr(inner) => inner.syntax(),
            Expr::ParenExpr(inner) => inner.syntax(),
            Expr::PathExpr(inner) => inner.syntax(),
            Expr::LambdaExpr(inner) => inner.syntax(),
            Expr::IfExpr(inner) => inner.syntax(),
            Expr::LoopExpr(inner) => inner.syntax(),
            Expr::ForExpr(inner) => inner.syntax(),
            Expr::WhileExpr(inner) => inner.syntax(),
            Expr::ContinueExpr(inner) => inner.syntax(),
            Expr::BreakExpr(inner) => inner.syntax(),
            Expr::Label(inner) => inner.syntax(),
            Expr::BlockExpr(inner) => inner.syntax(),
            Expr::ReturnExpr(inner) => inner.syntax(),
            Expr::MatchExpr(inner) => inner.syntax(),
            Expr::MatchArmList(inner) => inner.syntax(),
            Expr::MatchArm(inner) => inner.syntax(),
            Expr::MatchGuard(inner) => inner.syntax(),
            Expr::StructLit(inner) => inner.syntax(),
            Expr::NamedFieldList(inner) => inner.syntax(),
            Expr::NamedField(inner) => inner.syntax(),
            Expr::CallExpr(inner) => inner.syntax(),
            Expr::IndexExpr(inner) => inner.syntax(),
            Expr::MethodCallExpr(inner) => inner.syntax(),
            Expr::FieldExpr(inner) => inner.syntax(),
            Expr::TryExpr(inner) => inner.syntax(),
            Expr::CastExpr(inner) => inner.syntax(),
            Expr::RefExpr(inner) => inner.syntax(),
            Expr::PrefixExpr(inner) => inner.syntax(),
            Expr::RangeExpr(inner) => inner.syntax(),
            Expr::BinExpr(inner) => inner.syntax(),
            Expr::Literal(inner) => inner.syntax(),
        }
    }
}

impl<'a> Expr<'a> {}

// ExprStmt
#[derive(Debug, Clone, Copy)]
pub struct ExprStmt<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ExprStmt<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            EXPR_STMT => Some(ExprStmt { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ExprStmt<'a> {
    pub fn expr(self) -> Option<Expr<'a>> {
        super::child_opt(self)
    }
}

// ExternCrateItem
#[derive(Debug, Clone, Copy)]
pub struct ExternCrateItem<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ExternCrateItem<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            EXTERN_CRATE_ITEM => Some(ExternCrateItem { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ExternCrateItem<'a> {}

// FieldExpr
#[derive(Debug, Clone, Copy)]
pub struct FieldExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for FieldExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FIELD_EXPR => Some(FieldExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> FieldExpr<'a> {}

// FieldPatList
#[derive(Debug, Clone, Copy)]
pub struct FieldPatList<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for FieldPatList<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FIELD_PAT_LIST => Some(FieldPatList { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> FieldPatList<'a> {}

// FnDef
#[derive(Debug, Clone, Copy)]
pub struct FnDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for FnDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FN_DEF => Some(FnDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for FnDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for FnDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for FnDef<'a> {}
impl<'a> FnDef<'a> {
    pub fn param_list(self) -> Option<ParamList<'a>> {
        super::child_opt(self)
    }

    pub fn body(self) -> Option<Block<'a>> {
        super::child_opt(self)
    }
}

// FnPointerType
#[derive(Debug, Clone, Copy)]
pub struct FnPointerType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for FnPointerType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FN_POINTER_TYPE => Some(FnPointerType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> FnPointerType<'a> {}

// ForExpr
#[derive(Debug, Clone, Copy)]
pub struct ForExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ForExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FOR_EXPR => Some(ForExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ForExpr<'a> {
    pub fn pat(self) -> Option<Pat<'a>> {
        super::child_opt(self)
    }

    pub fn iterable(self) -> Option<Expr<'a>> {
        super::child_opt(self)
    }

    pub fn body(self) -> Option<Block<'a>> {
        super::child_opt(self)
    }
}

// ForType
#[derive(Debug, Clone, Copy)]
pub struct ForType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ForType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FOR_TYPE => Some(ForType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ForType<'a> {}

// IfExpr
#[derive(Debug, Clone, Copy)]
pub struct IfExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for IfExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            IF_EXPR => Some(IfExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> IfExpr<'a> {
    pub fn condition(self) -> Option<Condition<'a>> {
        super::child_opt(self)
    }
}

// ImplItem
#[derive(Debug, Clone, Copy)]
pub struct ImplItem<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ImplItem<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            IMPL_ITEM => Some(ImplItem { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ImplItem<'a> {}

// ImplTraitType
#[derive(Debug, Clone, Copy)]
pub struct ImplTraitType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ImplTraitType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            IMPL_TRAIT_TYPE => Some(ImplTraitType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ImplTraitType<'a> {}

// IndexExpr
#[derive(Debug, Clone, Copy)]
pub struct IndexExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for IndexExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            INDEX_EXPR => Some(IndexExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> IndexExpr<'a> {}

// Label
#[derive(Debug, Clone, Copy)]
pub struct Label<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Label<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            LABEL => Some(Label { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Label<'a> {}

// LambdaExpr
#[derive(Debug, Clone, Copy)]
pub struct LambdaExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for LambdaExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            LAMBDA_EXPR => Some(LambdaExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> LambdaExpr<'a> {}

// LetStmt
#[derive(Debug, Clone, Copy)]
pub struct LetStmt<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for LetStmt<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            LET_STMT => Some(LetStmt { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> LetStmt<'a> {
    pub fn pat(self) -> Option<Pat<'a>> {
        super::child_opt(self)
    }

    pub fn initializer(self) -> Option<Expr<'a>> {
        super::child_opt(self)
    }
}

// Literal
#[derive(Debug, Clone, Copy)]
pub struct Literal<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Literal<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            LITERAL => Some(Literal { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Literal<'a> {}

// LoopExpr
#[derive(Debug, Clone, Copy)]
pub struct LoopExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for LoopExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            LOOP_EXPR => Some(LoopExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> LoopExpr<'a> {
    pub fn body(self) -> Option<Block<'a>> {
        super::child_opt(self)
    }
}

// MatchArm
#[derive(Debug, Clone, Copy)]
pub struct MatchArm<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for MatchArm<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            MATCH_ARM => Some(MatchArm { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> MatchArm<'a> {}

// MatchArmList
#[derive(Debug, Clone, Copy)]
pub struct MatchArmList<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for MatchArmList<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            MATCH_ARM_LIST => Some(MatchArmList { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> MatchArmList<'a> {}

// MatchExpr
#[derive(Debug, Clone, Copy)]
pub struct MatchExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for MatchExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            MATCH_EXPR => Some(MatchExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> MatchExpr<'a> {}

// MatchGuard
#[derive(Debug, Clone, Copy)]
pub struct MatchGuard<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for MatchGuard<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            MATCH_GUARD => Some(MatchGuard { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> MatchGuard<'a> {}

// MethodCallExpr
#[derive(Debug, Clone, Copy)]
pub struct MethodCallExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for MethodCallExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            METHOD_CALL_EXPR => Some(MethodCallExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> MethodCallExpr<'a> {}

// Module
#[derive(Debug, Clone, Copy)]
pub struct Module<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Module<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            MODULE => Some(Module { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for Module<'a> {}
impl<'a> ast::AttrsOwner<'a> for Module<'a> {}
impl<'a> Module<'a> {
    pub fn items(self) -> impl Iterator<Item = ModuleItem<'a>> + 'a {
        super::children(self)
    }
}

// ModuleItem
#[derive(Debug, Clone, Copy)]
pub enum ModuleItem<'a> {
    StructDef(StructDef<'a>),
    EnumDef(EnumDef<'a>),
    FnDef(FnDef<'a>),
    TraitDef(TraitDef<'a>),
    ImplItem(ImplItem<'a>),
    UseItem(UseItem<'a>),
    ExternCrateItem(ExternCrateItem<'a>),
}

impl<'a> AstNode<'a> for ModuleItem<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_DEF => Some(ModuleItem::StructDef(StructDef { syntax })),
            ENUM_DEF => Some(ModuleItem::EnumDef(EnumDef { syntax })),
            FN_DEF => Some(ModuleItem::FnDef(FnDef { syntax })),
            TRAIT_DEF => Some(ModuleItem::TraitDef(TraitDef { syntax })),
            IMPL_ITEM => Some(ModuleItem::ImplItem(ImplItem { syntax })),
            USE_ITEM => Some(ModuleItem::UseItem(UseItem { syntax })),
            EXTERN_CRATE_ITEM => Some(ModuleItem::ExternCrateItem(ExternCrateItem { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            ModuleItem::StructDef(inner) => inner.syntax(),
            ModuleItem::EnumDef(inner) => inner.syntax(),
            ModuleItem::FnDef(inner) => inner.syntax(),
            ModuleItem::TraitDef(inner) => inner.syntax(),
            ModuleItem::ImplItem(inner) => inner.syntax(),
            ModuleItem::UseItem(inner) => inner.syntax(),
            ModuleItem::ExternCrateItem(inner) => inner.syntax(),
        }
    }
}

impl<'a> ModuleItem<'a> {}

// Name
#[derive(Debug, Clone, Copy)]
pub struct Name<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Name<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NAME => Some(Name { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Name<'a> {}

// NameRef
#[derive(Debug, Clone, Copy)]
pub struct NameRef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for NameRef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NAME_REF => Some(NameRef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> NameRef<'a> {}

// NamedField
#[derive(Debug, Clone, Copy)]
pub struct NamedField<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for NamedField<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NAMED_FIELD => Some(NamedField { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> NamedField<'a> {}

// NamedFieldDef
#[derive(Debug, Clone, Copy)]
pub struct NamedFieldDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for NamedFieldDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NAMED_FIELD_DEF => Some(NamedFieldDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for NamedFieldDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for NamedFieldDef<'a> {}
impl<'a> NamedFieldDef<'a> {}

// NamedFieldList
#[derive(Debug, Clone, Copy)]
pub struct NamedFieldList<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for NamedFieldList<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NAMED_FIELD_LIST => Some(NamedFieldList { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> NamedFieldList<'a> {}

// NeverType
#[derive(Debug, Clone, Copy)]
pub struct NeverType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for NeverType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            NEVER_TYPE => Some(NeverType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> NeverType<'a> {}

// NominalDef
#[derive(Debug, Clone, Copy)]
pub enum NominalDef<'a> {
    StructDef(StructDef<'a>),
    EnumDef(EnumDef<'a>),
}

impl<'a> AstNode<'a> for NominalDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_DEF => Some(NominalDef::StructDef(StructDef { syntax })),
            ENUM_DEF => Some(NominalDef::EnumDef(EnumDef { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            NominalDef::StructDef(inner) => inner.syntax(),
            NominalDef::EnumDef(inner) => inner.syntax(),
        }
    }
}

impl<'a> ast::NameOwner<'a> for NominalDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for NominalDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for NominalDef<'a> {}
impl<'a> NominalDef<'a> {}

// Param
#[derive(Debug, Clone, Copy)]
pub struct Param<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Param<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PARAM => Some(Param { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Param<'a> {
    pub fn pat(self) -> Option<Pat<'a>> {
        super::child_opt(self)
    }
}

// ParamList
#[derive(Debug, Clone, Copy)]
pub struct ParamList<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ParamList<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PARAM_LIST => Some(ParamList { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ParamList<'a> {
    pub fn params(self) -> impl Iterator<Item = Param<'a>> + 'a {
        super::children(self)
    }
}

// ParenExpr
#[derive(Debug, Clone, Copy)]
pub struct ParenExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ParenExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PAREN_EXPR => Some(ParenExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ParenExpr<'a> {}

// ParenType
#[derive(Debug, Clone, Copy)]
pub struct ParenType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ParenType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PAREN_TYPE => Some(ParenType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ParenType<'a> {}

// Pat
#[derive(Debug, Clone, Copy)]
pub enum Pat<'a> {
    RefPat(RefPat<'a>),
    BindPat(BindPat<'a>),
    PlaceholderPat(PlaceholderPat<'a>),
    PathPat(PathPat<'a>),
    StructPat(StructPat<'a>),
    FieldPatList(FieldPatList<'a>),
    TupleStructPat(TupleStructPat<'a>),
    TuplePat(TuplePat<'a>),
    SlicePat(SlicePat<'a>),
    RangePat(RangePat<'a>),
}

impl<'a> AstNode<'a> for Pat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            REF_PAT => Some(Pat::RefPat(RefPat { syntax })),
            BIND_PAT => Some(Pat::BindPat(BindPat { syntax })),
            PLACEHOLDER_PAT => Some(Pat::PlaceholderPat(PlaceholderPat { syntax })),
            PATH_PAT => Some(Pat::PathPat(PathPat { syntax })),
            STRUCT_PAT => Some(Pat::StructPat(StructPat { syntax })),
            FIELD_PAT_LIST => Some(Pat::FieldPatList(FieldPatList { syntax })),
            TUPLE_STRUCT_PAT => Some(Pat::TupleStructPat(TupleStructPat { syntax })),
            TUPLE_PAT => Some(Pat::TuplePat(TuplePat { syntax })),
            SLICE_PAT => Some(Pat::SlicePat(SlicePat { syntax })),
            RANGE_PAT => Some(Pat::RangePat(RangePat { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            Pat::RefPat(inner) => inner.syntax(),
            Pat::BindPat(inner) => inner.syntax(),
            Pat::PlaceholderPat(inner) => inner.syntax(),
            Pat::PathPat(inner) => inner.syntax(),
            Pat::StructPat(inner) => inner.syntax(),
            Pat::FieldPatList(inner) => inner.syntax(),
            Pat::TupleStructPat(inner) => inner.syntax(),
            Pat::TuplePat(inner) => inner.syntax(),
            Pat::SlicePat(inner) => inner.syntax(),
            Pat::RangePat(inner) => inner.syntax(),
        }
    }
}

impl<'a> Pat<'a> {}

// PathExpr
#[derive(Debug, Clone, Copy)]
pub struct PathExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PathExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PATH_EXPR => Some(PathExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PathExpr<'a> {}

// PathPat
#[derive(Debug, Clone, Copy)]
pub struct PathPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PathPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PATH_PAT => Some(PathPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PathPat<'a> {}

// PathType
#[derive(Debug, Clone, Copy)]
pub struct PathType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PathType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PATH_TYPE => Some(PathType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PathType<'a> {}

// PlaceholderPat
#[derive(Debug, Clone, Copy)]
pub struct PlaceholderPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PlaceholderPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PLACEHOLDER_PAT => Some(PlaceholderPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PlaceholderPat<'a> {}

// PlaceholderType
#[derive(Debug, Clone, Copy)]
pub struct PlaceholderType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PlaceholderType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PLACEHOLDER_TYPE => Some(PlaceholderType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PlaceholderType<'a> {}

// PointerType
#[derive(Debug, Clone, Copy)]
pub struct PointerType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PointerType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            POINTER_TYPE => Some(PointerType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PointerType<'a> {}

// PrefixExpr
#[derive(Debug, Clone, Copy)]
pub struct PrefixExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for PrefixExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PREFIX_EXPR => Some(PrefixExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> PrefixExpr<'a> {}

// RangeExpr
#[derive(Debug, Clone, Copy)]
pub struct RangeExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for RangeExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            RANGE_EXPR => Some(RangeExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> RangeExpr<'a> {}

// RangePat
#[derive(Debug, Clone, Copy)]
pub struct RangePat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for RangePat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            RANGE_PAT => Some(RangePat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> RangePat<'a> {}

// RefExpr
#[derive(Debug, Clone, Copy)]
pub struct RefExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for RefExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            REF_EXPR => Some(RefExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> RefExpr<'a> {}

// RefPat
#[derive(Debug, Clone, Copy)]
pub struct RefPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for RefPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            REF_PAT => Some(RefPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> RefPat<'a> {}

// ReferenceType
#[derive(Debug, Clone, Copy)]
pub struct ReferenceType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ReferenceType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            REFERENCE_TYPE => Some(ReferenceType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ReferenceType<'a> {}

// ReturnExpr
#[derive(Debug, Clone, Copy)]
pub struct ReturnExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for ReturnExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            RETURN_EXPR => Some(ReturnExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ReturnExpr<'a> {}

// Root
#[derive(Debug, Clone, Copy)]
pub struct Root<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for Root<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            ROOT => Some(Root { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> Root<'a> {
    pub fn items(self) -> impl Iterator<Item = ModuleItem<'a>> + 'a {
        super::children(self)
    }

    pub fn functions(self) -> impl Iterator<Item = FnDef<'a>> + 'a {
        super::children(self)
    }

    pub fn modules(self) -> impl Iterator<Item = Module<'a>> + 'a {
        super::children(self)
    }
}

// SlicePat
#[derive(Debug, Clone, Copy)]
pub struct SlicePat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for SlicePat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            SLICE_PAT => Some(SlicePat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> SlicePat<'a> {}

// SliceType
#[derive(Debug, Clone, Copy)]
pub struct SliceType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for SliceType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            SLICE_TYPE => Some(SliceType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> SliceType<'a> {}

// StaticDef
#[derive(Debug, Clone, Copy)]
pub struct StaticDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for StaticDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STATIC_DEF => Some(StaticDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for StaticDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for StaticDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for StaticDef<'a> {}
impl<'a> StaticDef<'a> {}

// Stmt
#[derive(Debug, Clone, Copy)]
pub enum Stmt<'a> {
    ExprStmt(ExprStmt<'a>),
    LetStmt(LetStmt<'a>),
}

impl<'a> AstNode<'a> for Stmt<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            EXPR_STMT => Some(Stmt::ExprStmt(ExprStmt { syntax })),
            LET_STMT => Some(Stmt::LetStmt(LetStmt { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            Stmt::ExprStmt(inner) => inner.syntax(),
            Stmt::LetStmt(inner) => inner.syntax(),
        }
    }
}

impl<'a> Stmt<'a> {}

// StructDef
#[derive(Debug, Clone, Copy)]
pub struct StructDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for StructDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_DEF => Some(StructDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for StructDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for StructDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for StructDef<'a> {}
impl<'a> StructDef<'a> {
    pub fn fields(self) -> impl Iterator<Item = NamedFieldDef<'a>> + 'a {
        super::children(self)
    }
}

// StructLit
#[derive(Debug, Clone, Copy)]
pub struct StructLit<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for StructLit<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_LIT => Some(StructLit { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> StructLit<'a> {}

// StructPat
#[derive(Debug, Clone, Copy)]
pub struct StructPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for StructPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_PAT => Some(StructPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> StructPat<'a> {}

// TokenTree
#[derive(Debug, Clone, Copy)]
pub struct TokenTree<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TokenTree<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TOKEN_TREE => Some(TokenTree { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TokenTree<'a> {}

// TraitDef
#[derive(Debug, Clone, Copy)]
pub struct TraitDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TraitDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TRAIT_DEF => Some(TraitDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for TraitDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for TraitDef<'a> {}
impl<'a> TraitDef<'a> {}

// TryExpr
#[derive(Debug, Clone, Copy)]
pub struct TryExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TryExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TRY_EXPR => Some(TryExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TryExpr<'a> {}

// TupleExpr
#[derive(Debug, Clone, Copy)]
pub struct TupleExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TupleExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_EXPR => Some(TupleExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TupleExpr<'a> {}

// TuplePat
#[derive(Debug, Clone, Copy)]
pub struct TuplePat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TuplePat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_PAT => Some(TuplePat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TuplePat<'a> {}

// TupleStructPat
#[derive(Debug, Clone, Copy)]
pub struct TupleStructPat<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TupleStructPat<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_STRUCT_PAT => Some(TupleStructPat { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TupleStructPat<'a> {}

// TupleType
#[derive(Debug, Clone, Copy)]
pub struct TupleType<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TupleType<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_TYPE => Some(TupleType { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TupleType<'a> {}

// TypeDef
#[derive(Debug, Clone, Copy)]
pub struct TypeDef<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TypeDef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TYPE_DEF => Some(TypeDef { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for TypeDef<'a> {}
impl<'a> ast::TypeParamsOwner<'a> for TypeDef<'a> {}
impl<'a> ast::AttrsOwner<'a> for TypeDef<'a> {}
impl<'a> TypeDef<'a> {}

// TypeParam
#[derive(Debug, Clone, Copy)]
pub struct TypeParam<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TypeParam<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TYPE_PARAM => Some(TypeParam { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> ast::NameOwner<'a> for TypeParam<'a> {}
impl<'a> TypeParam<'a> {}

// TypeParamList
#[derive(Debug, Clone, Copy)]
pub struct TypeParamList<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for TypeParamList<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            TYPE_PARAM_LIST => Some(TypeParamList { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> TypeParamList<'a> {
    pub fn type_params(self) -> impl Iterator<Item = TypeParam<'a>> + 'a {
        super::children(self)
    }
}

// TypeRef
#[derive(Debug, Clone, Copy)]
pub enum TypeRef<'a> {
    ParenType(ParenType<'a>),
    TupleType(TupleType<'a>),
    NeverType(NeverType<'a>),
    PathType(PathType<'a>),
    PointerType(PointerType<'a>),
    ArrayType(ArrayType<'a>),
    SliceType(SliceType<'a>),
    ReferenceType(ReferenceType<'a>),
    PlaceholderType(PlaceholderType<'a>),
    FnPointerType(FnPointerType<'a>),
    ForType(ForType<'a>),
    ImplTraitType(ImplTraitType<'a>),
    DynTraitType(DynTraitType<'a>),
}

impl<'a> AstNode<'a> for TypeRef<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            PAREN_TYPE => Some(TypeRef::ParenType(ParenType { syntax })),
            TUPLE_TYPE => Some(TypeRef::TupleType(TupleType { syntax })),
            NEVER_TYPE => Some(TypeRef::NeverType(NeverType { syntax })),
            PATH_TYPE => Some(TypeRef::PathType(PathType { syntax })),
            POINTER_TYPE => Some(TypeRef::PointerType(PointerType { syntax })),
            ARRAY_TYPE => Some(TypeRef::ArrayType(ArrayType { syntax })),
            SLICE_TYPE => Some(TypeRef::SliceType(SliceType { syntax })),
            REFERENCE_TYPE => Some(TypeRef::ReferenceType(ReferenceType { syntax })),
            PLACEHOLDER_TYPE => Some(TypeRef::PlaceholderType(PlaceholderType { syntax })),
            FN_POINTER_TYPE => Some(TypeRef::FnPointerType(FnPointerType { syntax })),
            FOR_TYPE => Some(TypeRef::ForType(ForType { syntax })),
            IMPL_TRAIT_TYPE => Some(TypeRef::ImplTraitType(ImplTraitType { syntax })),
            DYN_TRAIT_TYPE => Some(TypeRef::DynTraitType(DynTraitType { syntax })),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            TypeRef::ParenType(inner) => inner.syntax(),
            TypeRef::TupleType(inner) => inner.syntax(),
            TypeRef::NeverType(inner) => inner.syntax(),
            TypeRef::PathType(inner) => inner.syntax(),
            TypeRef::PointerType(inner) => inner.syntax(),
            TypeRef::ArrayType(inner) => inner.syntax(),
            TypeRef::SliceType(inner) => inner.syntax(),
            TypeRef::ReferenceType(inner) => inner.syntax(),
            TypeRef::PlaceholderType(inner) => inner.syntax(),
            TypeRef::FnPointerType(inner) => inner.syntax(),
            TypeRef::ForType(inner) => inner.syntax(),
            TypeRef::ImplTraitType(inner) => inner.syntax(),
            TypeRef::DynTraitType(inner) => inner.syntax(),
        }
    }
}

impl<'a> TypeRef<'a> {}

// UseItem
#[derive(Debug, Clone, Copy)]
pub struct UseItem<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for UseItem<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            USE_ITEM => Some(UseItem { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> UseItem<'a> {}

// WhereClause
#[derive(Debug, Clone, Copy)]
pub struct WhereClause<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for WhereClause<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            WHERE_CLAUSE => Some(WhereClause { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> WhereClause<'a> {}

// WhileExpr
#[derive(Debug, Clone, Copy)]
pub struct WhileExpr<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for WhileExpr<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            WHILE_EXPR => Some(WhileExpr { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> WhileExpr<'a> {
    pub fn condition(self) -> Option<Condition<'a>> {
        super::child_opt(self)
    }

    pub fn body(self) -> Option<Block<'a>> {
        super::child_opt(self)
    }
}

