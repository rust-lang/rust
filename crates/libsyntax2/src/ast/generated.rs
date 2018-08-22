use {
    ast,
    SyntaxNodeRef, AstNode,
    SyntaxKind::*,
};

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

// File
#[derive(Debug, Clone, Copy)]
pub struct File<'a> {
    syntax: SyntaxNodeRef<'a>,
}

impl<'a> AstNode<'a> for File<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        match syntax.kind() {
            FILE => Some(File { syntax }),
            _ => None,
        }
    }
    fn syntax(self) -> SyntaxNodeRef<'a> { self.syntax }
}

impl<'a> File<'a> {
    pub fn functions(self) -> impl Iterator<Item = FnDef<'a>> + 'a {
        super::children(self)
    }

    pub fn modules(self) -> impl Iterator<Item = Module<'a>> + 'a {
        super::children(self)
    }
}

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
impl<'a> FnDef<'a> {}

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
    pub fn modules(self) -> impl Iterator<Item = Module<'a>> + 'a {
        super::children(self)
    }
}

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

impl<'a> ast::NameOwner<'a> for NamedField<'a> {}
impl<'a> ast::AttrsOwner<'a> for NamedField<'a> {}
impl<'a> NamedField<'a> {}

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
    pub fn fields(self) -> impl Iterator<Item = NamedField<'a>> + 'a {
        super::children(self)
    }
}

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

