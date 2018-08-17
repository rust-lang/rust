use {
    ast,
    SyntaxNode, OwnedRoot, TreeRoot, AstNode,
    SyntaxKind::*,
};

// ArrayType
#[derive(Debug, Clone, Copy)]
pub struct ArrayType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ArrayType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            ARRAY_TYPE => Some(ArrayType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ArrayType<R> {}

// Attr
#[derive(Debug, Clone, Copy)]
pub struct Attr<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Attr<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            ATTR => Some(Attr { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> Attr<R> {
    pub fn value(&self) -> Option<TokenTree<R>> {
        self.syntax()
            .children()
            .filter_map(TokenTree::cast)
            .next()
    }
}

// ConstDef
#[derive(Debug, Clone, Copy)]
pub struct ConstDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ConstDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            CONST_DEF => Some(ConstDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for ConstDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for ConstDef<R> {}
impl<R: TreeRoot> ConstDef<R> {}

// DynTraitType
#[derive(Debug, Clone, Copy)]
pub struct DynTraitType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for DynTraitType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            DYN_TRAIT_TYPE => Some(DynTraitType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> DynTraitType<R> {}

// EnumDef
#[derive(Debug, Clone, Copy)]
pub struct EnumDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for EnumDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            ENUM_DEF => Some(EnumDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for EnumDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for EnumDef<R> {}
impl<R: TreeRoot> EnumDef<R> {}

// File
#[derive(Debug, Clone, Copy)]
pub struct File<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for File<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FILE => Some(File { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> File<R> {
    pub fn functions<'a>(&'a self) -> impl Iterator<Item = FnDef<R>> + 'a {
        self.syntax()
            .children()
            .filter_map(FnDef::cast)
    }
}

// FnDef
#[derive(Debug, Clone, Copy)]
pub struct FnDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for FnDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FN_DEF => Some(FnDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for FnDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for FnDef<R> {}
impl<R: TreeRoot> FnDef<R> {}

// FnPointerType
#[derive(Debug, Clone, Copy)]
pub struct FnPointerType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for FnPointerType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FN_POINTER_TYPE => Some(FnPointerType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> FnPointerType<R> {}

// ForType
#[derive(Debug, Clone, Copy)]
pub struct ForType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ForType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FOR_TYPE => Some(ForType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ForType<R> {}

// ImplItem
#[derive(Debug, Clone, Copy)]
pub struct ImplItem<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ImplItem<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            IMPL_ITEM => Some(ImplItem { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ImplItem<R> {}

// ImplTraitType
#[derive(Debug, Clone, Copy)]
pub struct ImplTraitType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ImplTraitType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            IMPL_TRAIT_TYPE => Some(ImplTraitType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ImplTraitType<R> {}

// Module
#[derive(Debug, Clone, Copy)]
pub struct Module<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Module<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            MODULE => Some(Module { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for Module<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for Module<R> {}
impl<R: TreeRoot> Module<R> {}

// Name
#[derive(Debug, Clone, Copy)]
pub struct Name<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Name<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            NAME => Some(Name { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> Name<R> {}

// NameRef
#[derive(Debug, Clone, Copy)]
pub struct NameRef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for NameRef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            NAME_REF => Some(NameRef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> NameRef<R> {}

// NamedField
#[derive(Debug, Clone, Copy)]
pub struct NamedField<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for NamedField<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            NAMED_FIELD => Some(NamedField { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for NamedField<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for NamedField<R> {}
impl<R: TreeRoot> NamedField<R> {}

// NeverType
#[derive(Debug, Clone, Copy)]
pub struct NeverType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for NeverType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            NEVER_TYPE => Some(NeverType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> NeverType<R> {}

// NominalDef
#[derive(Debug, Clone, Copy)]
pub enum NominalDef<R: TreeRoot = OwnedRoot> {
    StructDef(StructDef<R>),
    EnumDef(EnumDef<R>),
}

impl<R: TreeRoot> AstNode<R> for NominalDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_DEF => Some(NominalDef::StructDef(StructDef { syntax })),
            ENUM_DEF => Some(NominalDef::EnumDef(EnumDef { syntax })),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> {
        match self {
            NominalDef::StructDef(inner) => inner.syntax(),
            NominalDef::EnumDef(inner) => inner.syntax(),
        }
    }
}

impl<R: TreeRoot> ast::AttrsOwner<R> for NominalDef<R> {}
impl<R: TreeRoot> NominalDef<R> {}

// ParenType
#[derive(Debug, Clone, Copy)]
pub struct ParenType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ParenType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            PAREN_TYPE => Some(ParenType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ParenType<R> {}

// PathType
#[derive(Debug, Clone, Copy)]
pub struct PathType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for PathType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            PATH_TYPE => Some(PathType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> PathType<R> {}

// PlaceholderType
#[derive(Debug, Clone, Copy)]
pub struct PlaceholderType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for PlaceholderType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            PLACEHOLDER_TYPE => Some(PlaceholderType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> PlaceholderType<R> {}

// PointerType
#[derive(Debug, Clone, Copy)]
pub struct PointerType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for PointerType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            POINTER_TYPE => Some(PointerType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> PointerType<R> {}

// ReferenceType
#[derive(Debug, Clone, Copy)]
pub struct ReferenceType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ReferenceType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            REFERENCE_TYPE => Some(ReferenceType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ReferenceType<R> {}

// SliceType
#[derive(Debug, Clone, Copy)]
pub struct SliceType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for SliceType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            SLICE_TYPE => Some(SliceType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> SliceType<R> {}

// StaticDef
#[derive(Debug, Clone, Copy)]
pub struct StaticDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for StaticDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            STATIC_DEF => Some(StaticDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for StaticDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for StaticDef<R> {}
impl<R: TreeRoot> StaticDef<R> {}

// StructDef
#[derive(Debug, Clone, Copy)]
pub struct StructDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for StructDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            STRUCT_DEF => Some(StructDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for StructDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for StructDef<R> {}
impl<R: TreeRoot> StructDef<R> {
    pub fn fields<'a>(&'a self) -> impl Iterator<Item = NamedField<R>> + 'a {
        self.syntax()
            .children()
            .filter_map(NamedField::cast)
    }
}

// TokenTree
#[derive(Debug, Clone, Copy)]
pub struct TokenTree<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for TokenTree<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            TOKEN_TREE => Some(TokenTree { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> TokenTree<R> {}

// TraitDef
#[derive(Debug, Clone, Copy)]
pub struct TraitDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for TraitDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            TRAIT_DEF => Some(TraitDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for TraitDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for TraitDef<R> {}
impl<R: TreeRoot> TraitDef<R> {}

// TupleType
#[derive(Debug, Clone, Copy)]
pub struct TupleType<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for TupleType<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            TUPLE_TYPE => Some(TupleType { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> TupleType<R> {}

// TypeDef
#[derive(Debug, Clone, Copy)]
pub struct TypeDef<R: TreeRoot = OwnedRoot> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for TypeDef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            TYPE_DEF => Some(TypeDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for TypeDef<R> {}
impl<R: TreeRoot> ast::AttrsOwner<R> for TypeDef<R> {}
impl<R: TreeRoot> TypeDef<R> {}

// TypeRef
#[derive(Debug, Clone, Copy)]
pub enum TypeRef<R: TreeRoot = OwnedRoot> {
    ParenType(ParenType<R>),
    TupleType(TupleType<R>),
    NeverType(NeverType<R>),
    PathType(PathType<R>),
    PointerType(PointerType<R>),
    ArrayType(ArrayType<R>),
    SliceType(SliceType<R>),
    ReferenceType(ReferenceType<R>),
    PlaceholderType(PlaceholderType<R>),
    FnPointerType(FnPointerType<R>),
    ForType(ForType<R>),
    ImplTraitType(ImplTraitType<R>),
    DynTraitType(DynTraitType<R>),
}

impl<R: TreeRoot> AstNode<R> for TypeRef<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
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
    fn syntax(&self) -> &SyntaxNode<R> {
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

impl<R: TreeRoot> TypeRef<R> {}

