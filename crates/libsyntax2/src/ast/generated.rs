use std::sync::Arc;
use {
    ast,
    SyntaxNode, SyntaxRoot, TreeRoot, AstNode,
    SyntaxKind::*,
};

// ConstDef
#[derive(Debug, Clone, Copy)]
pub struct ConstDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> ConstDef<R> {}

// EnumDef
#[derive(Debug, Clone, Copy)]
pub struct EnumDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> EnumDef<R> {}

// File
#[derive(Debug, Clone, Copy)]
pub struct File<R: TreeRoot = Arc<SyntaxRoot>> {
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
pub struct FnDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> FnDef<R> {}

// Module
#[derive(Debug, Clone, Copy)]
pub struct Module<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> Module<R> {}

// Name
#[derive(Debug, Clone, Copy)]
pub struct Name<R: TreeRoot = Arc<SyntaxRoot>> {
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
pub struct NameRef<R: TreeRoot = Arc<SyntaxRoot>> {
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

// StaticDef
#[derive(Debug, Clone, Copy)]
pub struct StaticDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> StaticDef<R> {}

// StructDef
#[derive(Debug, Clone, Copy)]
pub struct StructDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> StructDef<R> {}

// TraitDef
#[derive(Debug, Clone, Copy)]
pub struct TraitDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> TraitDef<R> {}

// TypeDef
#[derive(Debug, Clone, Copy)]
pub struct TypeDef<R: TreeRoot = Arc<SyntaxRoot>> {
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
impl<R: TreeRoot> TypeDef<R> {}

