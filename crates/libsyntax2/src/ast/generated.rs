use std::sync::Arc;
use {
    ast,
    SyntaxNode, SyntaxRoot, TreeRoot, AstNode,
    SyntaxKind::*,
};

// ConstItem
#[derive(Debug, Clone, Copy)]
pub struct ConstItem<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for ConstItem<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            CONST_ITEM => Some(ConstItem { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for ConstItem<R> {}
impl<R: TreeRoot> ConstItem<R> {}

// Enum
#[derive(Debug, Clone, Copy)]
pub struct Enum<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Enum<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            ENUM => Some(Enum { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for Enum<R> {}
impl<R: TreeRoot> Enum<R> {}

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
    pub fn functions<'a>(&'a self) -> impl Iterator<Item = Function<R>> + 'a {
        self.syntax()
            .children()
            .filter_map(Function::cast)
    }
}

// Function
#[derive(Debug, Clone, Copy)]
pub struct Function<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Function<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FUNCTION => Some(Function { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for Function<R> {}
impl<R: TreeRoot> Function<R> {}

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

// StaticItem
#[derive(Debug, Clone, Copy)]
pub struct StaticItem<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for StaticItem<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            STATIC_ITEM => Some(StaticItem { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for StaticItem<R> {}
impl<R: TreeRoot> StaticItem<R> {}

// Struct
#[derive(Debug, Clone, Copy)]
pub struct Struct<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Struct<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            STRUCT => Some(Struct { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for Struct<R> {}
impl<R: TreeRoot> Struct<R> {}

// Trait
#[derive(Debug, Clone, Copy)]
pub struct Trait<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for Trait<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            TRAIT => Some(Trait { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}

impl<R: TreeRoot> ast::NameOwner<R> for Trait<R> {}
impl<R: TreeRoot> Trait<R> {}

