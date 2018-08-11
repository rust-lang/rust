use std::sync::Arc;
use {
    SyntaxNode, SyntaxRoot, TreeRoot, AstNode,
    SyntaxKind::*,
};

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

impl<R: TreeRoot> ConstItem<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> Enum<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> Function<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> Module<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> StaticItem<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> Struct<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

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

impl<R: TreeRoot> Trait<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

