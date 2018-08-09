use std::sync::Arc;
use {
    SyntaxNode, SyntaxRoot, TreeRoot, AstNode,
    SyntaxKind::*,
};


#[derive(Debug)]
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


#[derive(Debug)]
pub struct FnItem<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl<R: TreeRoot> AstNode<R> for FnItem<R> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self> {
        match syntax.kind() {
            FN_ITEM => Some(FnItem { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode<R> { &self.syntax }
}


#[derive(Debug)]
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

