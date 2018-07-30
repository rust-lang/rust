use std::sync::Arc;
use {SyntaxNode, TreeRoot, SyntaxRoot};

#[derive(Debug)]
pub struct File<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>
}

impl File<Arc<SyntaxRoot>> {
    pub fn parse(text: &str) -> Self {
        File { syntax: ::parse(text.to_owned()) }
    }
}

impl<R: TreeRoot> File<R> {
    pub fn syntax(&self) -> SyntaxNode<R> {
        self.syntax.clone()
    }
}
