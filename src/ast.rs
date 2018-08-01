use std::sync::Arc;
use {SyntaxNode, SyntaxRoot, TreeRoot};

#[derive(Debug)]
pub struct File<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

impl File<Arc<SyntaxRoot>> {
    pub fn parse(text: &str) -> Self {
        File {
            syntax: ::parse(text),
        }
    }
}

impl<R: TreeRoot> File<R> {
    pub fn syntax(&self) -> SyntaxNode<R> {
        self.syntax.clone()
    }
}
