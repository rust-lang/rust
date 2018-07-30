use std::sync::Arc;
use {SyntaxNode, TreeRoot, SyntaxRoot, SyntaxNodeRef};

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

    pub fn for_each_node(&self, mut f: impl FnMut(SyntaxNodeRef)) {
        let syntax = self.syntax();
        let syntax = syntax.borrow();
        go(syntax, &mut f);

        fn go(syntax: SyntaxNodeRef, f: &mut FnMut(SyntaxNodeRef)) {
            f(syntax);
            syntax.children().into_iter().for_each(f)
        }
    }
}
