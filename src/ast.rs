use std::sync::Arc;
use {
    SyntaxNode, SyntaxRoot, TreeRoot,
    SyntaxKind::*,
};

#[derive(Debug)]
pub struct File<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

#[derive(Debug)]
pub struct Function<R: TreeRoot = Arc<SyntaxRoot>> {
    syntax: SyntaxNode<R>,
}

#[derive(Debug)]
pub struct Name<R: TreeRoot = Arc<SyntaxRoot>> {
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
    pub fn functions<'a>(&'a self) -> impl Iterator<Item = Function<R>> + 'a {
        self.syntax
            .children()
            .filter(|node| node.kind() == FN_ITEM)
            .map(|node| Function { syntax: node })
    }
}

impl<R: TreeRoot> Function<R> {
    pub fn syntax(&self) -> SyntaxNode<R> {
        self.syntax.clone()
    }

    pub fn name(&self) -> Option<Name<R>> {
        self.syntax
            .children()
            .filter(|node| node.kind() == NAME)
            .map(|node| Name { syntax: node })
            .next()
    }

    pub fn has_atom_attr(&self, atom: &str) -> bool {
        self.syntax
            .children()
            .filter(|node| node.kind() == ATTR)
            .any(|attr| {
                let mut metas = attr.children().filter(|node| node.kind() == META_ITEM);
                let meta = match metas.next() {
                    None => return false,
                    Some(meta) => {
                        if metas.next().is_some() {
                            return false;
                        }
                        meta
                    }
                };
                let mut children = meta.children();
                match children.next() {
                    None => false,
                    Some(child) => {
                        if children.next().is_some() {
                            return false;
                        }
                        child.kind() == IDENT && child.text() == atom
                    }
                }
            })
    }
}

impl<R: TreeRoot> Name<R> {
    pub fn text(&self) -> String {
        self.syntax.text()
    }
}



impl<R: TreeRoot> File<R> {
    pub fn syntax(&self) -> SyntaxNode<R> {
        self.syntax.clone()
    }
}
