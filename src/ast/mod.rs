mod generated;

use std::sync::Arc;
use {
    SyntaxNode, SyntaxRoot, TreeRoot,
    SyntaxKind::*,
};
pub use self::generated::*;

pub trait AstNode<R: TreeRoot>: Sized {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self>;
    fn syntax(&self) -> &SyntaxNode<R>;
}

impl File<Arc<SyntaxRoot>> {
    pub fn parse(text: &str) -> Self {
        File::cast(::parse(text)).unwrap()
    }
}

impl<R: TreeRoot> File<R> {
    pub fn functions<'a>(&'a self) -> impl Iterator<Item = Function<R>> + 'a {
        self.syntax()
            .children()
            .filter_map(Function::cast)
    }
}

impl<R: TreeRoot> Function<R> {
    pub fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }

    pub fn has_atom_attr(&self, atom: &str) -> bool {
        self.syntax()
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
        self.syntax().text()
    }
}
