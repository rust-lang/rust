mod generated;

use std::sync::Arc;

use smol_str::SmolStr;

use {
    SyntaxNode, SyntaxRoot, TreeRoot, SyntaxError,
    SyntaxKind::*,
};
pub use self::generated::*;

pub trait AstNode<R: TreeRoot>: Sized {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self>;
    fn syntax(&self) -> &SyntaxNode<R>;
}

pub trait NameOwner<R: TreeRoot>: AstNode<R> {
    fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

impl File<Arc<SyntaxRoot>> {
    pub fn parse(text: &str) -> Self {
        File::cast(::parse(text)).unwrap()
    }
}

impl<R: TreeRoot> File<R> {
    pub fn errors(&self) -> Vec<SyntaxError> {
        self.syntax().root.errors.clone()
    }
}

impl<R: TreeRoot> Function<R> {
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
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child()
            .unwrap();
        ident.leaf_text().unwrap()
    }
}
