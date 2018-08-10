use std::{fmt, sync::Arc};

use {
    yellow::{RedNode, TreeRoot, SyntaxRoot, RedPtr},
    SyntaxKind::{self, *},
    TextRange, TextUnit,
};


#[derive(Clone, Copy)]
pub struct SyntaxNode<R: TreeRoot = Arc<SyntaxRoot>> {
    pub(crate) root: R,
    // Guaranteed to not dangle, because `root` holds a
    // strong reference to red's ancestor
    red: RedPtr,
}

unsafe impl<R: TreeRoot> Send for SyntaxNode<R> {}
unsafe impl<R: TreeRoot> Sync for SyntaxNode<R> {}

impl<R1: TreeRoot, R2: TreeRoot> PartialEq<SyntaxNode<R1>> for SyntaxNode<R2> {
    fn eq(&self, other: &SyntaxNode<R1>) -> bool {
        self.red == other.red
    }
}

impl<R: TreeRoot> Eq for SyntaxNode<R> {}

pub type SyntaxNodeRef<'a> = SyntaxNode<&'a SyntaxRoot>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SyntaxError {
    pub msg: String,
    pub offset: TextUnit,
}

impl SyntaxNode<Arc<SyntaxRoot>> {
    pub(crate) fn new_owned(root: SyntaxRoot) -> Self {
        let root = Arc::new(root);
        let red = RedPtr::new(&root.red);
        SyntaxNode { root, red }
    }
}

impl<R: TreeRoot> SyntaxNode<R> {
    pub fn as_ref<'a>(&'a self) -> SyntaxNode<&'a SyntaxRoot> {
        SyntaxNode {
            root: &*self.root,
            red: self.red,
        }
    }

    pub fn kind(&self) -> SyntaxKind {
        self.red().green().kind()
    }

    pub fn range(&self) -> TextRange {
        let red = self.red();
        TextRange::offset_len(red.start_offset(), red.green().text_len())
    }

    pub fn text(&self) -> String {
        self.red().green().text()
    }

    pub fn children<'a>(&'a self) -> impl Iterator<Item = SyntaxNode<R>> + 'a {
        let red = self.red();
        let n_children = red.n_children();
        (0..n_children).map(move |i| SyntaxNode {
            root: self.root.clone(),
            red: red.get_child(i).unwrap(),
        })
    }

    pub fn parent(&self) -> Option<SyntaxNode<R>> {
        let parent = self.red().parent()?;
        Some(SyntaxNode {
            root: self.root.clone(),
            red: parent,
        })
    }

    pub fn first_child(&self) -> Option<SyntaxNode<R>> {
        self.children().next()
    }

    pub fn next_sibling(&self) -> Option<SyntaxNode<R>> {
        let red = self.red();
        let parent = self.parent()?;
        let next_sibling_idx = red.index_in_parent()? + 1;
        let sibling_red = parent.red().get_child(next_sibling_idx)?;
        Some(SyntaxNode {
            root: self.root.clone(),
            red: sibling_red,
        })
    }

    pub fn is_leaf(&self) -> bool {
        self.first_child().is_none()
    }

    fn red(&self) -> &RedNode {
        unsafe { self.red.get(&self.root) }
    }
}

impl<R: TreeRoot> fmt::Debug for SyntaxNode<R> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.kind(), self.range())?;
        if has_short_text(self.kind()) {
            write!(fmt, " \"{}\"", self.text())?;
        }
        Ok(())
    }
}

fn has_short_text(kind: SyntaxKind) -> bool {
    match kind {
        IDENT | LIFETIME | INT_NUMBER | FLOAT_NUMBER => true,
        _ => false,
    }
}
