use std::{fmt, sync::Arc};

use smol_str::SmolStr;

use {
    yellow::{RedNode, TreeRoot, SyntaxRoot, RedPtr, RefRoot, OwnedRoot},
    SyntaxKind::{self, *},
    TextRange, TextUnit,
};


#[derive(Clone, Copy)]
pub struct SyntaxNode<R: TreeRoot = OwnedRoot> {
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

pub type SyntaxNodeRef<'a> = SyntaxNode<RefRoot<'a>>;

#[test]
fn syntax_node_ref_is_copy() {
    fn assert_copy<T: Copy>(){}
    assert_copy::<SyntaxNodeRef>()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SyntaxError {
    pub msg: String,
    pub offset: TextUnit,
}

impl SyntaxNode<OwnedRoot> {
    pub(crate) fn new_owned(root: SyntaxRoot) -> Self {
        let root = OwnedRoot(Arc::new(root));
        let red = RedPtr::new(&root.syntax_root().red);
        SyntaxNode { root, red }
    }
}

impl<R: TreeRoot> SyntaxNode<R> {
    pub fn as_ref<'a>(&'a self) -> SyntaxNode<RefRoot<'a>> {
        SyntaxNode {
            root: self.root.borrowed(),
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

    pub fn children(&self) -> impl Iterator<Item = SyntaxNode<R>> {
        let red = self.red;
        let n_children = self.red().n_children();
        let root = self.root.clone();
        (0..n_children).map(move |i| {
            let red = unsafe { red.get(&root) };
            SyntaxNode {
                root: root.clone(),
                red: red.get_child(i).unwrap(),
            }
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
        let red = self.red().get_child(0)?;
        Some(SyntaxNode { root: self.root.clone(), red })
    }

    pub fn last_child(&self) -> Option<SyntaxNode<R>> {
        let n = self.red().n_children();
        let n = n.checked_sub(1)?;
        let red = self.red().get_child(n)?;
        Some(SyntaxNode { root: self.root.clone(), red })
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

    pub fn prev_sibling(&self) -> Option<SyntaxNode<R>> {
        let red = self.red();
        let parent = self.parent()?;
        let prev_sibling_idx = red.index_in_parent()?.checked_sub(1)?;
        let sibling_red = parent.red().get_child(prev_sibling_idx)?;
        Some(SyntaxNode {
            root: self.root.clone(),
            red: sibling_red,
        })
    }

    pub fn is_leaf(&self) -> bool {
        self.first_child().is_none()
    }

    pub fn leaf_text(&self) -> Option<SmolStr> {
        self.red().green().leaf_text()
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
