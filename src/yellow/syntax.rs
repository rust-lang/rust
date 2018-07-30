use std::{
    fmt,
    sync::Arc,
    ptr,
    ops::Deref,
};

use {
    TextRange, TextUnit,
    SyntaxKind::{self, *},
    yellow::{RedNode, GreenNode},
};

pub trait TreeRoot: Deref<Target=SyntaxRoot> + Clone {}
impl TreeRoot for Arc<SyntaxRoot> {}
impl<'a> TreeRoot for &'a SyntaxRoot {}

#[derive(Clone, Copy)]
pub struct SyntaxNode<ROOT: TreeRoot = Arc<SyntaxRoot>> {
    pub(crate) root: ROOT,
    // guaranteed to be alive bc SyntaxRoot holds a strong ref
    red: ptr::NonNull<RedNode>,
}

pub type SyntaxNodeRef<'a> = SyntaxNode<&'a SyntaxRoot>;

#[derive(Debug)]
pub struct SyntaxRoot {
    red: RedNode,
    pub(crate) errors: Vec<SyntaxError>,
}

impl SyntaxRoot {
    pub(crate) fn new(green: GreenNode, errors: Vec<SyntaxError>) -> SyntaxRoot {
        SyntaxRoot {
            red: RedNode::new_root(green),
            errors,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct SyntaxError {
    pub(crate) message: String,
    pub(crate) offset: TextUnit,
}

impl SyntaxNode<Arc<SyntaxRoot>> {
    pub(crate) fn new_owned(root: SyntaxRoot) -> Self {
        let root = Arc::new(root);
        let red_weak = ptr::NonNull::from(&root.red);
        SyntaxNode { root, red: red_weak }
    }
}

impl<ROOT: TreeRoot> SyntaxNode<ROOT> {
    pub fn borrow<'a>(&'a self) -> SyntaxNode<&'a SyntaxRoot> {
        SyntaxNode {
            root: &*self.root,
            red: ptr::NonNull::clone(&self.red),
        }
    }

    pub fn kind(&self) -> SyntaxKind {
        self.red().green().kind()
    }

    pub fn range(&self) -> TextRange {
        let red = self.red();
        TextRange::offset_len(
            red.start_offset(),
            red.green().text_len(),
        )
    }

    pub fn text(&self) -> String {
        self.red().green().text()
    }

    pub fn children(&self) -> Vec<SyntaxNode<ROOT>> {
        let red = self.red();
        let n_children = red.n_children();
        let mut res = Vec::with_capacity(n_children);
        for i in 0..n_children {
            res.push(SyntaxNode {
                root: self.root.clone(),
                red: red.nth_child(i),
            });
        }
        res
    }

    pub fn parent(&self) -> Option<SyntaxNode<ROOT>> {
        let parent = self.red().parent()?;
        Some(SyntaxNode {
            root: self.root.clone(),
            red: parent,
        })
    }

    fn red(&self) -> &RedNode {
        unsafe { self.red.as_ref() }
    }
}

impl<ROOT: TreeRoot> fmt::Debug for SyntaxNode<ROOT> {
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
        IDENT | LIFETIME => true,
        _ => false,
    }
}
