use std::{
    fmt,
    sync::Arc,
};

use {
    TextRange, TextUnit,
    SyntaxKind::{self, *},
    yellow::{Ptr, RedNode, GreenNode},
};

#[derive(Clone)]
pub struct SyntaxNode {
    pub(crate) root: SyntaxRoot,
    red: Ptr<RedNode>,
}

#[derive(Clone)]
pub struct SyntaxRoot {
    red: Arc<RedNode>,
    pub(crate) errors: Arc<Vec<SyntaxError>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct SyntaxError {
    pub(crate) message: String,
    pub(crate) offset: TextUnit,
}

impl SyntaxNode {
    pub(crate) fn new(root: GreenNode, errors: Vec<SyntaxError>) -> SyntaxNode {
        let root = Arc::new(RedNode::new_root(root));
        let red = Ptr::new(&root);
        let root = SyntaxRoot { red: root, errors: Arc::new(errors) };
        SyntaxNode { root, red }
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

    pub fn children(&self) -> Vec<SyntaxNode> {
        let red = self.red();
        let n_children = red.n_children();
        let mut res = Vec::with_capacity(n_children);
        for i in 0..n_children {
            res.push(SyntaxNode {
                root: self.root.clone(),
                red: Ptr::new(&red.nth_child(Ptr::clone(&self.red), i)),
            });
        }
        res
    }

    fn red(&self) -> &RedNode {
        // Safe b/c root ptr keeps red alive
        unsafe { self.red.get() }
    }
}

impl fmt::Debug for SyntaxNode {
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
