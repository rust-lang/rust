use std::{
    fmt,
    sync::Arc,
};

use {
    TextRange, TextUnit, SyntaxKind,
    yellow::{Ptr, RedNode, GreenNode, TextLen},
};

#[derive(Clone)]
pub struct SyntaxNode {
    pub(crate) root: SyntaxRoot,
    red: Ptr<RedNode>,
    trivia_pos: Option<(usize, usize)>,
}

#[derive(Clone)]
pub struct SyntaxRoot {
    red: Arc<RedNode>,
    pub(crate) errors: Arc<Vec<SError>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct SError {
    pub(crate) message: String,
    pub(crate) offset: TextUnit,
}

impl SyntaxNode {
    pub(crate) fn new(root: Arc<GreenNode>, errors: Vec<SError>) -> SyntaxNode {
        let root = Arc::new(RedNode::new_root(root));
        let red = Ptr::new(&root);
        let root = SyntaxRoot { red: root, errors: Arc::new(errors) };
        SyntaxNode { root, red, trivia_pos: None }
    }

    pub fn kind(&self) -> SyntaxKind {
        let green = self.red().green();
        match self.trivia_pos {
            None => green.kind(),
            Some((i, j)) => green.nth_trivias(i)[j].kind
        }
    }

    pub fn range(&self) -> TextRange {
        let red = self.red();
        let green = red.green();
        match self.trivia_pos {
            None => TextRange::offset_len(red.start_offset(), red.green().text_len()),
            Some((i, j)) => {
                let trivias = green.nth_trivias(i);
                let offset = if i == 0 {
                    red.start_offset()
                } else {
                    let prev_child = red.nth_child(Ptr::clone(&self.red), i - 1);
                    let mut offset = prev_child.start_offset() + prev_child.green().text_len();
                    for k in 0..j {
                        offset += &trivias[k].text_len();
                    }
                    offset
                };
                TextRange::offset_len(offset, trivias[j].text_len())
            }
        }
    }

    pub fn text(&self) -> String {
        let green = self.red().green();
        match self.trivia_pos {
            None => green.text(),
            Some((i, j)) => green.nth_trivias(i)[j].text.clone()
        }
    }

    pub fn children(&self) -> Vec<SyntaxNode> {
        let mut res = Vec::new();
        let red = self.red();
        let green = red.green();
        if green.is_leaf() || self.trivia_pos.is_some() {
            return Vec::new();
        }
        for (j, _) in green.nth_trivias(0).iter().enumerate() {
            res.push(SyntaxNode {
                root: self.root.clone(),
                red: Ptr::clone(&self.red),
                trivia_pos: Some((0, j)),
            })
        }

        let n_children = red.n_children();
        for i in 0..n_children {
            res.push(SyntaxNode {
                root: self.root.clone(),
                red: Ptr::new(&red.nth_child(Ptr::clone(&self.red), i)),
                trivia_pos: None,
            });
            for (j, _) in green.nth_trivias(i + 1).iter().enumerate() {
                res.push(SyntaxNode {
                    root: self.root.clone(),
                    red: self.red.clone(),
                    trivia_pos: Some((i + 1, j)),
                })
            }
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
    use syntax_kinds::*;
    match kind {
        IDENT | LIFETIME => true,
        _ => false,
    }
}
