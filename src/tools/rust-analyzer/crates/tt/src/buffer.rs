//! Stateful iteration over token trees.
//!
//! We use this as the source of tokens for parser.
use crate::{Leaf, Subtree, TokenTree, TokenTreesView};

pub struct Cursor<'a, Span> {
    buffer: &'a [TokenTree<Span>],
    index: usize,
    subtrees_stack: Vec<usize>,
}

impl<'a, Span: Copy> Cursor<'a, Span> {
    pub fn new(buffer: &'a [TokenTree<Span>]) -> Self {
        Self { buffer, index: 0, subtrees_stack: Vec::new() }
    }

    /// Check whether it is eof
    pub fn eof(&self) -> bool {
        self.index == self.buffer.len() && self.subtrees_stack.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.subtrees_stack.is_empty()
    }

    fn last_subtree(&self) -> Option<(usize, &'a Subtree<Span>)> {
        self.subtrees_stack.last().map(|&subtree_idx| {
            let TokenTree::Subtree(subtree) = &self.buffer[subtree_idx] else {
                panic!("subtree pointing to non-subtree");
            };
            (subtree_idx, subtree)
        })
    }

    pub fn end(&mut self) -> &'a Subtree<Span> {
        let (last_subtree_idx, last_subtree) =
            self.last_subtree().expect("called `Cursor::end()` without an open subtree");
        // +1 because `Subtree.len` excludes the subtree itself.
        assert_eq!(
            last_subtree_idx + last_subtree.usize_len() + 1,
            self.index,
            "called `Cursor::end()` without finishing a subtree"
        );
        self.subtrees_stack.pop();
        last_subtree
    }

    /// Returns the `TokenTree` at the cursor if it is not at the end of a subtree.
    pub fn token_tree(&self) -> Option<&'a TokenTree<Span>> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            if last_subtree_idx + last_subtree.usize_len() + 1 == self.index {
                return None;
            }
        }
        self.buffer.get(self.index)
    }

    /// Bump the cursor, and enters a subtree if it is on one.
    pub fn bump(&mut self) {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            assert_ne!(
                last_subtree_idx + last_subtree.usize_len() + 1,
                self.index,
                "called `Cursor::bump()` when at the end of a subtree"
            );
        }
        if let TokenTree::Subtree(_) = self.buffer[self.index] {
            self.subtrees_stack.push(self.index);
        }
        self.index += 1;
    }

    pub fn bump_or_end(&mut self) {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            if last_subtree_idx + last_subtree.usize_len() + 1 == self.index {
                self.subtrees_stack.pop();
                return;
            }
        }
        // +1 because `Subtree.len` excludes the subtree itself.
        if let TokenTree::Subtree(_) = self.buffer[self.index] {
            self.subtrees_stack.push(self.index);
        }
        self.index += 1;
    }

    pub fn peek_two_leaves(&self) -> Option<[&'a Leaf<Span>; 2]> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            let last_end = last_subtree_idx + last_subtree.usize_len() + 1;
            if last_end == self.index || last_end == self.index + 1 {
                return None;
            }
        }
        self.buffer.get(self.index..self.index + 2).and_then(|it| match it {
            [TokenTree::Leaf(a), TokenTree::Leaf(b)] => Some([a, b]),
            _ => None,
        })
    }

    pub fn crossed(&self) -> TokenTreesView<'a, Span> {
        assert!(self.is_root());
        TokenTreesView::new(&self.buffer[..self.index])
    }
}
