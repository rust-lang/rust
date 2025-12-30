//! Stateful iteration over token trees.
//!
//! We use this as the source of tokens for parser.
use crate::{Leaf, Subtree, TokenTree, TokenTreesView, dispatch_ref};

pub struct Cursor<'a> {
    buffer: TokenTreesView<'a>,
    index: usize,
    subtrees_stack: Vec<usize>,
}

impl<'a> Cursor<'a> {
    pub fn new(buffer: TokenTreesView<'a>) -> Self {
        Self { buffer, index: 0, subtrees_stack: Vec::new() }
    }

    /// Check whether it is eof
    pub fn eof(&self) -> bool {
        self.index == self.buffer.len() && self.subtrees_stack.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.subtrees_stack.is_empty()
    }

    fn at(&self, idx: usize) -> Option<TokenTree> {
        dispatch_ref! {
            match self.buffer.repr => tt => Some(tt.get(idx)?.to_api(self.buffer.span_parts))
        }
    }

    fn last_subtree(&self) -> Option<(usize, Subtree)> {
        self.subtrees_stack.last().map(|&subtree_idx| {
            let Some(TokenTree::Subtree(subtree)) = self.at(subtree_idx) else {
                panic!("subtree pointing to non-subtree");
            };
            (subtree_idx, subtree)
        })
    }

    pub fn end(&mut self) -> Subtree {
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
    pub fn token_tree(&self) -> Option<TokenTree> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            if last_subtree_idx + last_subtree.usize_len() + 1 == self.index {
                return None;
            }
        }
        self.at(self.index)
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
        if let Some(TokenTree::Subtree(_)) = self.at(self.index) {
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
        if let Some(TokenTree::Subtree(_)) = self.at(self.index) {
            self.subtrees_stack.push(self.index);
        }
        self.index += 1;
    }

    pub fn peek_two_leaves(&self) -> Option<[Leaf; 2]> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            let last_end = last_subtree_idx + last_subtree.usize_len() + 1;
            if last_end == self.index || last_end == self.index + 1 {
                return None;
            }
        }
        self.at(self.index).zip(self.at(self.index + 1)).and_then(|it| match it {
            (TokenTree::Leaf(a), TokenTree::Leaf(b)) => Some([a, b]),
            _ => None,
        })
    }

    pub fn crossed(&self) -> TokenTreesView<'a> {
        assert!(self.is_root());
        TokenTreesView {
            repr: self.buffer.repr.get(..self.index).unwrap(),
            span_parts: self.buffer.span_parts,
        }
    }
}
