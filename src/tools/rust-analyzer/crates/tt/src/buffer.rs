//! Stateful iteration over token trees.
//!
//! We use this as the source of tokens for parser.
use crate::{Leaf, Subtree, TokenTree, TokenTreesView, storage::TokenTreesSlice};

pub struct Cursor<'a> {
    origin: TokenTreesSlice<'a>,
    buffer_before_current: TokenTreesSlice<'a>,
    buffer_after_current: TokenTreesSlice<'a>,
    /// The number of times we called [`Self::advance()`]. Also the index of [`Self::next`].
    advances_count: usize,
    len: usize,
    next: Option<TokenTree>,
    subtrees_stack: Vec<(usize, Subtree)>,
}

impl<'a> Cursor<'a> {
    pub fn new(origin: TokenTreesView<'a>) -> Self {
        let mut buffer_after_current = origin.slice;
        let buffer_before_current = buffer_after_current;
        let len = origin.len;
        let next = if len >= 1 { buffer_after_current.advance() } else { None };
        Self {
            origin: origin.slice,
            buffer_after_current,
            buffer_before_current,
            advances_count: 0,
            len,
            next,
            subtrees_stack: Vec::new(),
        }
    }

    /// Check whether it is eof
    pub fn eof(&self) -> bool {
        self.next.is_none() && self.subtrees_stack.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.subtrees_stack.is_empty()
    }

    fn last_subtree(&self) -> Option<(usize, Subtree)> {
        self.subtrees_stack.last().copied()
    }

    pub(crate) fn remaining(&self) -> TokenTreesView<'a> {
        TokenTreesView { slice: self.buffer_before_current, len: self.len - self.advances_count }
    }

    pub fn end(&mut self) -> Subtree {
        let (last_subtree_idx, last_subtree) =
            self.last_subtree().expect("called `Cursor::end()` without an open subtree");
        // +1 because `Subtree.len` excludes the subtree itself.
        assert_eq!(
            last_subtree_idx + last_subtree.usize_len() + 1,
            self.advances_count,
            "called `Cursor::end()` without finishing a subtree"
        );
        self.subtrees_stack.pop();
        last_subtree
    }

    /// Returns the `TokenTree` at the cursor if it is not at the end of a subtree.
    pub fn token_tree(&self) -> Option<TokenTree> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            if last_subtree_idx + last_subtree.usize_len() + 1 == self.advances_count {
                return None;
            }
        }
        self.next.clone()
    }

    fn advance(&mut self) {
        if self.advances_count >= self.len {
            return;
        }
        if let Some(TokenTree::Subtree(subtree)) = self.next {
            self.subtrees_stack.push((self.advances_count, subtree));
        }
        self.advances_count += 1;
        self.buffer_before_current = self.buffer_after_current;
        self.next =
            if self.advances_count < self.len { self.buffer_after_current.advance() } else { None };
    }

    /// Bump the cursor, and enters a subtree if it is on one.
    pub fn bump(&mut self) {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            assert_ne!(
                last_subtree_idx + last_subtree.usize_len() + 1,
                self.advances_count,
                "called `Cursor::bump()` when at the end of a subtree"
            );
        }
        self.advance();
    }

    pub fn bump_or_end(&mut self) {
        // +1 because `Subtree.len` excludes the subtree itself.
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree()
            && last_subtree_idx + last_subtree.usize_len() + 1 == self.advances_count
        {
            self.subtrees_stack.pop();
            return;
        }
        self.advance();
    }

    pub fn peek_two_leaves(&self) -> Option<[Leaf; 2]> {
        if let Some((last_subtree_idx, last_subtree)) = self.last_subtree() {
            // +1 because `Subtree.len` excludes the subtree itself.
            let last_end = last_subtree_idx + last_subtree.usize_len() + 1;
            if last_end == self.advances_count || last_end == self.advances_count + 1 {
                return None;
            }
        }
        let mut buffer = self.buffer_after_current;
        let next_next = if self.advances_count + 1 < self.len { buffer.advance() } else { None };
        self.next.clone().zip(next_next).and_then(|it| match it {
            (TokenTree::Leaf(a), TokenTree::Leaf(b)) => Some([a, b]),
            _ => None,
        })
    }

    pub fn crossed(&self) -> TokenTreesView<'a> {
        assert!(self.is_root());
        TokenTreesView { slice: self.origin, len: self.advances_count }
    }
}
