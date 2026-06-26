//! A "Parser" structure for token trees. We use this when parsing a declarative
//! macro definition into a list of patterns and templates.

use std::fmt;

use arrayvec::ArrayVec;
use intern::sym;
use span::Span;

use crate::{
    Ident, Leaf, MAX_GLUED_PUNCT_LEN, Punct, Spacing, Subtree, TokenTree, TokenTreesView,
    buffer::Cursor,
};

#[derive(Clone)]
pub struct TtIter<'a> {
    inner: TokenTreesView<'a>,
}

impl fmt::Debug for TtIter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TtIter").field("remaining", &self.remaining()).finish()
    }
}

#[derive(Clone, Copy)]
pub struct TtIterSavepoint<'a>(TokenTreesView<'a>);

impl<'a> TtIterSavepoint<'a> {
    pub fn remaining(self) -> TokenTreesView<'a> {
        self.0
    }
}

impl<'a> TtIter<'a> {
    pub(crate) fn new(tt: TokenTreesView<'a>) -> TtIter<'a> {
        TtIter { inner: tt }
    }

    pub fn expect_char(&mut self, char: char) -> Result<(), ()> {
        match self.next() {
            Some(TtElement::Leaf(Leaf::Punct(Punct { char: c, .. }))) if c == char => Ok(()),
            _ => Err(()),
        }
    }

    pub fn expect_any_char(&mut self, chars: &[char]) -> Result<(), ()> {
        match self.next() {
            Some(TtElement::Leaf(Leaf::Punct(Punct { char: c, .. }))) if chars.contains(&c) => {
                Ok(())
            }
            _ => Err(()),
        }
    }

    pub fn expect_subtree(&mut self) -> Result<(Subtree, TtIter<'a>), ()> {
        match self.next() {
            Some(TtElement::Subtree(subtree, iter)) => Ok((subtree, iter)),
            _ => Err(()),
        }
    }

    pub fn expect_leaf(&mut self) -> Result<Leaf, ()> {
        match self.next() {
            Some(TtElement::Leaf(it)) => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_dollar(&mut self) -> Result<(), ()> {
        match self.expect_leaf()? {
            Leaf::Punct(Punct { char: '$', .. }) => Ok(()),
            _ => Err(()),
        }
    }

    pub fn expect_comma(&mut self) -> Result<(), ()> {
        match self.expect_leaf()? {
            Leaf::Punct(Punct { char: ',', .. }) => Ok(()),
            _ => Err(()),
        }
    }

    pub fn expect_ident(&mut self) -> Result<Ident, ()> {
        match self.expect_leaf()? {
            Leaf::Ident(it) if it.sym != sym::underscore => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_ident_or_underscore(&mut self) -> Result<Ident, ()> {
        match self.expect_leaf()? {
            Leaf::Ident(it) => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_literal(&mut self) -> Result<Leaf, ()> {
        let it = self.expect_leaf()?;
        match &it {
            Leaf::Literal(_) => Ok(it),
            Leaf::Ident(ident) if ident.sym == sym::true_ || ident.sym == sym::false_ => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_single_punct(&mut self) -> Result<Punct, ()> {
        match self.expect_leaf()? {
            Leaf::Punct(it) => Ok(it),
            _ => Err(()),
        }
    }

    /// Returns consecutive `Punct`s that can be glued together.
    ///
    /// This method currently may return a single quotation, which is part of lifetime ident and
    /// conceptually not a punct in the context of mbe. Callers should handle this.
    pub fn expect_glued_punct(&mut self) -> Result<ArrayVec<Punct, MAX_GLUED_PUNCT_LEN>, ()> {
        let TtElement::Leaf(Leaf::Punct(first)) = self.next().ok_or(())? else {
            return Err(());
        };

        let mut res = ArrayVec::new();
        if first.spacing == Spacing::Alone {
            res.push(first);
            return Ok(res);
        }

        let (second, third) = match self.peek_two() {
            [Some(TokenTree::Leaf(Leaf::Punct(p2))), Some(TokenTree::Leaf(Leaf::Punct(p3)))]
                if p2.spacing == Spacing::Joint =>
            {
                (p2, Some(p3))
            }
            [Some(TokenTree::Leaf(Leaf::Punct(p2))), _] => (p2, None),
            _ => {
                res.push(first);
                return Ok(res);
            }
        };

        match (first.char, second.char, third.map(|it| it.char)) {
            ('.', '.', Some('.' | '=')) | ('<', '<', Some('=')) | ('>', '>', Some('=')) => {
                let _ = self.next().unwrap();
                let _ = self.next().unwrap();
                res.push(first);
                res.push(second);
                res.push(third.unwrap());
            }
            ('-' | '!' | '*' | '/' | '&' | '%' | '^' | '+' | '<' | '=' | '>' | '|', '=', _)
            | ('-' | '=' | '>', '>', _)
            | ('<', '-', _)
            | (':', ':', _)
            | ('.', '.', _)
            | ('&', '&', _)
            | ('<', '<', _)
            | ('|', '|', _) => {
                let _ = self.next().unwrap();
                res.push(first);
                res.push(second);
            }
            _ => res.push(first),
        }
        Ok(res)
    }

    /// This method won't check for subtrees, so the nth token tree may not be the nth sibling of the current tree.
    fn peek_two(&self) -> [Option<TokenTree>; 2] {
        let mut iter = self.inner.iter_flat_tokens();
        [iter.next(), iter.next()]
    }

    pub fn peek(&self) -> Option<TtElement<'a>> {
        if self.inner.is_empty() {
            return None;
        }
        let mut slice = self.inner.slice;
        match slice.advance()? {
            TokenTree::Leaf(leaf) => Some(TtElement::Leaf(leaf)),
            TokenTree::Subtree(subtree) => {
                let nested_iter =
                    TtIter { inner: TokenTreesView { len: subtree.usize_len(), slice } };
                Some(TtElement::Subtree(subtree, nested_iter))
            }
        }
    }

    /// Equivalent to `peek().is_none()`, but a bit faster.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn next_span(&self) -> Option<Span> {
        Some(self.peek()?.first_span())
    }

    pub fn remaining(&self) -> TokenTreesView<'a> {
        self.inner
    }

    /// **Warning**: This advances **flat** token trees, subtrees account for children+1!
    pub fn flat_advance_to(&mut self, up_to: &Cursor<'a>) {
        self.inner = up_to.remaining();
    }

    pub fn savepoint(&self) -> TtIterSavepoint<'a> {
        TtIterSavepoint(self.inner)
    }

    pub fn from_savepoint(&self, savepoint: TtIterSavepoint<'a>) -> TokenTreesView<'a> {
        TokenTreesView { slice: savepoint.0.slice, len: savepoint.0.len - self.inner.len }
    }

    pub fn next_as_view(&mut self) -> Option<TokenTreesView<'a>> {
        let savepoint = self.savepoint();
        self.next()?;
        Some(self.from_savepoint(savepoint))
    }
}

#[derive(Clone)]
pub enum TtElement<'a> {
    Leaf(Leaf),
    Subtree(Subtree, TtIter<'a>),
}

impl fmt::Debug for TtElement<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(leaf) => f.debug_tuple("Leaf").field(leaf).finish(),
            Self::Subtree(subtree, inner) => {
                f.debug_tuple("Subtree").field(subtree).field(inner).finish()
            }
        }
    }
}

impl TtElement<'_> {
    #[inline]
    pub fn first_span(&self) -> Span {
        match self {
            TtElement::Leaf(it) => *it.span(),
            TtElement::Subtree(it, _) => it.delimiter.open,
        }
    }
}

impl<'a> Iterator for TtIter<'a> {
    type Item = TtElement<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.is_empty() {
            return None;
        }
        self.inner.len -= 1;
        let (tt, subtree_slice) = self.inner.slice.advance_skip_subtree()?;
        match tt {
            TokenTree::Leaf(leaf) => Some(TtElement::Leaf(leaf)),
            TokenTree::Subtree(subtree) => {
                self.inner.len -= subtree.usize_len();
                let nested_iter = TtIter {
                    inner: TokenTreesView { len: subtree.usize_len(), slice: subtree_slice },
                };
                Some(TtElement::Subtree(subtree, nested_iter))
            }
        }
    }
}
