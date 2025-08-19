//! A "Parser" structure for token trees. We use this when parsing a declarative
//! macro definition into a list of patterns and templates.

use std::fmt;

use arrayvec::ArrayVec;
use intern::sym;

use crate::{Ident, Leaf, MAX_GLUED_PUNCT_LEN, Punct, Spacing, Subtree, TokenTree, TokenTreesView};

#[derive(Clone)]
pub struct TtIter<'a, S> {
    inner: std::slice::Iter<'a, TokenTree<S>>,
}

impl<S: Copy + fmt::Debug> fmt::Debug for TtIter<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TtIter").field("remaining", &self.remaining()).finish()
    }
}

#[derive(Clone, Copy)]
pub struct TtIterSavepoint<'a, S>(&'a [TokenTree<S>]);

impl<'a, S: Copy> TtIterSavepoint<'a, S> {
    pub fn remaining(self) -> TokenTreesView<'a, S> {
        TokenTreesView::new(self.0)
    }
}

impl<'a, S: Copy> TtIter<'a, S> {
    pub(crate) fn new(tt: &'a [TokenTree<S>]) -> TtIter<'a, S> {
        TtIter { inner: tt.iter() }
    }

    pub fn expect_char(&mut self, char: char) -> Result<(), ()> {
        match self.next() {
            Some(TtElement::Leaf(&Leaf::Punct(Punct { char: c, .. }))) if c == char => Ok(()),
            _ => Err(()),
        }
    }

    pub fn expect_any_char(&mut self, chars: &[char]) -> Result<(), ()> {
        match self.next() {
            Some(TtElement::Leaf(Leaf::Punct(Punct { char: c, .. }))) if chars.contains(c) => {
                Ok(())
            }
            _ => Err(()),
        }
    }

    pub fn expect_subtree(&mut self) -> Result<(&'a Subtree<S>, TtIter<'a, S>), ()> {
        match self.next() {
            Some(TtElement::Subtree(subtree, iter)) => Ok((subtree, iter)),
            _ => Err(()),
        }
    }

    pub fn expect_leaf(&mut self) -> Result<&'a Leaf<S>, ()> {
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

    pub fn expect_ident(&mut self) -> Result<&'a Ident<S>, ()> {
        match self.expect_leaf()? {
            Leaf::Ident(it) if it.sym != sym::underscore => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_ident_or_underscore(&mut self) -> Result<&'a Ident<S>, ()> {
        match self.expect_leaf()? {
            Leaf::Ident(it) => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_literal(&mut self) -> Result<&'a Leaf<S>, ()> {
        let it = self.expect_leaf()?;
        match it {
            Leaf::Literal(_) => Ok(it),
            Leaf::Ident(ident) if ident.sym == sym::true_ || ident.sym == sym::false_ => Ok(it),
            _ => Err(()),
        }
    }

    pub fn expect_single_punct(&mut self) -> Result<&'a Punct<S>, ()> {
        match self.expect_leaf()? {
            Leaf::Punct(it) => Ok(it),
            _ => Err(()),
        }
    }

    /// Returns consecutive `Punct`s that can be glued together.
    ///
    /// This method currently may return a single quotation, which is part of lifetime ident and
    /// conceptually not a punct in the context of mbe. Callers should handle this.
    pub fn expect_glued_punct(&mut self) -> Result<ArrayVec<Punct<S>, MAX_GLUED_PUNCT_LEN>, ()> {
        let TtElement::Leaf(&Leaf::Punct(first)) = self.next().ok_or(())? else {
            return Err(());
        };

        let mut res = ArrayVec::new();
        if first.spacing == Spacing::Alone {
            res.push(first);
            return Ok(res);
        }

        let (second, third) = match (self.peek_n(0), self.peek_n(1)) {
            (Some(TokenTree::Leaf(Leaf::Punct(p2))), Some(TokenTree::Leaf(Leaf::Punct(p3))))
                if p2.spacing == Spacing::Joint =>
            {
                (p2, Some(p3))
            }
            (Some(TokenTree::Leaf(Leaf::Punct(p2))), _) => (p2, None),
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
                res.push(*second);
                res.push(*third.unwrap());
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
                res.push(*second);
            }
            _ => res.push(first),
        }
        Ok(res)
    }

    /// This method won't check for subtrees, so the nth token tree may not be the nth sibling of the current tree.
    fn peek_n(&self, n: usize) -> Option<&'a TokenTree<S>> {
        self.inner.as_slice().get(n)
    }

    pub fn peek(&self) -> Option<TtElement<'a, S>> {
        match self.inner.as_slice().first()? {
            TokenTree::Leaf(leaf) => Some(TtElement::Leaf(leaf)),
            TokenTree::Subtree(subtree) => {
                let nested_iter =
                    TtIter { inner: self.inner.as_slice()[1..][..subtree.usize_len()].iter() };
                Some(TtElement::Subtree(subtree, nested_iter))
            }
        }
    }

    /// Equivalent to `peek().is_none()`, but a bit faster.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn next_span(&self) -> Option<S> {
        Some(self.inner.as_slice().first()?.first_span())
    }

    pub fn remaining(&self) -> TokenTreesView<'a, S> {
        TokenTreesView::new(self.inner.as_slice())
    }

    /// **Warning**: This advances `skip` **flat** token trees, subtrees account for children+1!
    pub fn flat_advance(&mut self, skip: usize) {
        self.inner = self.inner.as_slice()[skip..].iter();
    }

    pub fn savepoint(&self) -> TtIterSavepoint<'a, S> {
        TtIterSavepoint(self.inner.as_slice())
    }

    pub fn from_savepoint(&self, savepoint: TtIterSavepoint<'a, S>) -> TokenTreesView<'a, S> {
        let len = (self.inner.as_slice().as_ptr() as usize - savepoint.0.as_ptr() as usize)
            / size_of::<TokenTree<S>>();
        TokenTreesView::new(&savepoint.0[..len])
    }

    pub fn next_as_view(&mut self) -> Option<TokenTreesView<'a, S>> {
        let savepoint = self.savepoint();
        self.next()?;
        Some(self.from_savepoint(savepoint))
    }
}

#[derive(Clone)]
pub enum TtElement<'a, S> {
    Leaf(&'a Leaf<S>),
    Subtree(&'a Subtree<S>, TtIter<'a, S>),
}

impl<S: Copy + fmt::Debug> fmt::Debug for TtElement<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(leaf) => f.debug_tuple("Leaf").field(leaf).finish(),
            Self::Subtree(subtree, inner) => {
                f.debug_tuple("Subtree").field(subtree).field(inner).finish()
            }
        }
    }
}

impl<S: Copy> TtElement<'_, S> {
    #[inline]
    pub fn first_span(&self) -> S {
        match self {
            TtElement::Leaf(it) => *it.span(),
            TtElement::Subtree(it, _) => it.delimiter.open,
        }
    }
}

impl<'a, S> Iterator for TtIter<'a, S> {
    type Item = TtElement<'a, S>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next()? {
            TokenTree::Leaf(leaf) => Some(TtElement::Leaf(leaf)),
            TokenTree::Subtree(subtree) => {
                let nested_iter =
                    TtIter { inner: self.inner.as_slice()[..subtree.usize_len()].iter() };
                self.inner = self.inner.as_slice()[subtree.usize_len()..].iter();
                Some(TtElement::Subtree(subtree, nested_iter))
            }
        }
    }
}
