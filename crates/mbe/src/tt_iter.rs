//! FIXME: write short doc here

use crate::{subtree_source::SubtreeTokenSource, ExpandError, ExpandResult};

use parser::TreeSink;
use syntax::SyntaxKind;
use tt::buffer::{Cursor, TokenBuffer};

macro_rules! err {
    () => {
        ExpandError::BindingError(format!(""))
    };
    ($($tt:tt)*) => {
        ExpandError::BindingError(format!($($tt)*))
    };
}

#[derive(Debug, Clone)]
pub(crate) struct TtIter<'a> {
    pub(crate) inner: std::slice::Iter<'a, tt::TokenTree>,
}

impl<'a> TtIter<'a> {
    pub(crate) fn new(subtree: &'a tt::Subtree) -> TtIter<'a> {
        TtIter { inner: subtree.token_trees.iter() }
    }

    pub(crate) fn expect_char(&mut self, char: char) -> Result<(), ()> {
        match self.next() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: c, .. }))) if *c == char => {
                Ok(())
            }
            _ => Err(()),
        }
    }

    pub(crate) fn expect_subtree(&mut self) -> Result<&'a tt::Subtree, ()> {
        match self.next() {
            Some(tt::TokenTree::Subtree(it)) => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_leaf(&mut self) -> Result<&'a tt::Leaf, ()> {
        match self.next() {
            Some(tt::TokenTree::Leaf(it)) => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_ident(&mut self) -> Result<&'a tt::Ident, ()> {
        match self.expect_leaf()? {
            tt::Leaf::Ident(it) => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_literal(&mut self) -> Result<&'a tt::Leaf, ()> {
        let it = self.expect_leaf()?;
        match it {
            tt::Leaf::Literal(_) => Ok(it),
            tt::Leaf::Ident(ident) if ident.text == "true" || ident.text == "false" => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_punct(&mut self) -> Result<&'a tt::Punct, ()> {
        match self.expect_leaf()? {
            tt::Leaf::Punct(it) => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_fragment(
        &mut self,
        fragment_kind: parser::FragmentKind,
    ) -> ExpandResult<Option<tt::TokenTree>> {
        struct OffsetTokenSink<'a> {
            cursor: Cursor<'a>,
            error: bool,
        }

        impl<'a> TreeSink for OffsetTokenSink<'a> {
            fn token(&mut self, kind: SyntaxKind, mut n_tokens: u8) {
                if kind == SyntaxKind::LIFETIME_IDENT {
                    n_tokens = 2;
                }
                for _ in 0..n_tokens {
                    self.cursor = self.cursor.bump_subtree();
                }
            }
            fn start_node(&mut self, _kind: SyntaxKind) {}
            fn finish_node(&mut self) {}
            fn error(&mut self, _error: parser::ParseError) {
                self.error = true;
            }
        }

        let buffer = TokenBuffer::from_tokens(&self.inner.as_slice());
        let mut src = SubtreeTokenSource::new(&buffer);
        let mut sink = OffsetTokenSink { cursor: buffer.begin(), error: false };

        parser::parse_fragment(&mut src, &mut sink, fragment_kind);

        let mut err = None;
        if !sink.cursor.is_root() || sink.error {
            err = Some(err!("expected {:?}", fragment_kind));
        }

        let mut curr = buffer.begin();
        let mut res = vec![];

        if sink.cursor.is_root() {
            while curr != sink.cursor {
                if let Some(token) = curr.token_tree() {
                    res.push(token);
                }
                curr = curr.bump();
            }
        }
        self.inner = self.inner.as_slice()[res.len()..].iter();
        if res.len() == 0 && err.is_none() {
            err = Some(err!("no tokens consumed"));
        }
        let res = match res.len() {
            1 => Some(res[0].cloned()),
            0 => None,
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: None,
                token_trees: res.into_iter().map(|it| it.cloned()).collect(),
            })),
        };
        ExpandResult { value: res, err }
    }

    pub(crate) fn peek_n(&self, n: usize) -> Option<&tt::TokenTree> {
        self.inner.as_slice().get(n)
    }
}

impl<'a> Iterator for TtIter<'a> {
    type Item = &'a tt::TokenTree;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> std::iter::ExactSizeIterator for TtIter<'a> {}
