//! A "Parser" structure for token trees. We use this when parsing a declarative
//! macro definition into a list of patterns and templates.

use smallvec::{smallvec, SmallVec};
use syntax::SyntaxKind;

use crate::{to_parser_input::to_parser_input, tt, ExpandError, ExpandResult};

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
            Some(&tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: c, .. }))) if c == char => {
                Ok(())
            }
            _ => Err(()),
        }
    }

    pub(crate) fn expect_any_char(&mut self, chars: &[char]) -> Result<(), ()> {
        match self.next() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: c, .. })))
                if chars.contains(c) =>
            {
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
            tt::Leaf::Ident(it) if it.text != "_" => Ok(it),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_ident_or_underscore(&mut self) -> Result<&'a tt::Ident, ()> {
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

    pub(crate) fn expect_u32_literal(&mut self) -> Result<u32, ()> {
        match self.expect_literal()? {
            tt::Leaf::Literal(lit) => lit.text.parse().map_err(drop),
            _ => Err(()),
        }
    }

    pub(crate) fn expect_single_punct(&mut self) -> Result<&'a tt::Punct, ()> {
        match self.expect_leaf()? {
            tt::Leaf::Punct(it) => Ok(it),
            _ => Err(()),
        }
    }

    /// Returns consecutive `Punct`s that can be glued together.
    ///
    /// This method currently may return a single quotation, which is part of lifetime ident and
    /// conceptually not a punct in the context of mbe. Callers should handle this.
    pub(crate) fn expect_glued_punct(&mut self) -> Result<SmallVec<[tt::Punct; 3]>, ()> {
        let tt::TokenTree::Leaf(tt::Leaf::Punct(first)) = self.next().ok_or(())?.clone() else {
            return Err(());
        };

        if first.spacing == tt::Spacing::Alone {
            return Ok(smallvec![first]);
        }

        let (second, third) = match (self.peek_n(0), self.peek_n(1)) {
            (
                Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p2))),
                Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p3))),
            ) if p2.spacing == tt::Spacing::Joint => (p2, Some(p3)),
            (Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p2))), _) => (p2, None),
            _ => return Ok(smallvec![first]),
        };

        match (first.char, second.char, third.map(|it| it.char)) {
            ('.', '.', Some('.' | '=')) | ('<', '<', Some('=')) | ('>', '>', Some('=')) => {
                let _ = self.next().unwrap();
                let _ = self.next().unwrap();
                Ok(smallvec![first, *second, *third.unwrap()])
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
                Ok(smallvec![first, *second])
            }
            _ => Ok(smallvec![first]),
        }
    }

    pub(crate) fn expect_fragment(
        &mut self,
        entry_point: parser::PrefixEntryPoint,
    ) -> ExpandResult<Option<tt::TokenTree>> {
        let buffer = tt::buffer::TokenBuffer::from_tokens(self.inner.as_slice());
        let parser_input = to_parser_input(&buffer);
        let tree_traversal = entry_point.parse(&parser_input);

        let mut cursor = buffer.begin();
        let mut error = false;
        for step in tree_traversal.iter() {
            match step {
                parser::Step::Token { kind, mut n_input_tokens } => {
                    if kind == SyntaxKind::LIFETIME_IDENT {
                        n_input_tokens = 2;
                    }
                    for _ in 0..n_input_tokens {
                        cursor = cursor.bump_subtree();
                    }
                }
                parser::Step::FloatSplit { .. } => {
                    // FIXME: We need to split the tree properly here, but mutating the token trees
                    // in the buffer is somewhat tricky to pull off.
                    cursor = cursor.bump_subtree();
                }
                parser::Step::Enter { .. } | parser::Step::Exit => (),
                parser::Step::Error { .. } => error = true,
            }
        }

        let err = if error || !cursor.is_root() {
            Some(ExpandError::binding_error(format!("expected {entry_point:?}")))
        } else {
            None
        };

        let mut curr = buffer.begin();
        let mut res = vec![];

        if cursor.is_root() {
            while curr != cursor {
                let Some(token) = curr.token_tree() else { break };
                res.push(token.cloned());
                curr = curr.bump();
            }
        }

        self.inner = self.inner.as_slice()[res.len()..].iter();
        let res = match res.len() {
            0 | 1 => res.pop(),
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter::unspecified(),
                token_trees: res,
            })),
        };
        ExpandResult { value: res, err }
    }

    pub(crate) fn peek_n(&self, n: usize) -> Option<&'a tt::TokenTree> {
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
