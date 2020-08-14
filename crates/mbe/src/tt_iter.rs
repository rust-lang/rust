//! FIXME: write short doc here

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
