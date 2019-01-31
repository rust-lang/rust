use crate::tt;

pub(crate) struct TtCursor<'a> {
    subtree: &'a tt::Subtree,
    pos: usize,
}

impl<'a> TtCursor<'a> {
    pub(crate) fn new(subtree: &'a tt::Subtree) -> TtCursor<'a> {
        TtCursor { subtree, pos: 0 }
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.pos == self.subtree.token_trees.len()
    }

    pub(crate) fn current(&self) -> Option<&'a tt::TokenTree> {
        self.subtree.token_trees.get(self.pos)
    }

    pub(crate) fn at_punct(&self) -> Option<&'a tt::Punct> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(it))) => Some(it),
            _ => None,
        }
    }

    pub(crate) fn at_char(&self, char: char) -> bool {
        match self.at_punct() {
            Some(tt::Punct { char: c }) if *c == char => true,
            _ => false,
        }
    }

    pub(crate) fn at_ident(&mut self) -> Option<&'a tt::Ident> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Ident(i))) => Some(i),
            _ => None,
        }
    }

    pub(crate) fn bump(&mut self) {
        self.pos += 1;
    }
    pub(crate) fn rev_bump(&mut self) {
        self.pos -= 1;
    }

    pub(crate) fn eat(&mut self) -> Option<&'a tt::TokenTree> {
        match self.current() {
            Some(it) => {
                self.bump();
                Some(it)
            }
            None => None,
        }
    }

    pub(crate) fn eat_subtree(&mut self) -> Option<&'a tt::Subtree> {
        match self.current()? {
            tt::TokenTree::Subtree(sub) => {
                self.bump();
                Some(sub)
            }
            _ => return None,
        }
    }

    pub(crate) fn eat_punct(&mut self) -> Option<&'a tt::Punct> {
        if let Some(it) = self.at_punct() {
            self.bump();
            return Some(it);
        }
        None
    }

    pub(crate) fn eat_ident(&mut self) -> Option<&'a tt::Ident> {
        if let Some(i) = self.at_ident() {
            self.bump();
            return Some(i);
        }
        None
    }

    pub(crate) fn expect_char(&mut self, char: char) -> Option<()> {
        if self.at_char(char) {
            self.bump();
            return Some(());
        }
        None
    }
}
