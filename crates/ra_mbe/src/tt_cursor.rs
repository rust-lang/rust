use crate::ParseError;

#[derive(Clone)]
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
            Some(tt::Punct { char: c, .. }) if *c == char => true,
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
        self.current().map(|it| {
            self.bump();
            it
        })
    }

    pub(crate) fn eat_subtree(&mut self) -> Result<&'a tt::Subtree, ParseError> {
        match self.current() {
            Some(tt::TokenTree::Subtree(sub)) => {
                self.bump();
                Ok(sub)
            }
            _ => Err(ParseError::Expected(String::from("subtree"))),
        }
    }

    pub(crate) fn eat_punct(&mut self) -> Option<&'a tt::Punct> {
        self.at_punct().map(|it| {
            self.bump();
            it
        })
    }

    pub(crate) fn eat_ident(&mut self) -> Option<&'a tt::Ident> {
        self.at_ident().map(|i| {
            self.bump();
            i
        })
    }

    pub(crate) fn expect_char(&mut self, char: char) -> Result<(), ParseError> {
        if self.at_char(char) {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::Expected(format!("`{}`", char)))
        }
    }
}
