use crate::{subtree_parser::Parser, ParseError};

use ra_parser::FragmentKind;
use smallvec::{smallvec, SmallVec};

#[derive(Debug, Clone)]
pub(crate) struct TtCursor<'a> {
    subtree: &'a tt::Subtree,
    pos: usize,
}

pub(crate) struct TtCursorMemento {
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

    pub(crate) fn at_literal(&mut self) -> Option<&'a tt::Literal> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Literal(i))) => Some(i),
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

    pub(crate) fn eat_literal(&mut self) -> Option<&'a tt::Literal> {
        self.at_literal().map(|i| {
            self.bump();
            i
        })
    }

    pub(crate) fn eat_fragment(&mut self, fragment_kind: FragmentKind) -> Option<tt::TokenTree> {
        let parser = Parser::new(&mut self.pos, self.subtree);
        parser.parse_fragment(fragment_kind)
    }

    pub(crate) fn eat_lifetime(&mut self) -> Option<tt::TokenTree> {
        // check if it start from "`"
        if let Some(ident) = self.at_ident() {
            if ident.text.chars().next()? != '\'' {
                return None;
            }
        }

        self.eat_ident().cloned().map(|ident| tt::Leaf::from(ident).into())
    }

    pub(crate) fn try_eat_vis(&mut self) -> Option<tt::TokenTree> {
        // `vis` matcher is optional
        let old_pos = self.pos;
        let parser = Parser::new(&mut self.pos, self.subtree);

        let res = parser.parse_fragment(FragmentKind::Visibility);
        if res.is_none() {
            self.pos = old_pos;
        }
        res
    }

    pub(crate) fn expect_char(&mut self, char: char) -> Result<(), ParseError> {
        if self.at_char(char) {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::Expected(format!("`{}`", char)))
        }
    }

    fn eat_punct3(&mut self, p: tt::Punct) -> Option<SmallVec<[tt::Punct; 3]>> {
        let sec = *self.eat_punct()?;
        let third = *self.eat_punct()?;
        Some(smallvec![p, sec, third])
    }

    fn eat_punct2(&mut self, p: tt::Punct) -> Option<SmallVec<[tt::Punct; 3]>> {
        let sec = *self.eat_punct()?;
        Some(smallvec![p, sec])
    }

    fn eat_multi_char_punct<'b, I>(
        &mut self,
        p: tt::Punct,
        iter: &mut TokenPeek<'b, I>,
    ) -> Option<SmallVec<[tt::Punct; 3]>>
    where
        I: Iterator<Item = &'b tt::TokenTree>,
    {
        if let Some((m, _)) = iter.current_punct3(p) {
            if let r @ Some(_) = match m {
                ('<', '<', '=') | ('>', '>', '=') | ('.', '.', '.') | ('.', '.', '=') => {
                    self.eat_punct3(p)
                }
                _ => None,
            } {
                return r;
            }
        }

        if let Some((m, _)) = iter.current_punct2(p) {
            if let r @ Some(_) = match m {
                ('<', '=')
                | ('>', '=')
                | ('+', '=')
                | ('-', '=')
                | ('|', '=')
                | ('&', '=')
                | ('^', '=')
                | ('/', '=')
                | ('*', '=')
                | ('%', '=')
                | ('&', '&')
                | ('|', '|')
                | ('<', '<')
                | ('>', '>')
                | ('-', '>')
                | ('!', '=')
                | ('=', '>')
                | ('=', '=')
                | ('.', '.')
                | (':', ':') => self.eat_punct2(p),

                _ => None,
            } {
                return r;
            }
        }

        None
    }

    pub(crate) fn eat_seperator(&mut self) -> Option<crate::Separator> {
        match self.eat()? {
            tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => {
                Some(crate::Separator::Literal(lit.clone()))
            }
            tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                Some(crate::Separator::Ident(ident.clone()))
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => {
                match punct.char {
                    '*' | '+' | '?' => return None,
                    _ => {}
                };

                // FIXME: The parser is only handle some compositeable punct,
                // But at this phase, some punct still is jointed.
                // So we by pass that check here.
                let mut peekable = TokenPeek::new(self.subtree.token_trees[self.pos..].iter());
                let puncts = self.eat_multi_char_punct(*punct, &mut peekable);
                let puncts = puncts.unwrap_or_else(|| smallvec![*punct]);

                Some(crate::Separator::Puncts(puncts))
            }
            _ => None,
        }
    }

    #[must_use]
    pub(crate) fn save(&self) -> TtCursorMemento {
        TtCursorMemento { pos: self.pos }
    }

    pub(crate) fn rollback(&mut self, memento: TtCursorMemento) {
        self.pos = memento.pos;
    }
}

pub(crate) struct TokenPeek<'a, I>
where
    I: Iterator<Item = &'a tt::TokenTree>,
{
    iter: itertools::MultiPeek<I>,
}

// helper function
fn to_punct(tt: &tt::TokenTree) -> Option<&tt::Punct> {
    if let tt::TokenTree::Leaf(tt::Leaf::Punct(pp)) = tt {
        return Some(pp);
    }
    None
}

impl<'a, I> TokenPeek<'a, I>
where
    I: Iterator<Item = &'a tt::TokenTree>,
{
    pub fn new(iter: I) -> Self {
        TokenPeek { iter: itertools::multipeek(iter) }
    }

    pub fn current_punct2(&mut self, p: tt::Punct) -> Option<((char, char), bool)> {
        if p.spacing != tt::Spacing::Joint {
            return None;
        }

        self.iter.reset_peek();
        let p1 = to_punct(self.iter.peek()?)?;
        Some(((p.char, p1.char), p1.spacing == tt::Spacing::Joint))
    }

    pub fn current_punct3(&mut self, p: tt::Punct) -> Option<((char, char, char), bool)> {
        self.current_punct2(p).and_then(|((p0, p1), last_joint)| {
            if !last_joint {
                None
            } else {
                let p2 = to_punct(*self.iter.peek()?)?;
                Some(((p0, p1, p2.char), p2.spacing == tt::Spacing::Joint))
            }
        })
    }
}
